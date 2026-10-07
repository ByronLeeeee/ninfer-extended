#pragma once

#include "ops/softmax_attention/dense/context/causal_prefill_kernel.cuh"

namespace ninfer::ops::detail {

template <bool Packed>
__launch_bounds__(64, 4) __global__ void causal_attention_wide_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, std::int32_t tokens, std::int32_t heads,
    std::int32_t kv_heads, __nv_bfloat16* __restrict__ out,
    std::int64_t q_stride_d, std::int64_t q_stride_h, std::int64_t q_stride_t,
    std::int64_t k_stride_d, std::int64_t k_stride_h, std::int64_t k_stride_t,
    std::int64_t v_stride_d, std::int64_t v_stride_h, std::int64_t v_stride_t,
    const std::int32_t* sequence_begin = nullptr,
    const std::int32_t* sequence_length = nullptr) {
    constexpr int Br = 32, Bc = 64, ReductionBc = 32;
    constexpr int D             = kCausalAttentionHeadDim;
    constexpr int Dp            = kCausalAttentionPaddedD;
    constexpr int Threads       = Br * 2;
    constexpr int QKNt          = Bc / 8;
    constexpr int QKKs          = D / 16;
    constexpr int PVNt          = D / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr int ReductionNt   = ReductionBc / 8;
    constexpr int ProbabilityGroups = Bc / ReductionBc;
    constexpr int PVGroupKs     = ReductionBc / 16;
    constexpr int RowBytes      = Dp * static_cast<int>(sizeof(__nv_bfloat16));
    constexpr float ScaleLog2E  = 0.0883883476483184406f * 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    int begin = 0, end = tokens;
    if constexpr (Packed) {
        begin = sequence_begin[blockIdx.z];
        end = begin + sequence_length[blockIdx.z];
    }
    const int query_begin = begin + static_cast<int>(blockIdx.x) * Br;
    if (query_begin >= end) return;
    const CausalAttentionTile tile{query_begin, begin, min(end, query_begin + Br), 0};
    const int head = static_cast<int>(blockIdx.y);
    const int kv_head=head/(heads/kv_heads);
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    extern __shared__ __align__(16) __nv_bfloat16 shared[];
    __nv_bfloat16* q_s = shared;
    __nv_bfloat16* k_s = q_s + Br * Dp;
    __nv_bfloat16* v_s = k_s + Bc * Dp;

    const int gid       = lane >> 2;
    const int lid       = lane & 3;
    const int a_mat     = lane >> 3;
    const int a_rin     = lane & 7;
    const int a_rowoff  = a_rin + ((a_mat & 1) << 3);
    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_row0 = warp * 16;

    const unsigned q_sbase     = smem_addr(q_s);
    const unsigned k_sbase     = smem_addr(k_s);
    const unsigned v_sbase     = smem_addr(v_s);
    const unsigned q_lane_base = q_sbase + static_cast<unsigned>((warp_row0 + a_rowoff) * RowBytes);
    const unsigned q_as        = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r         = static_cast<unsigned>(a_rin << 4);
    const unsigned k_lane_base = k_sbase + static_cast<unsigned>(b_rin * RowBytes) +
                                 static_cast<unsigned>((lane >> 4) * 8 * RowBytes);
    const unsigned k_as        = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r         = static_cast<unsigned>(b_rin << 4);
    const unsigned v_lane_base = v_sbase + static_cast<unsigned>(((lane >> 3) & 1) * 8 * RowBytes) +
                                 static_cast<unsigned>(b_rin * RowBytes);
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    causal_attention_stage_q<Br, Threads>(q_s, q, tile.q0, tile.end, head, tid, q_stride_d,
                                          q_stride_h, q_stride_t);

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int item = 0; item < 4; ++item) { acc[n][item] = 0.0f; }
    }
    float m0 = -CUDART_INF_F;
    float m1 = -CUDART_INF_F;
    float l0 = 0.0f;
    float l1 = 0.0f;

    cp_commit();
    causal_attention_stage_kv<Bc, Threads>(k_s, k, tile.begin, tile.end, kv_head, tid, k_stride_d,
                                           k_stride_h, k_stride_t);
    cp_commit();

    const int key_blocks = (tile.end - tile.begin + Bc - 1) / Bc;
    for (int kb = 0; kb < key_blocks; ++kb) {
        const int key0 = tile.begin + kb * Bc;
        cp_wait<0>();
        __syncthreads();

        causal_attention_stage_kv<Bc, Threads>(v_s, v, key0, tile.end, kv_head, tid, v_stride_d,
                                               v_stride_h, v_stride_t);
        cp_commit();

        unsigned p_frag[PVKs][4];
        unsigned p_residual[PVKs][4];
        float group_alpha0[ProbabilityGroups];
        float group_alpha1[ProbabilityGroups];
        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
        unsigned af[2][4];
        unsigned bf[2][QKNt][2];
        ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                    causal_attention_swz_addr(q_lane_base, 0u, q_as, q_r));
#pragma unroll
        for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
            ldmatrix_x4(
                bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                causal_attention_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 8 * RowBytes),
                                          0u, k_as, k_r));
        }
#pragma unroll
        for (int ks = 0; ks < QKKs; ++ks) {
            const int cur = ks & 1;
            const int nxt = cur ^ 1;
            if (ks + 1 < QKKs) {
                const unsigned ck = static_cast<unsigned>((ks + 1) << 5);
                ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                            causal_attention_swz_addr(q_lane_base, ck, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0],
                                bf[nxt][nt2 + 1][1],
                                causal_attention_swz_addr(
                                    k_lane_base + static_cast<unsigned>(nt2 * 8 * RowBytes), ck,
                                    k_as, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                         af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
            }
        }

        const int row0       = warp_row0 + gid;
        const int row1       = row0 + 8;
        const int query0     = tile.q0 + row0;
        const int query1     = tile.q0 + row1;
        // Transfer a wider key tile while retaining independent softmax updates
        // in ReductionBc-column groups and their original FP32 reduction order.
#pragma unroll
        for (int group = 0; group < ProbabilityGroups; ++group) {
            const bool probability_active = ProbabilityGroups == 1 ||
                key0 + group * ReductionBc < tile.end;
            if (probability_active) {
                float block_max0=-CUDART_INF_F,block_max1=-CUDART_INF_F;
#pragma unroll
                for(int nt=group*ReductionNt;nt<(group+1)*ReductionNt;++nt){
                    const int key_a=key0+nt*8+2*lid,key_b=key_a+1;
                    score[nt][0]=query0<tile.end&&key_a<=query0&&key_a<tile.end?score[nt][0]:-CUDART_INF_F;
                    score[nt][1]=query0<tile.end&&key_b<=query0&&key_b<tile.end?score[nt][1]:-CUDART_INF_F;
                    score[nt][2]=query1<tile.end&&key_a<=query1&&key_a<tile.end?score[nt][2]:-CUDART_INF_F;
                    score[nt][3]=query1<tile.end&&key_b<=query1&&key_b<tile.end?score[nt][3]:-CUDART_INF_F;
                    block_max0=fmaxf(block_max0,fmaxf(score[nt][0],score[nt][1]));
                    block_max1=fmaxf(block_max1,fmaxf(score[nt][2],score[nt][3]));
                }
                block_max0 = warp_max<4>(block_max0, FullMask);
                block_max1 = warp_max<4>(block_max1, FullMask);

                const float next_m0 = fmaxf(m0, block_max0);
                const float next_m1 = fmaxf(m1, block_max1);
                const float m0_l2   = next_m0 * ScaleLog2E;
                const float m1_l2   = next_m1 * ScaleLog2E;
                const float alpha0  = exp2_approx(__fmaf_rn(m0, ScaleLog2E, -m0_l2));
                const float alpha1  = exp2_approx(__fmaf_rn(m1, ScaleLog2E, -m1_l2));
                if constexpr (ProbabilityGroups == 1) {
#pragma unroll
                    for (int n = 0; n < PVNt; ++n) {
                        acc[n][0] *= alpha0;
                        acc[n][1] *= alpha0;
                        acc[n][2] *= alpha1;
                        acc[n][3] *= alpha1;
                    }
                } else {
                    group_alpha0[group] = alpha0;
                    group_alpha1[group] = alpha1;
                }

                float block_sum0 = 0.0f;
                float block_sum1 = 0.0f;
#pragma unroll
                for (int nt = group*ReductionNt; nt < (group+1)*ReductionNt; ++nt) {
                    const float p00 = score[nt][0] > -CUDART_INF_F
                                          ? exp2_approx(__fmaf_rn(score[nt][0], ScaleLog2E, -m0_l2))
                                          : 0.0f;
                    const float p01 = score[nt][1] > -CUDART_INF_F
                                          ? exp2_approx(__fmaf_rn(score[nt][1], ScaleLog2E, -m0_l2))
                                          : 0.0f;
                    const float p10 = score[nt][2] > -CUDART_INF_F
                                          ? exp2_approx(__fmaf_rn(score[nt][2], ScaleLog2E, -m1_l2))
                                          : 0.0f;
                    const float p11 = score[nt][3] > -CUDART_INF_F
                                          ? exp2_approx(__fmaf_rn(score[nt][3], ScaleLog2E, -m1_l2))
                                          : 0.0f;
                    block_sum0 += p00 + p01;
                    block_sum1 += p10 + p11;
                    // Recover the probability bits lost by a single BF16 MMA operand.
                    // Both products accumulate in FP32 and use the same represented V.
                    const float r00=p00-__bfloat162float(__float2bfloat16_rn(p00));
                    const float r01=p01-__bfloat162float(__float2bfloat16_rn(p01));
                    const float r10=p10-__bfloat162float(__float2bfloat16_rn(p10));
                    const float r11=p11-__bfloat162float(__float2bfloat16_rn(p11));
                    const int pk = nt >> 1;
                    if ((nt & 1) == 0) {
                        p_frag[pk][0] = pack_bf16x2(p00, p01);
                        p_frag[pk][1] = pack_bf16x2(p10, p11);
                        p_residual[pk][0] = pack_bf16x2(r00, r01);
                        p_residual[pk][1] = pack_bf16x2(r10, r11);
                    } else {
                        p_frag[pk][2] = pack_bf16x2(p00, p01);
                        p_frag[pk][3] = pack_bf16x2(p10, p11);
                        p_residual[pk][2] = pack_bf16x2(r00, r01);
                        p_residual[pk][3] = pack_bf16x2(r10, r11);
                    }
                }

                l0 = __fmaf_rn(l0, alpha0, block_sum0);
                l1 = __fmaf_rn(l1, alpha1, block_sum1);
                m0 = next_m0;
                m1 = next_m1;
            }
        }

        cp_wait<0>();
        __syncthreads();
        if (kb + 1 < key_blocks) {
            causal_attention_stage_kv<Bc, Threads>(k_s, k, key0 + Bc, tile.end, kv_head, tid,
                                                   k_stride_d, k_stride_h, k_stride_t);
            cp_commit();
        }

#pragma unroll
        for (int group = 0; group < ProbabilityGroups; ++group) {
            const bool probability_active = ProbabilityGroups == 1 ||
                key0 + group * ReductionBc < tile.end;
            if (probability_active) {
                if constexpr (ProbabilityGroups > 1) {
#pragma unroll
                    for (int n = 0; n < PVNt; ++n) {
                        acc[n][0] *= group_alpha0[group];
                        acc[n][1] *= group_alpha0[group];
                        acc[n][2] *= group_alpha1[group];
                        acc[n][3] *= group_alpha1[group];
                    }
                }
                constexpr int PVTilePairs = (PVNt + 1) / 2;
                constexpr int PVLoads     = PVGroupKs * PVTilePairs;
                unsigned vf[2][4];
                ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                              causal_attention_swz_addr(v_lane_base + group * PVGroupKs * 16 * RowBytes,
                                                        0u, v_as, v_r));
#pragma unroll
                for (int load = 0; load < PVLoads; ++load) {
                    const int pk   = group * PVGroupKs + load / PVTilePairs;
                    const int n2   = (load % PVTilePairs) * 2;
                    const int cur  = load & 1;
                    const int next = cur ^ 1;
                    if (load + 1 < PVLoads) {
                        const int next_pk = group * PVGroupKs + (load + 1) / PVTilePairs;
                        const int next_n2 = ((load + 1) % PVTilePairs) * 2;
                        ldmatrix_x4_t(vf[next][0], vf[next][1], vf[next][2], vf[next][3],
                                      causal_attention_swz_addr(
                                          v_lane_base + static_cast<unsigned>(next_pk * 16 * RowBytes),
                                          static_cast<unsigned>(next_n2 << 4), v_as, v_r));
                    }
                    mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[pk][0], p_frag[pk][1],
                             p_frag[pk][2], p_frag[pk][3], vf[cur][0], vf[cur][1]);
                    mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3],
                             p_residual[pk][0], p_residual[pk][1], p_residual[pk][2], p_residual[pk][3],
                             vf[cur][0], vf[cur][1]);
                    if (n2 + 1 < PVNt) {
                        mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3],
                                 p_frag[pk][0], p_frag[pk][1], p_frag[pk][2], p_frag[pk][3], vf[cur][2],
                                 vf[cur][3]);
                        mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3],
                                 p_residual[pk][0], p_residual[pk][1], p_residual[pk][2], p_residual[pk][3],
                                 vf[cur][2], vf[cur][3]);
                    }
                }
            }
        }
    }

    l0                 = warp_sum<4>(l0, FullMask);
    l1                 = warp_sum<4>(l1, FullMask);
    const float inv_l0 = l0 > 0.0f ? __frcp_rn(l0) : 0.0f;
    const float inv_l1 = l1 > 0.0f ? __frcp_rn(l1) : 0.0f;
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0     = n * 8 + 2 * lid;
        const int query0 = tile.q0 + warp_row0 + gid;
        const int query1 = query0 + 8;
        if (query0 < tile.end) {
            const std::int64_t offset =
                (static_cast<std::int64_t>(query0) * heads + head) * D + d0;
            store_vec(&out[offset], pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0));
        }
        if (query1 < tile.end) {
            const std::int64_t offset =
                (static_cast<std::int64_t>(query1) * heads + head) * D + d0;
            store_vec(&out[offset], pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1));
        }
    }
}

} // namespace ninfer::ops::detail
