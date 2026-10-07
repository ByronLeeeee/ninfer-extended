#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>

namespace ninfer::ops::detail {

__device__ __forceinline__ void dense_float_online_update(
    const float* q, const float* k, const float* v, float* acc,
    float& maximum, float& denom) {
    float score = 0;
#pragma unroll
    for (int d = 0; d < 4; ++d) score = fmaf(q[d], k[d], score);
#pragma unroll
    for (int delta = 16; delta; delta /= 2)
        score += __shfl_xor_sync(0xffffffff, score, delta);
    score *= rsqrtf(128.f);
    float next = fmaxf(maximum, score);
    float scale = expf(maximum - next), prob = expf(score - next);
    denom = denom * scale + prob;
#pragma unroll
    for (int d = 0; d < 4; ++d) acc[d] = fmaf(prob, v[d], acc[d] * scale);
    maximum = next;
}

// Query rows with the same range start reuse represented K/V values in
// registers. Each row retains four key streams and the original FP32 arithmetic
// order. Different starts use independent streams within the same CTA.
template<int QueryRows>
__global__ void dense_float_prefill_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* key,
    const __nv_bfloat16* value, __nv_bfloat16* out, int queries,
    int qheads, int kvheads, int qstride, int kvstride,
    const int* begins, const int* ends, const int* positions, bool causal) {
    constexpr int D = 128, Warps = 4;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int head = blockIdx.y, kvh = head / (qheads / kvheads);
    int query0 = blockIdx.x * QueryRows;
    int begin[QueryRows], end[QueryRows], maximum_end = 0;
    bool valid[QueryRows], common_begin = true;
    float qv[QueryRows][4], acc[QueryRows][4], maximum[QueryRows], denom[QueryRows];
#pragma unroll
    for (int row = 0; row < QueryRows; ++row) {
        int qi = query0 + row;
        valid[row] = qi < queries;
        begin[row] = valid[row] ? begins[qi] : begins[query0];
        end[row] = valid[row] ? ends[qi] : 0;
        if (valid[row] && causal) end[row] = min(end[row], positions[qi] + 1);
        common_begin &= begin[row] == begin[0];
        maximum_end = max(maximum_end, end[row]);
        maximum[row] = -INFINITY; denom[row] = 0;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            qv[row][d] = valid[row] ? __bfloat162float(q[static_cast<long long>(qi) * qstride + head * D + lane + d * 32]) : 0;
            acc[row][d] = 0;
        }
    }
    const auto* kbase = key + kvh * D;
    const auto* vbase = value + kvh * D;
    if (common_begin) {
        for (int j = begin[0] + warp; j < maximum_end; j += Warps) {
            float kv[4], vv[4];
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                long long offset = static_cast<long long>(j) * kvstride + lane + d * 32;
                kv[d] = __bfloat162float(kbase[offset]); vv[d] = __bfloat162float(vbase[offset]);
            }
#pragma unroll
            for (int row = 0; row < QueryRows; ++row)
                if (j < end[row]) dense_float_online_update(qv[row], kv, vv, acc[row], maximum[row], denom[row]);
        }
    } else {
#pragma unroll
        for (int row = 0; row < QueryRows; ++row) {
            for (int j = begin[row] + warp; j < end[row]; j += Warps) {
                float kv[4], vv[4];
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    long long offset = static_cast<long long>(j) * kvstride + lane + d * 32;
                    kv[d] = __bfloat162float(kbase[offset]); vv[d] = __bfloat162float(vbase[offset]);
                }
                dense_float_online_update(qv[row], kv, vv, acc[row], maximum[row], denom[row]);
            }
        }
    }
    __shared__ float sums[QueryRows * Warps * D];
    __shared__ float maxima[QueryRows * Warps], denoms[QueryRows * Warps];
#pragma unroll
    for (int row = 0; row < QueryRows; ++row) {
        int index = row * Warps + warp;
        if (lane == 0) { maxima[index] = maximum[row]; denoms[index] = denom[row]; }
#pragma unroll
        for (int d = 0; d < 4; ++d) sums[index * D + lane + d * 32] = acc[row][d];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int row = 0; row < QueryRows; ++row) if (valid[row]) {
            float maximum_all = -INFINITY, denom_all = 0;
#pragma unroll
            for (int w = 0; w < Warps; ++w) maximum_all = fmaxf(maximum_all, maxima[row * Warps + w]);
#pragma unroll
            for (int w = 0; w < Warps; ++w)
                if (denoms[row * Warps + w] > 0) denom_all += denoms[row * Warps + w] * expf(maxima[row * Warps + w] - maximum_all);
#pragma unroll
            for (int d = 0; d < 4; ++d) {
                float total = 0;
#pragma unroll
                for (int w = 0; w < Warps; ++w)
                    if (denoms[row * Warps + w] > 0) total += sums[(row * Warps + w) * D + lane + d * 32] * expf(maxima[row * Warps + w] - maximum_all);
                out[(static_cast<long long>(query0 + row) * qheads + head) * D + lane + d * 32] = __float2bfloat16_rn(total / denom_all);
            }
        }
    }
}

} // namespace ninfer::ops::detail
