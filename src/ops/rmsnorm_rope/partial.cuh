#pragma once

#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops::detail {

// Each warp owns one D256 head and retains the normalized BF16 values in
// registers. The first 64 dimensions use split-half RoPE; the rest keep norm.
template <int HeadsPerBlock>
__global__ __launch_bounds__(32 * HeadsPerBlock) void rmsnorm_rope_partial_kernel(
    const std::int32_t* positions, int tokens, int axes, const __nv_bfloat16* q_weight,
    const __nv_bfloat16* k_weight, __nv_bfloat16* q, __nv_bfloat16* k, int q_heads, int k_heads,
    float epsilon, float theta, bool unit_offset) {
    const int token = blockIdx.x;
    const int combined = blockIdx.y * HeadsPerBlock + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    __shared__ float cosine[32], sine[32];
    if (threadIdx.x < 32) {
        const int i = threadIdx.x;
        const int axis = axes == 3 ? i % 3 : 0;
        const float freq = powf(theta, -2.0f * float(i) / 64.0f);
        const float angle = float(positions[axis * tokens + token]) * freq;
        sincosf(angle, &sine[i], &cosine[i]);
    }
    __syncthreads();
    if (combined >= q_heads + k_heads) { return; }
    const bool query = combined < q_heads;
    const int head = query ? combined : combined - q_heads;
    const int heads = query ? q_heads : k_heads;
    auto* data = reinterpret_cast<__nv_bfloat162*>(query ? q : k);
    const auto* weight = reinterpret_cast<const __nv_bfloat162*>(query ? q_weight : k_weight);
    const std::int64_t base = (std::int64_t(token) * heads + head) * 128;
    __nv_bfloat162 input[4], norm[4];
    float2 weights[4];
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        input[i] = data[base + lane + 32 * i];
        weights[i] = __bfloat1622float2(weight[lane + 32 * i]);
        const float2 v = __bfloat1622float2(input[i]);
        sum += v.x * v.x + v.y * v.y;
    }
    sum = warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum / 256.0f + epsilon) : 0.0f;
    inv = __shfl_sync(kFullWarpMask, inv, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 v = __bfloat1622float2(input[i]);
        float2 w = weights[i];
        if (unit_offset) {
            w.x += 1.0f;
            w.y += 1.0f;
        }
        norm[i] = __floats2bfloat162_rn(v.x * inv * w.x, v.y * inv * w.y);
    }
    const float2 n = __bfloat1622float2(norm[0]);
    const float other_x = __shfl_xor_sync(kFullWarpMask, n.x, 16);
    const float other_y = __shfl_xor_sync(kFullWarpMask, n.y, 16);
    const int pair = (lane % 16) * 2;
    const float c0 = cosine[pair], s0 = sine[pair];
    const float c1 = cosine[pair + 1], s1 = sine[pair + 1];
    norm[0] = lane < 16
        ? __floats2bfloat162_rn(n.x * c0 - other_x * s0, n.y * c1 - other_y * s1)
        : __floats2bfloat162_rn(n.x * c0 + other_x * s0, n.y * c1 + other_y * s1);
#pragma unroll
    for (int i = 0; i < 4; ++i) { data[base + lane + 32 * i] = norm[i]; }
}

} // namespace ninfer::ops::detail
