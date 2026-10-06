#pragma once

#include "ops/kernel/rmsnorm.cuh"
#include <cuda_bf16.h>

namespace ninfer::ops::detail {

// The explicit normalized BF16 output is also the control projection operand.
// The first 256 threads retain the existing pair-wise RMS reduction ordering.
template <int InputRows, int Heads, int Threads>
__global__ __launch_bounds__(Threads) void bf16_norm_gating_small_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* norm, const __nv_bfloat16* ab,
    const float* alog, const float* bias, __nv_bfloat16* h, float* g, float* beta, float eps) {
    static_assert(InputRows % 512 == 0 && Threads >= 256 && Threads <= 1024);
    static_assert(Heads % (Threads / 32) == 0);
    constexpr int Pairs = InputRows / 512, Warps = Threads / 32;
    constexpr int HeadsPerWarp = Heads / Warps;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, token = blockIdx.x;
    const auto base = std::int64_t(token) * (InputRows / 2);
    __nv_bfloat162 values[Pairs], gains[Pairs];
    float sum = 0;
    if (tid < 256) {
#pragma unroll
        for (int i = 0; i < Pairs; ++i) {
            const int pair = tid + i * 256;
            values[i] = reinterpret_cast<const __nv_bfloat162*>(x)[base + pair];
            gains[i] = reinterpret_cast<const __nv_bfloat162*>(norm)[pair];
            const float2 v = __bfloat1622float2(values[i]);
            sum += v.x * v.x + v.y * v.y;
        }
    }
    __shared__ float partial[Warps], inverse;
    __shared__ __align__(16) __nv_bfloat16 normalized[InputRows];
    const float total = block_reduce_sum<Threads>(sum, partial);
    if (tid == 0) inverse = rsqrtf(total / InputRows + eps);
    __syncthreads();
    if (tid < 256) {
#pragma unroll
        for (int i = 0; i < Pairs; ++i) {
            const int pair = tid + i * 256;
            const float2 v = __bfloat1622float2(values[i]);
            const float2 gain = __bfloat1622float2(gains[i]);
            const auto result = __floats2bfloat162_rn(
                rmsnorm_epilogue<RmsEpilogue::Offset>(v.x, inverse, gain.x, 0),
                rmsnorm_epilogue<RmsEpilogue::Offset>(v.y, inverse, gain.y, 0));
            reinterpret_cast<__nv_bfloat162*>(normalized)[pair] = result;
            reinterpret_cast<__nv_bfloat162*>(h)[base + pair] = result;
        }
    }
    __syncthreads();
    float aa[HeadsPerWarp]{}, bb[HeadsPerWarp]{};
    for (int k = lane; k < InputRows; k += 32) {
        const float value = __bfloat162float(normalized[k]);
#pragma unroll
        for (int i = 0; i < HeadsPerWarp; ++i) {
            const int head = warp + i * Warps;
            aa[i] = fmaf(__bfloat162float(ab[head * InputRows + k]), value, aa[i]);
            bb[i] = fmaf(__bfloat162float(ab[(Heads + head) * InputRows + k]), value, bb[i]);
        }
    }
#pragma unroll
    for (int i = 0; i < HeadsPerWarp; ++i) {
        aa[i] = warp_reduce_sum(aa[i]);
        bb[i] = warp_reduce_sum(bb[i]);
        if (lane == 0) {
            const int head = warp + i * Warps;
            const float a = __bfloat162float(__float2bfloat16_rn(aa[i]));
            const float b = __bfloat162float(__float2bfloat16_rn(bb[i]));
            const float z = a + bias[head];
            const float sp = z > 20 ? z : log1pf(expf(z));
            g[token * Heads + head] = -expf(alog[head]) * sp;
            beta[token * Heads + head] = 1.f / (1.f + expf(-b));
        }
    }
}
} // namespace ninfer::ops::detail
