#pragma once

#include "ops/linear/bf16/bf16_simt.cuh"

namespace ninfer::ops::detail {

template <int RowsPerBranch, int OutputRows>
struct Bf16SwiGluRows {
    __device__ __forceinline__ int weight_row(int row0, int local_row) const {
        return row0 + local_row % RowsPerBranch +
               (local_row >= RowsPerBranch ? OutputRows : 0);
    }
};

__device__ __forceinline__ __nv_bfloat16 bf16_swiglu_value(float gate, float up) {
    // Preserve the established BF16 projection and SiLU boundaries.
    gate = __bfloat162float(__float2bfloat16_rn(gate));
    up = __bfloat162float(__float2bfloat16_rn(up));
    const float activation = __bfloat162float(__float2bfloat16_rn(gate / (1.f + expf(-gate))));
    return __float2bfloat16_rn(activation * up);
}

template <class Geometry, class Schedule>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm)
void bf16_swiglu_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                            const __nv_bfloat16* __restrict__ weight,
                            __nv_bfloat16* __restrict__ out) {
    static_assert(Schedule::kWarpsPerRow == 1 && Schedule::kRowsPerWarp % 2 == 0);
    constexpr int RowsPerBranch = Schedule::kRowsPerWarp / 2;
    constexpr int OutputRows = Geometry::kOutputRows / 2;
    static_assert(Geometry::kOutputRows % 2 == 0);
    static_assert(OutputRows % (Schedule::kWarpsPerCta * RowsPerBranch) == 0);
    __shared__ Bf16GemvSharedStorage<Geometry, Schedule> shared;
    const auto* activation = prepare_bf16_activation<Geometry, Schedule>(x, shared);
    const int lane = threadIdx.x % kWarpSize, warp = threadIdx.x / kWarpSize;
    const int row0 = (blockIdx.x * Schedule::kWarpsPerCta + warp) * RowsPerBranch;
    float accumulators[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};
    compute_bf16_gemv_rows<Geometry, Schedule>(activation, weight, row0, 0, lane,
        accumulators, Bf16SwiGluRows<RowsPerBranch, OutputRows>{});
    float totals[Schedule::kRowsPerWarp];
#pragma unroll
    for (int row = 0; row < Schedule::kRowsPerWarp; ++row) {
        float value = 0;
#pragma unroll
        for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain)
            value += accumulators[row][chain];
        totals[row] = warp_reduce_sum(value);
    }
    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < RowsPerBranch; ++row)
            out[row0 + row] = bf16_swiglu_value(totals[row], totals[row + RowsPerBranch]);
    }
}

template <class Geometry, int ActiveTokens, class Schedule>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm)
void bf16_swiglu_simt_kernel(const __nv_bfloat16* __restrict__ x,
                            const __nv_bfloat16* __restrict__ weight,
                            __nv_bfloat16* __restrict__ out, int live_tokens) {
    static_assert(Schedule::kWarpsPerRow == 1 && Schedule::kRowsPerWarp % 2 == 0);
    static_assert(ActiveTokens >= 2 && ActiveTokens <= 8);
    constexpr int RowsPerBranch = Schedule::kRowsPerWarp / 2;
    constexpr int OutputRows = Geometry::kOutputRows / 2;
    static_assert(Geometry::kOutputRows % 2 == 0);
    static_assert(OutputRows % (Schedule::kWarpsPerCta * RowsPerBranch) == 0);
    const int lane = threadIdx.x % kWarpSize, warp = threadIdx.x / kWarpSize;
    const int row0 = (blockIdx.x * Schedule::kWarpsPerCta + warp) * RowsPerBranch;
    float accumulators[Schedule::kRowsPerWarp][ActiveTokens][Schedule::kAccumulatorChains] = {};
    bf16_simt_compute_rows<Geometry, ActiveTokens, Schedule>(x, weight, row0, 0, lane,
        accumulators, live_tokens, Bf16SwiGluRows<RowsPerBranch, OutputRows>{});
#pragma unroll
    for (int row = 0; row < RowsPerBranch; ++row) {
#pragma unroll
        for (int token = 0; token < ActiveTokens; ++token) {
            float gate = 0, up = 0;
#pragma unroll
            for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
                gate += accumulators[row][token][chain];
                up += accumulators[row + RowsPerBranch][token][chain];
            }
            gate = warp_reduce_sum(gate);
            up = warp_reduce_sum(up);
            if (lane == 0 && token < live_tokens)
                out[static_cast<std::int64_t>(token) * OutputRows + row0 + row] =
                    bf16_swiglu_value(gate, up);
        }
    }
}

} // namespace ninfer::ops::detail
