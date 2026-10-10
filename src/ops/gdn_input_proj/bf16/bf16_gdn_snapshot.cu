#include "ops/gdn_input_proj/bf16/bf16_gdn_snapshot.h"

#include "core/device.h"
#include "ops/gdn_input_proj/gdn_conv.cuh"
#include "ops/linear/bf16/bf16_simt.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Bf16Geometry<8192, 1024>;
using Schedule = Bf16GemvSchedule<4, 1, 2, 8, 4, Bf16ActivationAccess::Direct,
    Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2, 8>;

__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm)
void bf16_gdn_snapshot_decode_kernel(const __nv_bfloat16* __restrict__ x,
                                    const __nv_bfloat16* __restrict__ weight,
                                    GdnConvEpilogue<SnapshotHistoryPublish> conv,
                                    __nv_bfloat16* z) {
    __shared__ Bf16GemvSharedStorage<Geometry, Schedule> shared;
    const auto* activation = prepare_bf16_activation<Geometry, Schedule>(x, shared);
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int row0 = blockIdx.x * Schedule::kRowsPerCta + warp * Schedule::kRowsPerWarp;
    float accumulators[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};
    compute_bf16_gemv_rows<Geometry, Schedule>(activation, weight, row0, 0, lane, accumulators);

    // Preserve the materialized projection's reduction and BF16 rounding, then
    // let adjacent lanes apply the convolution to adjacent rows.
    float projected = 0.0F;
#pragma unroll
    for (int row = 0; row < Schedule::kRowsPerWarp; ++row) {
        float sum = 0.0F;
#pragma unroll
        for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
            sum += accumulators[row][chain];
        }
        sum = warp_reduce_sum(sum);
        sum = __shfl_sync(kFullWarpMask, sum, 0);
        if (lane == row) { projected = sum; }
    }
    if (lane < Schedule::kRowsPerWarp) {
        const int row = row0 + lane;
        const __nv_bfloat16 rounded = __float2bfloat16_rn(projected);
        if (row < 6144) {
            const float values[1] = {__bfloat162float(rounded)};
            conv.store<1>(row, values);
        } else {
            z[row - 6144] = rounded;
        }
    }
}

using BatchSchedule = Bf16SimtSchedule<4, 1, 2, 8, 1, 4,
    Bf16SimtActivationAccess::WarpPacked, Bf16WeightCache::Default,
    Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;

template <int Batch>
__global__ __launch_bounds__(BatchSchedule::kThreads, BatchSchedule::kMinBlocksPerSm)
void bf16_gdn_snapshot_batch_kernel(const __nv_bfloat16* __restrict__ x,
                                    const __nv_bfloat16* __restrict__ weight,
                                    GdnConvEpilogue<SnapshotHistoryPublish> conv,
                                    __nv_bfloat16* z, int live_batch) {
    const int lane = threadIdx.x % kWarpSize, warp = threadIdx.x / kWarpSize;
    const int row0 = blockIdx.x * BatchSchedule::kRowsPerCta + warp * BatchSchedule::kRowsPerWarp;
    float accumulators[BatchSchedule::kRowsPerWarp][Batch][1] = {};
    bf16_simt_compute_rows<Geometry, Batch, BatchSchedule>(x, weight, row0, 0, lane,
                                                         accumulators, live_batch);
    // Each participating lane owns one (column,row) output. Publish all columns
    // together so convolution loads and state stores do not serialize by token.
    float projected = 0.0F;
#pragma unroll
    for (int token = 0; token < Batch; ++token) {
#pragma unroll
        for (int row = 0; row < BatchSchedule::kRowsPerWarp; ++row) {
            float sum = warp_reduce_sum(accumulators[row][token][0]);
            sum = __shfl_sync(kFullWarpMask, sum, 0);
            if (lane == token * BatchSchedule::kRowsPerWarp + row) projected = sum;
        }
    }
    const int token = lane / BatchSchedule::kRowsPerWarp;
    if (token < live_batch && lane < Batch * BatchSchedule::kRowsPerWarp) {
        const int row = row0 + lane % BatchSchedule::kRowsPerWarp;
        const auto rounded = __float2bfloat16_rn(projected);
        if (row < 6144) {
            const float values[1] = {__bfloat162float(rounded)};
            conv.batch_row = token;
            conv.store<1>(row, values);
        } else z[static_cast<std::int64_t>(token) * 2048 + row - 6144] = rounded;
    }
}
} // namespace

void bf16_gdn_snapshot_decode(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                              Tensor& states, const Tensor& valid_columns,
                              const Tensor& initial_slots, const Tensor& snapshot_slots,
                              Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                              cudaStream_t stream) {
    auto* history = static_cast<__nv_bfloat16*>(states.data);
    const GdnConvEpilogue<SnapshotHistoryPublish> conv{
        static_cast<const __nv_bfloat16*>(conv_weight.data), history,
        static_cast<const std::int32_t*>(initial_slots.data),
        static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data), 6144, 2048, 2048, 2048, 0, 1, 0,
        {history, static_cast<const std::int32_t*>(snapshot_slots.data), 6144}};
    const auto* input = static_cast<const __nv_bfloat16*>(x.data);
    const auto* parent = static_cast<const __nv_bfloat16*>(weight.qdata);
    auto* gate = static_cast<__nv_bfloat16*>(z.data);
    const int batch = x.ne[2];
    if (batch == 1) {
        bf16_gdn_snapshot_decode_kernel<<<8192 / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            input, parent, conv, gate);
    } else {
#define BATCH(B) bf16_gdn_snapshot_batch_kernel<B><<<8192 / BatchSchedule::kRowsPerCta, BatchSchedule::kThreads, 0, stream>>>(input, parent, conv, gate, batch)
        if (batch == 2) { BATCH(2); }
        else if (batch == 3) { BATCH(3); }
        else if (batch == 4) { BATCH(4); }
        else { BATCH(8); }
#undef BATCH
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
