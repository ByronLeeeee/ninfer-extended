#include "ops/rmsnorm_rope/launch.h"
#include "ops/rmsnorm_rope/partial.cuh"
#include "core/device.h"

namespace ninfer::ops::detail {
void rmsnorm_rope_partial_launch(const Tensor& positions, const Tensor& q_norm_weight,
                                 const Tensor& k_norm_weight, float epsilon, float theta,
                                 bool unit_offset, Tensor& q, Tensor& k, cudaStream_t stream) {
    constexpr int kHeadsPerBlock = 4;
    const dim3 grid(q.ne[2], (q.ne[1] + k.ne[1] + kHeadsPerBlock - 1) / kHeadsPerBlock);
    rmsnorm_rope_partial_kernel<kHeadsPerBlock><<<grid, 32 * kHeadsPerBlock, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data), q.ne[2], positions.ne[1],
        static_cast<const __nv_bfloat16*>(q_norm_weight.data),
        static_cast<const __nv_bfloat16*>(k_norm_weight.data),
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        q.ne[1], k.ne[1], epsilon, theta, unit_offset);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
