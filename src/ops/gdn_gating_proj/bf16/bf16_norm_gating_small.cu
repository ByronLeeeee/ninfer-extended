#include "ops/gdn_gating_proj/bf16/bf16_norm_gating_small.cuh"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"
#include "core/device.h"

namespace ninfer::ops::detail {
void bf16_gdn_norm_gating_small_launch(const Tensor& x, const Tensor& norm, float eps,
    Tensor& h, const Weight& ab, const Tensor& alog, const Tensor& bias,
    Tensor& g, Tensor& beta, cudaStream_t stream) {
    bf16_norm_gating_small_kernel<1024, 16, 512><<<x.ne[1], 512, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(norm.data),
        static_cast<const __nv_bfloat16*>(ab.qdata), static_cast<const float*>(alog.data),
        static_cast<const float*>(bias.data), static_cast<__nv_bfloat16*>(h.data),
        static_cast<float*>(g.data), static_cast<float*>(beta.data), eps);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
