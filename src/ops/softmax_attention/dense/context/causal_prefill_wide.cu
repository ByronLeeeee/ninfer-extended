#include "ops/softmax_attention/dense/context/causal_prefill_wide.h"
#include "ops/softmax_attention/dense/context/causal_prefill_wide.cuh"
#include "core/device.h"

namespace ninfer::ops::detail {
namespace {

template<bool Packed>
void launch(const void* q, const void* k, const void* v, void* out,
    int sequences, int maximum, int qh, int kh, int qs, int ks,
    const std::int32_t* begins, const std::int32_t* lengths, cudaStream_t stream) {
    constexpr int Br = 32, Bc = 64;
    constexpr int bytes = (Br + 2 * Bc) * 128 * 2;
    causal_attention_wide_kernel<Packed>
        <<<dim3((maximum - 1) / Br + 1, qh, sequences), Br * 2, bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k),
            static_cast<const __nv_bfloat16*>(v), maximum, qh, kh,
            static_cast<__nv_bfloat16*>(out), 1, 128, qs, 1, 128, ks, 1, 128, ks,
            begins, lengths);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_causal_attention_wide(const void* q, const void* k, const void* v, void* out,
    int sequences, int maximum, int qh, int kh, int qs, int ks,
    const std::int32_t* begins, const std::int32_t* lengths,
    bool packed, cudaStream_t stream) {
    if (packed) launch<true>(q, k, v, out, sequences, maximum, qh, kh, qs, ks, begins, lengths, stream);
    else launch<false>(q, k, v, out, sequences, maximum, qh, kh, qs, ks, nullptr, nullptr, stream);
}

} // namespace ninfer::ops::detail
