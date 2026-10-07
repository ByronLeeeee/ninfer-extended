#include "ninfer/ops/dense_attention.h"
#include "ops/softmax_attention/dense/context/causal_prefill_kernel.cuh"
#include "ops/softmax_attention/dense/context/causal_prefill_wide.h"
#include "core/device.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

template<int Bc, bool Packed>
void launch_causal(const void* q, const void* k, const void* v, void* out,
    int sequences, int maximum, int qh, int kh, int qs, int ks,
    const std::int32_t* begins, const std::int32_t* lengths, cudaStream_t stream) {
    constexpr int Br = 32;
    constexpr int bytes = (Br + 2 * Bc) * 128 * 2;
    if constexpr (Bc == 64) {
        detail::launch_causal_attention_wide(q, k, v, out, sequences, maximum,
                                             qh, kh, qs, ks, begins, lengths, Packed, stream);
    } else {
        detail::causal_attention_flash_kernel<Br, Bc, Packed>
        <<<dim3((maximum - 1) / Br + 1, qh, sequences), Br * 2, bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k),
            static_cast<const __nv_bfloat16*>(v), maximum, qh, kh,
            static_cast<__nv_bfloat16*>(out), 1, 128, qs, 1, 128, ks, 1, 128, ks,
            begins, lengths);
    }
    CUDA_CHECK(cudaGetLastError());
}

template<bool Packed>
void dispatch_causal(const void* q, const void* k, const void* v, void* out,
    int sequences, int maximum, int qh, int kh, int qs, int ks,
    const std::int32_t* begins, const std::int32_t* lengths, cudaStream_t stream,
    int multiprocessor_count) {
    const std::int64_t blocks = static_cast<std::int64_t>((maximum - 1) / 32 + 1)
        * qh * sequences;
    // Wider staging helps the long-input regime while the grid stays within
    // eight waves of physical SMs. Keep narrow staging for more saturated grids.
    if (maximum >= 1536 && multiprocessor_count > 0 &&
        blocks <= 8LL * multiprocessor_count)
        launch_causal<64, Packed>(q, k, v, out, sequences, maximum, qh, kh,
                                  qs, ks, begins, lengths, stream);
    else
        launch_causal<32, Packed>(q, k, v, out, sequences, maximum, qh, kh,
                                  qs, ks, begins, lengths, stream);
}

bool supported_profile(int qh, int kh) {
    return (qh == 16 && (kh == 4 || kh == 8 || kh == 16)) || (qh == 8 && kh == 8);
}

bool aligned(const void* p) {
    return p && reinterpret_cast<std::uintptr_t>(p) % 16 == 0;
}

} // namespace

void causal_bf16_attention(const void* q, const void* k, const void* v, void* out,
    int tokens, int qh, int kh, int qs, int ks, cudaStream_t stream,
    std::int32_t multiprocessor_count) {
    if (!supported_profile(qh, kh) || tokens <= 0 || qs < qh * 128 || ks < kh * 128 ||
        qs % 8 || ks % 8 || multiprocessor_count < 0 ||
        !aligned(q) || !aligned(k) || !aligned(v) || !aligned(out))
        throw std::invalid_argument("causal_bf16_attention: unsupported geometry or layout");
    dispatch_causal<false>(q, k, v, out, 1, tokens, qh, kh, qs, ks, nullptr, nullptr,
                          stream, multiprocessor_count);
}

void packed_causal_bf16_attention(const void* q, const void* k, const void* v, void* out,
    int sequences, int max_tokens, int qh, int kh, int qs, int ks,
    const std::int32_t* begins, const std::int32_t* lengths, cudaStream_t stream,
    std::int32_t multiprocessor_count) {
    if (!supported_profile(qh, kh) || sequences < 1 || sequences > 65535 || max_tokens <= 0 ||
        qs < qh * 128 || ks < kh * 128 || qs % 8 || ks % 8 || multiprocessor_count < 0 ||
        !aligned(q) || !aligned(k) || !aligned(v) || !aligned(out) ||
        !begins || !lengths || reinterpret_cast<std::uintptr_t>(begins) % alignof(std::int32_t) ||
        reinterpret_cast<std::uintptr_t>(lengths) % alignof(std::int32_t))
        throw std::invalid_argument("packed_causal_bf16_attention: unsupported geometry or layout");
    dispatch_causal<true>(q, k, v, out, sequences, max_tokens, qh, kh, qs, ks,
                         begins, lengths, stream, multiprocessor_count);
}

} // namespace ninfer::ops
