#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops::detail {

void launch_causal_attention_wide(const void* q, const void* k, const void* v, void* out,
    int sequences, int maximum, int qh, int kh, int qs, int ks,
    const std::int32_t* begins, const std::int32_t* lengths,
    bool packed, cudaStream_t stream);

} // namespace ninfer::ops::detail
