#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
// Single-column, single-row BF16 [8192,1024] snapshot implementation.
void bf16_gdn_snapshot_decode(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                              Tensor& states, const Tensor& valid_columns,
                              const Tensor& initial_slots, const Tensor& snapshot_slots,
                              Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                              cudaStream_t stream);
} // namespace ninfer::ops::detail
