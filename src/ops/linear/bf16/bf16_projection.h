#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops::detail {
bool bf16_projection_shape(int n, int k);
bool bf16_swiglu_shape(int n, int k);
int bf16_swiglu_small_max_tokens(int n, int k);
void bf16_projection_linear(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void bf16_projection_add(const Tensor&, const Weight&, Tensor&, WorkspaceArena&, cudaStream_t);
void bf16_projection_swiglu(const Tensor&, const Weight&, Tensor&, WorkspaceArena&, cudaStream_t,
                            std::int32_t multiprocessor_count);
void bf16_projection_split(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t);
void bf16_projection_attn(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&, cudaStream_t);
void bf16_gating_projection(const Tensor&, const Weight&, const Tensor&, const Tensor&, Tensor&, Tensor&, cudaStream_t);
} // namespace ninfer::ops::detail
