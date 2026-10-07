#pragma once
#include <cuda_runtime.h>

namespace ninfer::ops {
// Input is contiguous BF16 [rows,width], output is nonaliasing FP32 [rows,dimensions].
// Select the first dimensions columns, optionally divide by max(L2 norm,1e-12).
// Finite represented inputs; rows>0, width>=dimensions>=1. No device allocation,
// no input mutation. FP32 reductions and outputs qualify against a naive FP64 oracle.
void bf16_embedding_output(const void* input, float* output, int rows,
                           int width, int dimensions, bool normalize, cudaStream_t stream);
} // namespace ninfer::ops
