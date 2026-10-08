#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/gelu.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Ops: linear_bias_add and linear_bias_gelu
 *
 * Math / indexing:
 *   Let B be round-to-nearest-even BF16 and z[:,t] = B(B(W @ x[:,t]) + bias).
 *   linear_bias_add:  ideal[:,t] = residual[:,t] + z[:,t].
 *   linear_bias_gelu: ideal[:,t] = GELU(z[:,t]), with the formula from gelu.h.
 *
 * Logical shapes / supported domain:
 *   Contiguous BF16 x [K,T], bias [N], output/residual [N,T], T > 0,
 *   and BF16 Contiguous W [N,K]. LinearBiasAdd supports (N,K) = (768,768),
 *   (768,1536), (768,3072). LinearBiasGelu supports (3072,768), (3072,3072).
 *   Both GELU modes are supported. No activation quantization is performed.
 *
 * Numeric:
 *   Projection and bias-add rounding are observable BF16 seams so the fused
 *   operation retains the staged computation's values. An independent FP64
 *   oracle evaluates the represented weights/inputs and those two explicit
 *   casts, then evaluates ideal without rounding its final result. Public output
 *   rounding and private reduction/activation approximations belong to the
 *   numerical criterion.
 *
 * Effects / workspace / execution:
 *   Add updates only residual in place; Gelu writes only output. Neither output
 *   may overlap x, bias or W. Inputs and weights are preserved. No transient or
 *   persistent workspace is required; both functions support stream capture.
 */
void linear_bias_add(const Tensor& x, const Weight& w, const Tensor& bias,
                     Tensor& residual, cudaStream_t stream);
void linear_bias_gelu(const Tensor& x, const Weight& w, const Tensor& bias,
                      GeluMode mode, Tensor& output, cudaStream_t stream);

} // namespace ninfer::ops
