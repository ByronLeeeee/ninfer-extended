#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Apply plain per-head RMSNorm followed by full-head split-half 1-D RoPE in place.
 *
 * The pair profile is q BF16 [128,32,W,B], k BF16 [128,8,W,B], q_norm_weight and
 * k_norm_weight BF16 [128], and positions I32 [W,B], with W=2..16 and B=1..8. For each head and
 * physical token, the complete mathematical operation is
 *
 *   inv       = 1 / sqrt(sum_d x[d]^2 / 128 + 1e-6)
 *   n[d]      = x[d] * inv * norm_weight[d]
 *   angle(i)  = position * (1e7)^(-2*i/128), 0<=i<64
 *   out[i]    = n[i]    * cos(angle(i)) - n[i+64] * sin(angle(i))
 *   out[i+64] = n[i+64] * cos(angle(i)) + n[i]    * sin(angle(i)).
 *
 * Normalization and rotation form one semantic operation: there is no observable BF16
 * materialization of n. q and k are completely overwritten in place and must not overlap each
 * other, positions, or either norm weight. Read-only norm weights may overlap each other.
 * positions and weights remain unchanged. All tensors are contiguous and 4-byte aligned. The
 * independent oracle evaluates the complete formula naively in FP64 from the represented BF16
 * inputs; final BF16 outputs are promoted for comparison. Reduction order, coefficient range
 * reduction, and intermediate arithmetic precision are private implementation choices. The Op
 * owns no workspace or persistent state.
 */
void rmsnorm_rope(const Tensor& positions, const Tensor& q_norm_weight, const Tensor& k_norm_weight,
                  Tensor& q, Tensor& k, cudaStream_t stream);

/**
 * Single-K form of the same formula and effects. x is BF16 [128,8,T], norm_weight is BF16 [128],
 * and positions is I32 [T], with T=1..2048. x is completely overwritten in place and must not
 * overlap either read-only input.
 */
void rmsnorm_rope(const Tensor& positions, const Tensor& norm_weight, Tensor& x,
                  cudaStream_t stream);

/**
 * Per-head RMSNorm with an explicit BF16 boundary, followed by partial split-half RoPE.
 *
 * q and k are contiguous, 4-byte-aligned BF16 [256,Hq,T] and [256,Hk,T], with
 * Hq,Hk in 1..64 and T in 1..131072. Norm weights are BF16 [256]. positions is
 * contiguous I32 [T] or [T,3]. epsilon and theta must be positive and finite.
 * For each head, n[d] = BF16(x[d] / sqrt(sum(x*x)/256 + epsilon) *
 * (weight[d] + (unit_offset ? 1 : 0))). Only n[0..64) is rotated: pair i uses
 * dimensions i and i+32, angle = positions[t,axis] * theta^(-2*i/64), and
 * axis = i%3 for three-axis positions, otherwise zero. Dimensions 64..256
 * retain n. Both final outputs are BF16 and overwrite q and k in place.
 *
 * q and k must not overlap each other, positions, or either norm weight.
 * Read-only inputs remain unchanged. The independent FP64 oracle evaluates
 * the complete formula from represented BF16 inputs, including the specified
 * intermediate BF16 boundary. This Op has no workspace or persistent state.
 */
void rmsnorm_rope_partial(const Tensor& positions, const Tensor& q_norm_weight,
                          const Tensor& k_norm_weight, float epsilon, float theta,
                          bool unit_offset, Tensor& q, Tensor& k, cudaStream_t stream);

} // namespace ninfer::ops
