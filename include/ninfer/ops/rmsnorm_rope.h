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
 * Split a packed gated-query projection and apply one-centered per-head RMSNorm followed by
 * interleaved three-axis Text MRoPE to its query and key heads.
 *
 * Logical shapes: `projected` is BF16 [13312,T]. Its column t stores, for query head h in
 * [0,24), the query row block [512h, 512h+256) followed by the output-gate row block
 * [512h+256, 512h+512); key head g in [0,2) at [12288+256g, 12288+256g+256) and value head g at
 * [12800+256g, 12800+256g+256). `positions` is planar I32 [T,3] (axis-major: axis a of column t
 * at positions[a*T+t]). `q_norm_weight` and `k_norm_weight` are BF16 [256]. Outputs are q BF16
 * [256,24,T], gate BF16 [256,24,T], k BF16 [256,2,T] and v BF16 [256,2,T]. T is any positive
 * extent.
 *
 * Math, for every query head (weight w = q_norm_weight) and key head (w = k_norm_weight) of
 * column t with input vector x[0..256):
 *
 *   inv    = 1 / sqrt(sum_d x[d]^2 / 256 + 1e-6)
 *   n[d]   = x[d] * inv * (1 + w[d])
 *   phi(i) = positions[(i mod 3)*T + t] * (1e7)^(-2i/64),          0 <= i < 32
 *   out[i]    = n[i]    * cos(phi(i)) - n[i+32] * sin(phi(i))
 *   out[i+32] = n[i+32] * cos(phi(i)) + n[i]    * sin(phi(i))
 *   out[d]    = n[d],                                               64 <= d < 256
 *
 * gate and v are bit-exact copies of their projected rows. Normalization and rotation form one
 * operation: there is no observable BF16 materialization of n, and the only rounding boundary is
 * the final BF16 store of q and k. The oracle evaluates the formula naively in FP64 from the
 * represented BF16 inputs. Reduction order, phase range reduction and intermediate precision are
 * private implementation choices.
 *
 * Effects: q, gate, k and v are completely overwritten; nothing else is written. All tensors are
 * contiguous with 16-byte-aligned data, and the four outputs must not overlap each other or any
 * input. The Op owns no workspace or persistent state.
 */
void rmsnorm_rope(const Tensor& projected, const Tensor& positions, const Tensor& q_norm_weight,
                  const Tensor& k_norm_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                  cudaStream_t stream);

} // namespace ninfer::ops
