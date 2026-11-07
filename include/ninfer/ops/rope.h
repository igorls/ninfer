#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer::ops {

/**
 * YaRN frequency scaling (NTK-by-parts interpolation with attention-magnitude correction) for the
 * Text RoPE forms below. With R = rotary_dim, base frequency w_i = theta^(-2i/R) for pair
 * i in [0,R/2), and correction dimension c(r) = R*ln(L/(2*pi*r)) / (2*ln(theta)):
 *
 *   lo  = max(floor(c(beta_fast)), 0),   hi = min(ceil(c(beta_slow)), R-1)
 *   g_i = clamp((i - lo) / span, 0, 1),  span = hi - lo, or 0.001 when hi = lo
 *   w'_i = (1 - g_i) * w_i + g_i * w_i / factor
 *
 * phi(i,t) = position * w'_i replaces the unscaled angle and both rotated outputs of every pair
 * are multiplied by attention_factor m:
 *
 *   y[i] = m*(x[i]*cos(phi) - x[i+R/2]*sin(phi)),  y[i+R/2] = m*(x[i+R/2]*cos(phi) + x[i]*sin(phi))
 *
 * L is the position count the unscaled frequencies were trained for. factor and attention_factor
 * are positive and finite, factor > 1, L > 0, and 0 < beta_slow < beta_fast.
 */
struct RopeYarn {
    float factor                                  = 1;
    std::int32_t original_max_position_embeddings = 0;
    float beta_fast                               = 32;
    float beta_slow                               = 1;
    float attention_factor                        = 1;
};

/**
 * Applies split-half NeoX RoPE in place. For pair i in [0,rotary_dim/2), angle phi(i,t), and
 * each head:
 *
 *   y[i]     = x[i] * cos(phi) - x[i+R/2] * sin(phi)
 *   y[i+R/2] = x[i+R/2] * cos(phi) + x[i] * sin(phi), R = rotary_dim.
 *
 * Dimensions [rotary_dim,head_dim) are unchanged. Supported modes are:
 *
 * - Text 1-D: positions I32 [T], either head_dim=256 with even 0<rotary_dim<=256, or the
 *   head_dim=rotary_dim=128; phi=positions[t]*theta^(-2*i/rotary_dim).
 * - Text MRoPE: positions I32 [T,3], head_dim=256, rotary_dim=64; pair i uses axis i%3 with
 *   the same frequency as Text 1-D.
 * - Vision 2-D: positions I32 [T,2], head_dim=rotary_dim=D in {64,72}; the first D/4
 *   pairs use axis 0 and the remaining pairs use axis 1, with local frequency
 *   theta^(-2*(i%(D/4))/(D/2)). Positions are stored as two contiguous coordinate axes.
 *
 * positions is contiguous and theta is positive and finite. Q/K tensors are BF16
 * [head_dim,heads,T] with positive head counts, contiguous head features and heads, and an optional
 * padded token stride. q and k must not overlap one another or positions. The Op mutates only
 * dimensions [0,rotary_dim); unrotated dimensions and token padding remain bit-exact. Empty token
 * extents perform no device work. The Op uses no workspace or persistent state. `execution`
 * supplies the stream and the selected device's positive physical SM count.
 */
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          CudaExecutionView execution);

// Single-tensor form with the same formula and storage contract. The head count comes directly
// from x; Q versus K role does not change the transformation.
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
          CudaExecutionView execution);

// YaRN forms: Text 1-D (positions I32 [T]) or Text MRoPE (I32 [T,3], pair i uses axis i%3) with
// head_dim=256 and rotary_dim=64, the angle and magnitude defined by RopeYarn, and otherwise the
// storage, alias, padding and resource contract of the forms above. Angles are accurate for every
// I32 position; other geometries are rejected.
void rope(const Tensor& positions, int rotary_dim, float theta, const RopeYarn& yarn, Tensor& q,
          Tensor& k, CudaExecutionView execution);

void rope(const Tensor& positions, int rotary_dim, float theta, const RopeYarn& yarn, Tensor& x,
          CudaExecutionView execution);

} // namespace ninfer::ops
