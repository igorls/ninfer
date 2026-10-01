#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Stream-mixing weights of one four-stream hyper-connection transition: the one-centered group
 * RMSNorm gain `norm` (contiguous BF16 [10240]) and the two low-rank mixing matrices
 * `input_mix_down` (contiguous BF16 [320,10240]) and `input_mix_up` (contiguous BF16
 * [10240,320]).
 */
struct HyperConnectionWeights {
    Tensor norm;
    Weight input_mix_down;
    Weight input_mix_up;
};

/**
 * Returns the caller-owned transient capacity required by hyper_connection_prepare() and
 * hyper_connection_mix() for every T in the inclusive `[min_tokens,max_tokens]` interval.
 * Invalid intervals throw.
 */
[[nodiscard]] std::size_t hyper_connection_workspace_capacity_bytes(std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

/**
 * Op: four-stream hyper-connection block-input preparation.
 *
 * Math / indexing:
 *   For every token column t, with the four-stream state H = hidden[:,t] viewed as H[s,c] =
 *   hidden[s*2560+c,t] (s = 0..3, c = 0..2559):
 *     N[s,c]      = H[s,c] / sqrt(sum_c' H[s,c']^2 / 2560 + 1e-6) * (1 + norm[s*2560+c])
 *     L[r]        = silu(sum_j input_mix_down[r,j] * N[j] / 4),             r = 0..319
 *     M[s,c]      = sigmoid(sum_r input_mix_up[s*2560+c,r] * L[r])
 *     block_input[c,t] = (1/4) * sum_s M[s,c] * N[s,c]
 *     injection[s,t]   = 2 * sigmoid(sum_j block_inject[s,j] * N[j] / 4)
 *   where N[j] is the flattened N[s,c] with j = s*2560+c, silu(x) = x*sigmoid(x) and
 *   sigmoid(x) = 1/(1+exp(-x)).
 *
 * Logical shapes:
 *   hidden [10240,T], block_input [2560,T], injection [4,T]; T is any positive column extent.
 *
 * Supported domain:
 *   hidden, block_input and every weight are contiguous BF16 (weights QType::BF16, Contiguous
 *   layout, exact shapes above, block_inject [4,10240]); injection is contiguous FP32. Every
 *   tensor and weight plane is 16-byte aligned.
 *
 * Numeric:
 *   The oracle evaluates the complete formula in FP64 from the represented BF16 inputs. N, L,
 *   M and the stream mean are private: their precision, staging and reduction order are
 *   implementation choices. block_input is stored as BF16 and injection as FP32; both are the
 *   only observable rounding boundaries.
 *
 * Effects:
 *   Writes block_input and injection completely; hidden and the weights are unchanged. Outputs
 *   must not overlap each other, hidden, a weight plane or live workspace.
 *
 * Workspace:
 *   Caller-owned transient arena sized by hyper_connection_workspace_capacity_bytes().
 *
 * Execution:
 *   Enqueued on `stream` without host synchronization; valid inside CUDA Graph capture.
 */
void hyper_connection_prepare(const Tensor& hidden, const HyperConnectionWeights& weights,
                              const Weight& block_inject, Tensor& block_input, Tensor& injection,
                              WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Op: four-stream hyper-connection mixer (the final Text and MTP mixers).
 *
 * The same formula, domain, numeric boundaries and workspace as hyper_connection_prepare()
 * without the injection output: block_input[c,t] = (1/4) * sum_s M[s,c] * N[s,c].
 */
void hyper_connection_mix(const Tensor& hidden, const HyperConnectionWeights& weights,
                          Tensor& block_input, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Op: hyper-connection injection of a block output into the four-stream state, in place.
 *
 * Math / indexing:
 *   hidden[s*2560+c,t] <- BF16(hidden[s*2560+c,t] + block_output[c,t] * injection[s,t]).
 *
 * Supported domain:
 *   hidden contiguous BF16 [10240,T], block_output contiguous BF16 [2560,T], injection
 *   contiguous FP32 [4,T], every positive T, 16-byte-aligned storage.
 *
 * Numeric:
 *   The oracle evaluates each sum in FP64 from the represented inputs; the BF16 store of the
 *   updated state is the only rounding boundary.
 *
 * Effects:
 *   hidden is overwritten in place; block_output and injection are unchanged and must not overlap
 *   hidden. No workspace. Enqueued on `stream`; valid inside CUDA Graph capture.
 */
void hyper_connection_inject(const Tensor& block_output, const Tensor& injection, Tensor& hidden,
                             cudaStream_t stream);

/**
 * Op: four-stream hyper-connection state initialization.
 *
 * Math / indexing:
 *   hidden[s*2560+c,t] = x[c,t] for s = 0..3, c = 0..2559: every stream starts as a copy of the
 *   token's input column.
 *
 * Supported domain:
 *   x contiguous BF16 [2560,T], hidden contiguous BF16 [10240,T], every positive T, 16-byte-aligned
 *   storage.
 *
 * Numeric:
 *   Exact: every hidden element is the bit pattern of its source x element.
 *
 * Effects:
 *   hidden is overwritten completely and its incoming value is not read; x is unchanged and must
 *   not overlap hidden. No workspace. Enqueued on `stream`; valid inside CUDA Graph capture.
 */
void hyper_connection_expand(const Tensor& x, Tensor& hidden, cudaStream_t stream);

} // namespace ninfer::ops
