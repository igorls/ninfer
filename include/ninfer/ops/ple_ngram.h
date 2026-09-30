#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Op: decode of gathered u4z8_g16_fp16 per-layer-embedding rows.
 *
 * Math / indexing:
 *   For row r and column k (k = 0..159, group g = floor(k/16)):
 *     u          = (codes[k/2, r] >> (4 * (k % 2))) & 0xf      (lower k in the low nibble)
 *     w_hat[k,r] = binary32(binary32(u - 8) * binary32(scales[g, r]))
 *     out[k,r]   = BF16_RNE(w_hat[k,r]).
 *   A caller that gathers 16 rows per token views out [160,16*T] as the embedding [2560,T].
 *
 * Logical shapes:
 *   codes U8 [80,R], scales FP16 [10,R] (one binary16 multiplier per group of 16), out BF16
 *   [160,R], R >= 1. These are the logical-row views of the packed_u4_g16_v1 layout
 *   (docs/maintainer/storage-layouts.md section 8) with K = 160.
 *
 * Numeric:
 *   Exact: the represented binary32 weight and its BF16 round-to-nearest-even cast are the
 *   observable result, compared bit for bit.
 *
 * Effects:
 *   Writes out completely; codes and scales are unchanged and must not overlap out. All
 *   tensors are contiguous and 16-byte aligned. No workspace; valid inside CUDA Graph capture.
 */
void ple_ngram_decode(const Tensor& codes, const Tensor& scales, Tensor& out, cudaStream_t stream);

/**
 * Op: gated per-layer-embedding combine with a dilated causal convolution (Flash-Next PLE).
 *
 * Math / indexing:
 *   Per column t (one token) and stream s = 0..3, with X_s = X[s*2560 : (s+1)*2560, t] and the
 *   one-centered RMSNorm n(x, w) = x / sqrt(mean(x^2) + 1e-6) * (1 + w):
 *     raw_s  = dot(n(key_s, key_norm_s), n(hidden_s, query_norm_s)) / sqrt(2560)
 *     gate_s = sign(raw_s) * sqrt(max(|raw_s|, 1e-6))
 *     V_s    = sigmoid(gate_s) * value[:, t]
 *     u_t    = BF16_RNE(n(V_s, conv_norm_s))                   (all four streams, 10240 wide)
 *     conv   = sum_{i=0..3} conv_weight[:, i] * u_{t - 9 + 3 i}     (channel-wise)
 *     out[:, t] = BF16_RNE(V + silu(conv)).
 *   u_{t-9..t-1} before the first column come from the input state, which holds the previous
 *   nine u columns in order (oldest first). The new state is the last nine u columns of
 *   concat(state_in, u_0..u_{T-1}).
 *
 * Logical shapes:
 *   hidden and key [10240,T], value [2560,T], out [10240,T], query_norm, key_norm and
 *   conv_norm [10240], conv_weight [10240,4] (tap i at ne[1] = i), state [10240,9].
 *
 * Supported domain:
 *   BF16 everywhere, contiguous, 16-byte aligned; every positive T.
 *
 * Numeric:
 *   The persistent state representation BF16(u) is semantic: the convolution consumes the
 *   rounded u for history and in-call columns alike. The oracle evaluates everything else in
 *   FP64 from the represented inputs; V, the norm statistics and the convolution sum are
 *   private precision. out is the only other rounding boundary.
 *
 * Effects:
 *   out is overwritten. state_out receives the new state; state_in and state_out are either
 *   disjoint or exactly the same storage. No other input is modified, and out overlaps no input
 *   or state. Enqueued on `stream`; valid inside CUDA Graph capture.
 *
 * Workspace:
 *   Caller-owned arena sized by ple_ngram_workspace_capacity_bytes().
 */
[[nodiscard]] std::size_t ple_ngram_workspace_capacity_bytes(std::int32_t min_columns,
                                                             std::int32_t max_columns);

void ple_ngram(const Tensor& hidden, const Tensor& key, const Tensor& value,
               const Tensor& query_norm, const Tensor& key_norm, const Tensor& conv_norm,
               const Tensor& conv_weight, const Tensor& state_in, Tensor& state_out, Tensor& out,
               WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Batched snapshot form of ple_ngram() for independent rows (decode and speculative
 * verification).
 *
 * hidden and key are BF16 [10240,W,B], value [2560,W,B], out [10240,W,B]; `states` is BF16
 * [10240,9,Slots]; initial_slots and snapshot_base_slots are I32 [B]; valid_columns is I32 [B]
 * with values in [1,W], or an empty Tensor meaning W valid columns for every row. B = 1..8.
 * Row b starts from state slot initial_slots[b]; after its valid column j it writes the new state
 * to slot snapshot_base_slots[b] + j. Invalid-tail output columns are BF16 zero and mutate no
 * state. The caller reserves [base, base + W) for every row; reservations are pairwise disjoint
 * and no row overwrites another row's initial slot (a row's own initial slot may lie in its
 * reservation). Formula, numeric boundaries and workspace are those of ple_ngram() applied to
 * each row as a sequence of W columns.
 */
void ple_ngram_snapshot(const Tensor& hidden, const Tensor& key, const Tensor& value,
                        const Tensor& query_norm, const Tensor& key_norm, const Tensor& conv_norm,
                        const Tensor& conv_weight, Tensor& states, const Tensor& valid_columns,
                        const Tensor& initial_slots, const Tensor& snapshot_base_slots,
                        Tensor& out, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
