#pragma once

#include "core/device.h"

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Op: Candidate token log-probabilities
 *
 * Math / indexing:
 *   Let l[r,c] be the exact real value represented by logits[r,c], and allowed(r,c) be bit r
 *   of column c of `allowed` (bit r of word r/32, LSB first) when `allowed` is given, else
 *   true. For every column c and every token id t of interest,
 *
 *     raw[t,c]    = l[t,c] - log(sum_{r<valid_rows} exp(l[r,c]))
 *     masked[t,c] = l[t,c] - log(sum_{r<valid_rows, allowed(r,c)} exp(l[r,c]))  if allowed(t,c)
 *                 = -inf                                                          otherwise
 *
 *   The ids of interest are sampled_ids[c] for column c, whose two values land in
 *   sampled_out[0,c] (raw) and sampled_out[1,c] (masked); every candidate_ids[n], the same
 *   list for every column, whose values land in candidates_out[0,n,c] and candidates_out[1,n,c];
 *   and, when top_ids_out/top_out are given, the K most likely allowed tokens of column c ordered
 *   by descending l[r,c] with ties broken by ascending token id r, whose token ids land in
 *   top_ids_out[k,c] and whose raw/masked log-probabilities land in top_out[0,k,c] and
 *   top_out[1,k,c] (or id -1 and log-probabilities -inf for positions k beyond the column's
 *   allowed token count).
 *
 * Logical shapes:
 *   logits is [physical_rows,C] with C>0 and 1<=valid_rows<=physical_rows; sampled_ids is I32
 *   [C]; candidate_ids is I32 [N] with N>0 when given; allowed is I32 [mask_words] with
 *   mask_words*32>=valid_rows when given (broadcast), or [mask_words,C] for per-column masks;
 *   sampled_out is FP32 [2,C]; candidates_out is FP32 [2,N,C], required exactly when
 *   candidate_ids is given; top_ids_out is I32 [K,C] and top_out is FP32 [2,K,C] with 1<=K<=20,
 *   required together when top alternatives are requested. Every input id is in [0,valid_rows).
 *   Rows [valid_rows,physical_rows) do not participate.
 *
 * Supported domain:
 *   logits is contiguous finite BF16; ids and allowed are contiguous I32; outputs are contiguous
 *   FP32 (and I32 for top_ids_out). Storage has its dtype's natural alignment. A column whose
 *   allowed set is empty is a caller error and yields NaN in its masked values.
 *
 * Numeric:
 *   Outputs are FP32 approximations of the formulas above. Reduction association and private
 *   accumulator precision are implementation choices; the independent oracle evaluates the full
 *   formulas in FP64 from the represented BF16 inputs.
 *
 * Effects:
 *   Writes every output element and preserves every input. Outputs must not overlap any input
 *   or each other.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues work on stream and owns no persistent state.
 */
void candidate_logprobs(const Tensor& logits, std::int32_t valid_rows, const Tensor& sampled_ids,
                        const Tensor* candidate_ids, const Tensor* allowed, Tensor& sampled_out,
                        Tensor* candidates_out, CudaExecutionView execution, Tensor* top_ids_out = nullptr,
                        Tensor* top_out = nullptr);

} // namespace ninfer::ops
