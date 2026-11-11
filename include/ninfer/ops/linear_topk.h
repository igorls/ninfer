#pragma once

#include "core/device.h"
#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"


#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the caller-owned transient capacity required by linear_topk for every column count in the
 * inclusive `[min_columns,max_columns]` interval. The registered profile is identified by
 * its weight format and exact `[head_rows,input_rows]` geometry. Column counts are positive.
 */
[[nodiscard]] std::size_t linear_topk_workspace_capacity_bytes(QType qtype, std::int32_t head_rows,
                                                               std::int32_t input_rows,
                                                               std::int32_t min_columns,
                                                               std::int32_t max_columns);

/**
 * @brief Projects independent matrix columns through a full vocabulary head and returns
 * the stable top sixteen scores and global token ids per column.
 *
 * @details For any positive column count `U`, `hidden` is contiguous BF16 `[5120,U]`, `head` is
 * Q8_G32_FP16, FP8_E4M3FN_ROW_BF16, or contiguous BF16 `[248320,5120]`, and `valid_rows` is
 * 248077. For every
 * `t in [0,U)` and valid vocabulary row `v`, the ideal score is
 *
 * @f[
 *   s_{v,t}=\sum_{k=0}^{5119}\mathrm{FP32Dequant}(head)_{v,k}
 *                              \mathrm{FP32}(hidden_{k,t}).
 * @f]
 *
 * `candidate_scores` is contiguous FP32 `[16,U]` and `candidate_ids` is contiguous I32
 * `[16,U]`. Rank is stored fastest; a caller may view the output as `[16,K,B]` when `U=K*B`.
 * Candidates are ordered by descending computed score, with exact score ties resolved by lower
 * global token id. Physical rows `[valid_rows,248320)` never participate. Projection uses the
 * registered A16 arithmetic profile; candidate scores are returned directly in FP32 and no dense
 * vocabulary-logit tensor is an observable intermediate.
 *
 * All tensors and every weight plane are preserved except that both output tensors are completely
 * overwritten. Inputs, outputs, weight planes, and workspace must not overlap. The Op has no
 * persistent state and performs no internal device allocation.
 */
void linear_topk(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                 Tensor& candidate_ids, Tensor& candidate_scores, MemoryArena& workspace,
                 CudaExecutionView execution);

/**
 * @brief Projects through the optimized 131072-row proposal head and returns stable top sixteen
 * scores with shortlist rows mapped to global token ids.
 *
 * @details The tensor contract is the same as the full-head overload except that `head` is
 * Q4_G64_FP16 `[131072,5120]` and every head row participates. `row_to_global_ids` is contiguous
 * I32 `[131072]`; it contains distinct ids in `[0,248077)` and maps each local head row to the id
 * used for output and tie-breaking. Artifact binding establishes the map's range and uniqueness.
 */
void linear_topk(const Tensor& hidden, const Weight& head, const Tensor& row_to_global_ids,
                 Tensor& candidate_ids, Tensor& candidate_scores, MemoryArena& workspace,
                 CudaExecutionView execution);

// Capacity for the key-producing overload below, for every positive column extent in
// [min_columns,max_columns]. The query includes activation encoding and all candidate merges.
[[nodiscard]] std::size_t linear_topk_workspace_capacity_bytes(QType qtype, std::int32_t head_rows,
                                                               std::int32_t input_rows,
                                                               LinearPolicy policy,
                                                               std::int32_t min_columns,
                                                               std::int32_t max_columns);

/**
 * Local vocabulary top-16 as lossless FP32 score/global-ID keys, following vocabulary.h.
 * hidden is contiguous BF16 [K,D*B]; keys is contiguous I64 [16,D,B], D>0, B=1..8,
 * with candidate rank fastest. Both bases are 16-byte aligned.
 * Row v of native W [N,K] has global ID token_begin+v; only IDs below token_domain
 * participate. token_begin>=0, token_domain>0, token_begin+N<=INT_MAX. Empty support
 * produces zero keys. Lists are descending by computed FP32 score, ties by smaller global ID;
 * signed zero is canonicalized. Consumed inputs and scores are finite.
 *
 * Policy controls activation permission; MXFP8 requires an A8 permission. This overload uses
 * contiguous vocabulary shards, including Q4, and takes no row-to-ID map. Projection scores
 * have no BF16/FP16 boundary. Every output is overwritten and inputs are preserved.
 * All input, output, weight planes and live caller workspace are disjoint on execution.device().
 * Workspace is call-scoped, covers the complete operation and has no retained storage or hidden
 * device allocation. Descriptors/scalars are capture-time values; Graph replay reads current data.
 */
void linear_topk(const Tensor& hidden, const Weight& head, std::int32_t token_begin,
                 std::int32_t token_domain, LinearPolicy policy, Tensor& keys,
                 MemoryArena& workspace, CudaExecutionView execution);

} // namespace ninfer::ops
