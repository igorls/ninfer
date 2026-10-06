#pragma once

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Op: softmax attention over selected complete four-token blocks and the causal tail.
 *
 * Math / indexing:
 *   For query column c with inclusive cache position p = positions[c], let
 *   complete = floor((p+1)/4), n = counts[c] and S = {selections[j,c] : j < n}. The visible
 *   token set is
 *
 *     V(c) = { 4*s + r : s in S, r = 0..3 }  union  [4*complete, p].
 *
 *   Query head h reads KV head g = floor(h/12). With scale 1/16,
 *
 *     out[d,h,c] = sum_{t in V(c)} softmax_t( dot(q[:,h,c], K[:,g,t]) / 16 ) * V[d,g,t].
 *
 *   The tail holds (p+1) mod 4 tokens. An empty visible set (n = 0 and an empty tail) produces
 *   exact zero.
 *
 *   The selection format is shared with qsa_indexer_select: selections is I32 [512,C]; its first
 *   n entries are unique complete block ids in [0,complete) in any order (qsa_indexer_select writes
 *   them ascending), entries [n,512) are unused; 0 <= n <= min(512,complete).
 *
 * Logical shapes:
 *   q/out BF16 [256,24,C]; positions I32 [C]; selections I32 [512,C]; counts I32 [C]. The batched
 *   form adds table_rows I32 [C] selecting a row of cache.block_tables per column, with C in
 *   [1,48] (up to six verification columns in each of eight sequences). The shared-row form uses one host table row for all C in [1,262144] columns.
 *
 * Supported domain:
 *   Head dimension 256, 24 query heads, 2 KV heads, page-major paged KV exactly as
 *   kv_cache_append writes it ([256,64,2,Nphysical], 64-token pages, logical position p in page
 *   block_tables[p/64, row] at offset p%64). Registered cache profiles:
 *   - KvCacheStorage::BFloat16: K is BF16, V is FP16; K[:,g,t] and V[:,g,t] are their represented
 *     values.
 *   - KvCacheStorage::Fp8E4M3Row256: K and V code planes FP8 E4M3FN [256,64,2,N] with FP16 row
 *     scales [1,64,2,N]; K code rows represent R*K with R the fixed normalized H256 Hadamard
 *     transform used by kv_cache_append, V rows represent V. The Op evaluates
 *     dot(R*q, decode(K)) = dot(q, R^T decode(K)), i.e. the formula above in the cache's stored
 *     basis, with decode(x) = FP32(E4M3FN(code)) * FP32(scale).
 *   These are the v3 KV codecs. They differ from the v2 Flash-Next cache (BF16 V, unscaled
 *   unrotated FP8): FP16 V is exact for normal BF16 values, row-scaled FP8 is a different codec.
 *
 * Numeric:
 *   The oracle evaluates the complete formula in FP64 from represented q and decoded cache values.
 *   BF16 out is the only semantic rounding boundary. Softmax statistics, partition merging,
 *   staging precision of q, probabilities and decoded cache tiles, and reduction association are
 *   private implementation choices.
 *
 * Effects:
 *   Only out is written. No unselected cache value is read. The caller guarantees nonnegative
 *   positions, valid table rows, materialized pages and initialized K/V (and scales) for every
 *   visible token. All inputs, out, and live workspace are pairwise non-overlapping, contiguous and
 *   16-byte aligned.
 *
 * Workspace:
 *   The batched form uses caller-owned transient storage sized by
 *   selected_block_attention_workspace_capacity_bytes(); the shared-row form uses none.
 *
 * Execution:
 *   No device allocation or host/device synchronization; both forms are CUDA Graph capturable
 *   and replay with fresh device inputs.
 */
void selected_block_attention(const Tensor& q, const Tensor& positions, const Tensor& table_rows,
                              const Tensor& selections, const Tensor& counts,
                              const PagedKVBatchLayerView& cache, WorkspaceArena& workspace,
                              Tensor& out, cudaStream_t stream);

/**
 * Minimum transient capacity of the batched form for every column count C in
 * [min_columns,max_columns] (1 <= min_columns <= max_columns <= 48). Invalid intervals throw.
 */
[[nodiscard]] std::size_t selected_block_attention_workspace_capacity_bytes(
    std::int32_t min_columns, std::int32_t max_columns);

/**
 * Shared-row form of the same formula: every column reads cache.block_table (one sequence). C is
 * in [1,262144]. No transient workspace.
 */
void selected_block_attention(const Tensor& q, const Tensor& positions, const Tensor& selections,
                              const Tensor& counts, const PagedKVLayerView& cache, Tensor& out,
                              cudaStream_t stream);

} // namespace ninfer::ops
