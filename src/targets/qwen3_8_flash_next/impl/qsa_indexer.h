#pragma once

#include "core/arena.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct QsaIndexerCacheView {
    Tensor block_keys;    // BF16 [128,64,physical_pages]
    Tensor block_tables;  // I32 [logical_pages,table_rows]
    Tensor raw_keys;      // BF16 [128,4,state_slots]
    Tensor raw_positions; // I32 [3,4,state_slots]
};

[[nodiscard]] std::size_t
flash_next_qsa_indexer_workspace_capacity_bytes(std::int32_t maximum_blocks, std::int32_t batch);

// Updates the current raw-key/compressed-block state and returns the best complete block IDs.
// The incomplete visible tail is implicit in token_indices and is not included in selected_blocks.
// active_blocks is the host-selected decode envelope (the CUDA-graph context bucket, or the
// live complete-block frontier on the eager test hook). It covers every row's complete block
// count and is bounded by the startup-fixed maximum_blocks capacity. Padded score slots in
// [complete_blocks, active_blocks) are written -inf so they cannot enter the TopK set.
// When active_blocks <= 512 every complete block is selected: the indexer publishes the identity
// 0..complete_blocks-1 (padded with -1) and skips scoring and selection. The selected SET
// matches the sorted path; the published order is identity rather than score-descending.
// When active_blocks > 512, selection is packed-key DeviceTopK::MaxPairs (k=512) followed by a
// 512-wide stable descending reorder so the published SET and ORDER match the segmented radix sort.
void flash_next_qsa_indexer_decode(const Tensor& input, const AttentionWeights& weights,
                                   const Tensor& token_indices, const Tensor& mrope_positions,
                                   const Tensor& table_rows, const Tensor& source_state_slots,
                                   const Tensor& destination_state_slots, QsaIndexerCacheView cache,
                                   std::int32_t maximum_blocks, std::int32_t active_blocks,
                                   WorkspaceArena& workspace, Tensor& selected_blocks,
                                   Tensor& selected_counts, cudaStream_t stream);

void flash_next_qsa_indexer_prefill_chunk(
    const Tensor& input, const AttentionWeights& weights, const Tensor& token_indices,
    const Tensor& mrope_positions, std::int32_t table_row, std::int32_t source_state_slot,
    std::int32_t destination_state_slot, QsaIndexerCacheView cache, std::int32_t maximum_blocks,
    WorkspaceArena& workspace, Tensor& selected_blocks, Tensor& selected_counts,
    cudaStream_t stream);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
