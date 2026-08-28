#pragma once

#include "core/arena.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct QsaAttentionCacheView {
    Tensor key_pages;    // BF16 [256,64,2,physical_pages]
    Tensor value_pages;  // BF16 [256,64,2,physical_pages]
    Tensor block_tables; // I32 [logical_pages,table_rows]
};

[[nodiscard]] std::size_t flash_next_qsa_attention_workspace_capacity_bytes(std::int32_t batch);

void flash_next_qsa_attention_decode(const Tensor& input, const AttentionWeights& weights,
                                     const Tensor& token_indices, const Tensor& mrope_positions,
                                     const Tensor& table_rows, const Tensor& selected_blocks,
                                     const Tensor& selected_counts, QsaAttentionCacheView cache,
                                     WorkspaceArena& workspace, Tensor& output,
                                     cudaStream_t stream);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
