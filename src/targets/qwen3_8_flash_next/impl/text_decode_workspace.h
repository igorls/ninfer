#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "targets/qwen3_8_flash_next/impl/gdn_workspace.h"
#include "targets/qwen3_8_flash_next/impl/hyper_workspace.h"
#include "targets/qwen3_8_flash_next/impl/moe_workspace.h"
#include "targets/qwen3_8_flash_next/impl/ple_workspace.h"
#include "targets/qwen3_8_flash_next/impl/qsa_attention_workspace.h"
#include "targets/qwen3_8_flash_next/impl/qsa_indexer_workspace.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct FlashNextTextDecodeWorkspace {
    Tensor hyper_hidden;    // BF16 [10240, tokens]
    Tensor block_input;     // BF16 [2560, tokens]
    Tensor block_output;    // BF16 [2560, tokens]
    Tensor ple_injection;   // BF16 [10240, tokens]
    Tensor selected_blocks; // I32 [512, tokens]
    Tensor selected_counts; // I32 [tokens]
    FlashNextHyperWorkspace hyper_scratch;
    Tensor qsa_token_indices;
    Tensor qsa_mrope_positions;
    Tensor qsa_table_rows;
    Tensor qsa_source_slots;
    Tensor qsa_destination_slots;
    Tensor qsa_selected_blocks;
    Tensor qsa_selected_counts;
    FlashNextHyperWorkspace single_token_hyper_scratch;
};

template <class Arena>
FlashNextTextDecodeWorkspace allocate_flash_next_text_decode_workspace(Arena& arena,
                                                                       std::int32_t tokens) {
    FlashNextTextDecodeWorkspace ws{};
    ws.hyper_hidden               = arena.alloc(DType::BF16, {10'240, tokens}, 256);
    ws.block_input                = arena.alloc(DType::BF16, {2'560, tokens}, 256);
    ws.block_output               = arena.alloc(DType::BF16, {2'560, tokens}, 256);
    ws.ple_injection              = arena.alloc(DType::BF16, {10'240, tokens}, 256);
    ws.selected_blocks            = arena.alloc(DType::I32, {512, tokens}, 256);
    ws.selected_counts            = arena.alloc(DType::I32, {tokens}, 256);
    ws.hyper_scratch              = allocate_flash_next_hyper_workspace(arena, tokens);
    ws.qsa_token_indices          = arena.alloc(DType::I32, {1}, 16);
    ws.qsa_mrope_positions        = arena.alloc(DType::I32, {1, 3}, 16);
    ws.qsa_table_rows             = arena.alloc(DType::I32, {1}, 16);
    ws.qsa_source_slots           = arena.alloc(DType::I32, {1}, 16);
    ws.qsa_destination_slots      = arena.alloc(DType::I32, {1}, 16);
    ws.qsa_selected_blocks        = arena.alloc(DType::I32, {512, 1}, 256);
    ws.qsa_selected_counts        = arena.alloc(DType::I32, {1}, 16);
    ws.single_token_hyper_scratch = allocate_flash_next_hyper_workspace(arena, 1);
    return ws;
}

// Per-chunk staging tensors the executor carves from the workspace before the prefill core runs.
// Decode keeps these in persistent round tensors (sized by max_concurrency); a chunk can be far
// wider, so they come from the arena. The estimate in
// flash_next_text_prefill_workspace_capacity_bytes must allocate exactly this layout, which is
// why both sides call this helper instead of listing the tensors twice.
struct FlashNextPrefillChunkStaging {
    Tensor gathered_ple;    // BF16 [2560, tokens]
    Tensor token_ids;       // I32 [tokens]
    Tensor token_indices;   // I32 [tokens]
    Tensor mrope_positions; // I32 [3, tokens]
    Tensor embedding;       // BF16 [2560, tokens]
};

template <class Arena>
FlashNextPrefillChunkStaging allocate_flash_next_prefill_chunk_staging(Arena& arena,
                                                                       std::int32_t tokens) {
    FlashNextPrefillChunkStaging staging{};
    staging.gathered_ple    = arena.alloc(DType::BF16, {2'560, tokens}, 256);
    staging.token_ids       = arena.alloc(DType::I32, {tokens}, 256);
    staging.token_indices   = arena.alloc(DType::I32, {tokens}, 256);
    staging.mrope_positions = arena.alloc(DType::I32, {3, tokens}, 256);
    staging.embedding       = arena.alloc(DType::BF16, {2'560, tokens}, 256);
    return staging;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
