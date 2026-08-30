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

} // namespace ninfer::targets::qwen3_8_flash_next::detail
