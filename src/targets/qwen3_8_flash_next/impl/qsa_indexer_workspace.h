#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct FlashNextQsaIndexerWorkspace {
    Tensor projected;
    Tensor query;
    Tensor scores;
    Tensor sorted_scores;
    Tensor ids;
    Tensor sorted_ids;
    Tensor offsets;
    DeviceSpan sort_temp;
};

template <class Arena>
FlashNextQsaIndexerWorkspace
allocate_flash_next_qsa_indexer_workspace(Arena& arena, std::int32_t maximum_blocks,
                                          std::int32_t tokens, std::size_t sort_temp_bytes) {
    return {
        .projected     = arena.alloc(DType::BF16, {640, tokens}, 256),
        .query         = arena.alloc(DType::BF16, {128, 4, tokens}, 256),
        .scores        = arena.alloc(DType::FP32, {maximum_blocks, tokens}, 256),
        .sorted_scores = arena.alloc(DType::FP32, {maximum_blocks, tokens}, 256),
        .ids           = arena.alloc(DType::I32, {maximum_blocks, tokens}, 256),
        .sorted_ids    = arena.alloc(DType::I32, {maximum_blocks, tokens}, 256),
        .offsets       = arena.alloc(DType::I32, {tokens + 1}, 16),
        .sort_temp     = arena.alloc_bytes(sort_temp_bytes, 256),
    };
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
