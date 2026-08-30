#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

constexpr bool is_qsa_layer(std::size_t layer) noexcept {
    return layer >= 3 && ((layer - 3) % 4) == 0;
}

constexpr std::size_t qsa_ordinal(std::size_t layer) noexcept { return (layer - 3) / 4; }

constexpr std::size_t gdn_ordinal(std::size_t layer) noexcept {
    return layer - (layer >= 3 ? ((layer - 3) / 4 + 1) : 0);
}

void repeat_embedding_to_hyper_streams(const Tensor& embedding, Tensor& hyper_hidden,
                                       cudaStream_t stream);

void set_qsa_step_metadata(const Tensor& token_indices, const Tensor& mrope_positions,
                           std::int32_t t, std::int32_t table_row, std::int32_t src_slot,
                           std::int32_t dst_slot, Tensor& out_tok_idx, Tensor& out_mrope_pos,
                           Tensor& out_table_row, Tensor& out_src_slot, Tensor& out_dst_slot,
                           cudaStream_t stream);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
