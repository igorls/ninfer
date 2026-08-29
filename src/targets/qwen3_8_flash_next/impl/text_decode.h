#pragma once

#include "core/arena.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"
#include "targets/qwen3_8_flash_next/impl/text_decode_state.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct FlashNextDecodeStateSink {
    std::function<void(std::string_view name, const Tensor& device_tensor)> on_state;
};

[[nodiscard]] std::size_t
flash_next_text_decode_workspace_capacity_bytes(std::int32_t maximum_blocks, std::int32_t batch);

void flash_next_text_decode_core(const TextModelView& model, const Tensor& embedding,
                                 const Tensor& token_indices, const Tensor& mrope_positions,
                                 const Tensor& table_rows, const Tensor& source_slots,
                                 const Tensor& destination_slots,
                                 const Tensor& gathered_ple_embedding, std::int32_t maximum_blocks,
                                 std::int32_t active_blocks, FlashNextDecodeStateView state,
                                 WorkspaceArena& workspace, Tensor& final_hidden, Tensor& logits,
                                 cudaStream_t stream,
                                 const FlashNextDecodeStateSink* sink = nullptr);

void flash_next_text_decode(const TextModelView& model, const Tensor& token_ids,
                            const Tensor& token_indices, const Tensor& mrope_positions,
                            const Tensor& table_rows, const Tensor& source_slots,
                            const Tensor& destination_slots, const Tensor& gathered_ple_embedding,
                            std::int32_t maximum_blocks, std::int32_t active_blocks,
                            FlashNextDecodeStateView state, WorkspaceArena& workspace,
                            Tensor& final_hidden, Tensor& logits, cudaStream_t stream,
                            const FlashNextDecodeStateSink* sink = nullptr);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
