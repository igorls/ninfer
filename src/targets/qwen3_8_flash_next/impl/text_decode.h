#pragma once

#include <span>
#include "core/arena.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"
#include "targets/qwen3_8_flash_next/impl/text_decode_state.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// Device readout of the next-token distribution at chosen positions of one prefill chunk: the
// 4-stream hidden rows are gathered, passed through the final mixer and the output head a tile
// at a time, and each column resolved by ops::candidate_logprobs into its own block of
// (2 + 2 * candidates) floats. Positions are chunk-local, ascending; `logits` is a full-width
// column buffer the tiles may use before the chunk's own final logits are written.
struct FlashNextPromptReadout {
    std::span<const std::int32_t> local_positions;
    const std::int32_t* next_ids = nullptr;   // device I32, one per listed position
    const std::int32_t* candidate_ids = nullptr; // device I32 [candidates], may be null
    std::int32_t candidates = 0;
    float* readout = nullptr;                 // device blocks, one per listed position
    Tensor logits;                            // BF16 [248320, width]
};

struct FlashNextDecodeStateSink {
    std::function<void(std::string_view name, const Tensor& device_tensor)> on_state;
};

[[nodiscard]] std::size_t
flash_next_text_decode_workspace_capacity_bytes(std::int32_t maximum_blocks, std::int32_t batch, bool mtp = false);

[[nodiscard]] std::size_t
flash_next_text_prefill_workspace_capacity_bytes(std::int32_t maximum_blocks, std::int32_t tokens, bool mtp = false);

void flash_next_text_decode_core(const TextModelView& model, const Tensor& embedding,
                                 const Tensor& token_indices, const Tensor& mrope_positions,
                                 const Tensor& table_rows, const Tensor& source_slots,
                                 const Tensor& destination_slots,
                                 const Tensor& gathered_ple_embedding, std::int32_t maximum_blocks,
                                 std::int32_t active_blocks, FlashNextDecodeStateView state,
                                 WorkspaceArena& workspace, Tensor& final_hidden, Tensor& logits,
                                 cudaStream_t stream,
                                 const FlashNextDecodeStateSink* sink = nullptr,
                                 Tensor* out_hyper_hidden             = nullptr,
                                 bool aliased_recurrent_scan          = false,
                                 const Tensor* mtp_token_ids          = nullptr);

void flash_next_text_decode(const TextModelView& model, const Tensor& token_ids,
                            const Tensor& token_indices, const Tensor& mrope_positions,
                            const Tensor& table_rows, const Tensor& source_slots,
                            const Tensor& destination_slots, const Tensor& gathered_ple_embedding,
                            std::int32_t maximum_blocks, std::int32_t active_blocks,
                            FlashNextDecodeStateView state, WorkspaceArena& workspace,
                            Tensor& final_hidden, Tensor& logits, cudaStream_t stream,
                            const FlashNextDecodeStateSink* sink = nullptr);

void flash_next_text_prefill_chunk(const TextModelView& model, const Tensor& embedding,
                                   const Tensor& token_indices, const Tensor& mrope_positions,
                                   std::int32_t table_row, std::int32_t source_slot,
                                   std::int32_t destination_slot,
                                   const Tensor& gathered_ple_embedding, std::int32_t maximum_blocks,
                                   std::int32_t first_token_index, FlashNextDecodeStateView state,
                                   WorkspaceArena& workspace, Tensor& final_hidden, Tensor& logits,
                                   cudaStream_t stream,
                                   const FlashNextDecodeStateSink* sink = nullptr,
                                   bool use_qsa_prefill_mma            = false,
                                   Tensor* out_hyper_hidden            = nullptr,
                                   const FlashNextPromptReadout* prompt_readout = nullptr,
                                   const Tensor* mtp_token_ids         = nullptr);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
