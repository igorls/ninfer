#pragma once

#include "core/arena.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"
#include "targets/qwen3_8_flash_next/impl/qsa_attention.h"
#include "targets/qwen3_8_flash_next/impl/text_decode.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

[[nodiscard]] std::size_t flash_next_mtp_workspace_capacity_bytes(std::int32_t batch);

void flash_next_mtp_step(const TextModelView& model, const Tensor& input_embedding,
                         const Tensor& backbone_hyper_hidden, const Tensor& token_indices,
                         const Tensor& mrope_positions, const Tensor& table_rows,
                         const Tensor& selected_blocks, const Tensor& selected_counts,
                         QsaAttentionCacheView mtp_cache, WorkspaceArena& workspace,
                         Tensor& draft_logits, Tensor& draft_tokens, cudaStream_t stream,
                         const FlashNextDecodeStateSink* sink = nullptr,
                         Tensor* out_hyper_hidden             = nullptr);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
