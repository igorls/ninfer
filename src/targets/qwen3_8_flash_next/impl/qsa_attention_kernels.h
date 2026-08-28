#pragma once

#include "targets/qwen3_8_flash_next/impl/qsa_attention.h"
#include "targets/qwen3_8_flash_next/impl/qsa_attention_workspace.h"

#include <cuda_runtime.h>

namespace ninfer::targets::qwen3_8_flash_next::detail {

void flash_next_qsa_attention_launch(const Tensor& token_indices, const Tensor& mrope_positions,
                                     const Tensor& table_rows, const Tensor& selected_blocks,
                                     const Tensor& selected_counts, const Tensor& query_norm,
                                     const Tensor& key_norm, QsaAttentionCacheView cache,
                                     FlashNextQsaAttentionWorkspace& scratch, cudaStream_t stream);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
