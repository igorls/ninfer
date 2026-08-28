#pragma once

#include "core/tensor.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"
#include "targets/qwen3_8_flash_next/impl/qsa_attention.h"
#include "targets/qwen3_8_flash_next/impl/qsa_indexer.h"

#include <array>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct FlashNextDecodeStateView {
    std::array<QsaIndexerCacheView, kFullAttentionLayers> qsa_indexer_caches;
    std::array<QsaAttentionCacheView, kFullAttentionLayers> qsa_attention_caches;
    std::array<Tensor, kGdnLayers> gdn_convolution_states; // BF16 [10240, 3, state_slots]
    std::array<Tensor, kGdnLayers> gdn_ssm_states;         // FP32 [128, 128, 48, state_slots]
    Tensor ple_convolution_states;                         // BF16 [10240, 9, state_slots]
};

void validate_flash_next_decode_state(const FlashNextDecodeStateView& state,
                                      std::int32_t state_slots);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
