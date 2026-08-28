#pragma once

#include "core/tensor.h"
#include "targets/qwen3_8_flash_next/impl/expert_bank.h"
#include "targets/qwen3_8_flash_next/impl/ple_table.h"

#include <array>
#include <cstddef>

namespace ninfer {
class DeviceArena;
}

namespace ninfer::targets::qwen3_8_flash_next::detail {

inline constexpr std::size_t kFullAttentionLayers = 12;
inline constexpr std::size_t kGdnLayers           = 36;

struct HyperConnectionWeights {
    Weight block_inject;
    Tensor norm;
    Weight input_mix_down;
    Weight input_mix_up;
};

struct HyperMixerWeights {
    Tensor norm;
    Weight input_mix_down;
    Weight input_mix_up;
};

struct MoeWeights {
    Weight router;
    Weight shared_down;
    Weight shared_gate;
    Weight shared_up;
    Weight shared_gate_weight;
    Nvfp4ExpertBankView expert_gate_up;
    Nvfp4ExpertBankView expert_down;
};

struct TextLayerWeights {
    HyperConnectionWeights attention_hyper;
    MoeWeights moe;
    HyperConnectionWeights mlp_hyper;
};

struct GdnWeights {
    Tensor a_log;
    Tensor convolution;
    Tensor dt_bias;
    Weight a_b_projection;
    Tensor norm;
    Weight query_key_value_z;
    Weight output;
};

struct AttentionWeights {
    Weight indexer_query_key;
    Tensor indexer_key_norm;
    Tensor indexer_query_norm;
    Tensor key_norm;
    Tensor query_norm;
    Weight query_gate_key_value;
    Weight output;
};

struct PleWeights {
    Tensor convolution;
    Weight key_projection;
    Tensor conv_norm;
    Tensor key_norm;
    Tensor query_norm;
    Weight value_projection;
    PleTableView table;
};

struct TextModelView {
    DeviceArena* weights_arena = nullptr;
    Weight token_embedding;
    std::array<TextLayerWeights, 48> layers;
    std::array<AttentionWeights, kFullAttentionLayers> full_attention;
    std::array<GdnWeights, kGdnLayers> gdn;
    PleWeights ple;
    Weight output_head;
    HyperMixerWeights final_mixer;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
