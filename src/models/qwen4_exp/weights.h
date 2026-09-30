#pragma once

#include "models/qwen3_5/weights.h"

#include <optional>
#include <variant>
#include <vector>

namespace ninfer::models::qwen4_exp {

// The weight-handle vocabulary and the Vision tower are shared with Qwen3.5; everything else is
// Qwen4Exp's own model structure.
using qwen3_5::BoundWeight;
using qwen3_5::NormWeights;
using qwen3_5::VisionBlockWeights;
using qwen3_5::VisionWeights;
using qwen3_5::WeightId;
using qwen3_5::WeightUse;
using qwen3_5::WeightUseId;

// One hyper-connection around a block: the RMS norm over the concatenated streams, the low-rank
// input mix and the stream injection of the block output.
struct HyperConnectionWeights {
    WeightId norm, input_mix_down, input_mix_up, block_inject;
};

// The final stream reduction before the output head.
struct HyperMixerWeights {
    WeightId norm, input_mix_down, input_mix_up;
};

// Rows of query_gate_key_value are [query, output gate] per query head, then keys, then values.
struct AttentionWeights {
    WeightId query_gate_key_value, query_norm, key_norm, output;
    WeightId indexer_query_key, indexer_query_norm, indexer_key_norm;
};

// Rows of query_key_value_z are [query, key, value, z]; a_b_projection is [a, b].
struct GdnWeights {
    WeightId query_key_value_z, a_b_projection, a_log, dt_bias, convolution, norm, output;
};

// Expert banks are [experts, 2*intermediate, hidden] with rows [gate, up] per expert, and
// [experts, hidden, intermediate].
struct MoeWeights {
    WeightId router, shared_expert_gate;
    WeightId shared_gate, shared_up, shared_down;
    WeightId experts_gate_up, experts_down;
};

struct BlockWeights {
    HyperConnectionWeights attention_hyper, mlp_hyper;
    std::variant<AttentionWeights, GdnWeights> mixer;
    MoeWeights moe;
};

// Device half of the per-layer n-gram embedding; its Host table is Model::ple_table().
struct PleWeights {
    WeightId key_projection, value_projection;
    WeightId query_norm, key_norm, conv_norm, convolution;
};

struct TextWeights {
    WeightId token_embedding, output_head;
    WeightUseId output_head_use;
    std::vector<BlockWeights> layers;
    PleWeights ple;
    HyperMixerWeights final_mixer;
};

// One full-attention MoE layer; its embedding and output head are the Text ones.
struct MtpWeights {
    WeightId embedding_projection, hidden_projection, embedding_norm, hidden_norm;
    BlockWeights layer;
    HyperMixerWeights final_mixer;
    WeightId token_embedding, output_head;
    WeightUseId output_head_use;
};

// Handles refer to the frozen model's weight array. No artifact ID lookup is needed in execution.
struct ModelWeights {
    TextWeights text;
    std::optional<VisionWeights> vision;
    std::optional<MtpWeights> mtp;
};

} // namespace ninfer::models::qwen4_exp
