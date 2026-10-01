#pragma once

#include "models/load_options.h"
#include "models/qwen3_5/config.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::artifact {
struct Directory;
}

namespace ninfer::models::qwen4_exp {

// Leaf value types whose mathematics Qwen4Exp shares with Qwen3.5 (GDN core, attention heads,
// interleaved MRoPE, the Vision tower). Their Qwen4Exp values are parsed here.
using qwen3_5::AttentionConfig;
using qwen3_5::GdnConfig;
using qwen3_5::MixerKind;
using qwen3_5::MoeConfig;
using qwen3_5::RopeConfig;
using qwen3_5::VisionConfig;

// Four residual streams mixed through a low-rank projection around every block.
struct HyperConnectionConfig {
    std::uint32_t streams  = 0;
    std::uint32_t low_rank = 0;
};

// Query-selected attention: an indexer scores compressed key blocks and selects a token budget.
struct IndexerConfig {
    std::uint32_t num_heads           = 0;
    std::uint32_t num_key_value_heads = 0;
    std::uint32_t head_dim            = 0;
    std::uint32_t compress_ratio      = 0;
    std::uint32_t budget              = 0;
};

// Per-layer n-gram embeddings injected into one decoder layer.
struct PleConfig {
    std::uint32_t layer            = 0; // 0-based; the source ple_layer_ids counts from 1.
    std::uint32_t embed_dim        = 0;
    std::uint32_t conv_kernel_size = 0;
    std::uint32_t ngram_size       = 0;
    std::uint32_t heads_per_ngram  = 0;
    std::uint32_t shards           = 0;
    // Token that resets the n-gram history (the source text config's eos_token_id); history
    // begins as [boundary, boundary].
    std::int32_t boundary_token = 0;

    [[nodiscard]] std::uint32_t heads() const noexcept {
        return (ngram_size - 1) * heads_per_ngram;
    }

    [[nodiscard]] std::uint32_t head_width() const noexcept { return embed_dim / heads(); }
};

struct TextConfig {
    std::uint32_t hidden_size             = 0;
    std::uint32_t vocab_size              = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t max_position_embeddings = 0;
    float rms_norm_eps                    = 0;
    std::vector<MixerKind> layer_types;
    std::vector<std::uint32_t> compact_layer_indices;
    std::uint32_t full_attention_layers   = 0;
    std::uint32_t linear_attention_layers = 0;
    AttentionConfig attention;
    RopeConfig rope_parameters;
    GdnConfig gdn;
    MoeConfig moe;
    HyperConnectionConfig hyper_connection;
    IndexerConfig indexer;
    PleConfig ple;

    [[nodiscard]] std::uint64_t stream_width() const noexcept {
        return std::uint64_t(hyper_connection.streams) * hidden_size;
    }
};

struct Config {
    TextConfig text;
    std::optional<VisionConfig> vision;
    bool mtp = false;
};

[[nodiscard]] Config parse_config(const artifact::Directory& directory, const LoadOptions& options);

} // namespace ninfer::models::qwen4_exp
