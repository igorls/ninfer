#pragma once

#include "models/load_options.h"
#include "models/qwen3_5/config.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace ninfer::artifact {
struct Directory;
}

namespace ninfer::models::qwen4_exp {

// Companion component identities the converter writes and this package executes. The Text pair
// (Qwen4ExpForCausalLM / qwen4_exp_text) selects the package in models/registry.
inline constexpr std::string_view kMtpArchitecture = "Qwen4ExpMTP";
inline constexpr std::string_view kVisionModelType = "qwen4_exp_vision";

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

// Per-layer n-gram embeddings (PLE), injected before each listed decoder layer. Every PLE layer
// owns its hashed n-gram table, projections, norms, dilated convolution and convolution history;
// the geometry below is shared.
struct PleConfig {
    // Ascending, distinct, 0-based GDN decoder layers; the source ple_layer_ids counts from 1.
    std::vector<std::uint32_t> layers;
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

    // Stored u4z8_g16_fp16 row of one n-gram head: two codes per byte, one FP16 scale per 16.
    [[nodiscard]] std::size_t code_row_bytes() const noexcept { return head_width() / 2U; }

    [[nodiscard]] std::size_t scale_row_bytes() const noexcept {
        return head_width() / 16U * sizeof(std::uint16_t);
    }

    // Columns the causal convolution (conv_kernel_size taps at dilation ngram_size) keeps.
    [[nodiscard]] std::uint32_t history_columns() const noexcept {
        return (conv_kernel_size - 1) * ngram_size;
    }

    // Position of decoder `layer` in `layers`, or nullopt when the layer carries no PLE.
    [[nodiscard]] std::optional<std::size_t> index_of(std::uint32_t layer) const noexcept {
        for (std::size_t i = 0; i < layers.size(); ++i) {
            if (layers[i] == layer) { return i; }
        }
        return std::nullopt;
    }
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
