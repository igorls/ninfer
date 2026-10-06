#include "models/qwen4_exp/config.h"

#include "artifact/schema.h"
#include "models/registry.h"

#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace ninfer::models::qwen4_exp {
namespace {

// text_config.eos_token_id of the pinned Qwen/Qwen3.8-Flash-Next source config.
constexpr std::int32_t kPleBoundaryToken = 248044;

using artifact::ArtifactError;
using artifact::Json;
using artifact::require_members;

std::uint32_t integer(const Json& value, std::string_view label, bool positive = true) {
    const auto n = artifact::require_u64(value, label, positive);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(label) + ": config dimension exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

std::uint32_t dimension(const Json& config, const char* name) {
    return integer(config.at(name), name);
}

float positive_float(const Json& config, const char* name) {
    const auto& value = config.at(name);
    if (!value.is_number()) { throw ArtifactError(std::string(name) + " must be a real value"); }
    const float result = value.get<float>();
    if (!std::isfinite(result) || result <= 0) {
        throw ArtifactError(std::string(name) + " must be positive finite FP32");
    }
    return result;
}

std::string architecture(const Json& config) {
    const auto& names = config.at("architectures");
    if (!names.is_array() || names.size() != 1) {
        throw ArtifactError("architectures must name one implementation");
    }
    return artifact::require_id(names[0], "architecture");
}

const artifact::Component& companion(const artifact::Directory& directory, std::string_view name) {
    const auto& component = directory.component(name);
    if (component.target != "text") {
        throw ArtifactError(std::string(name) + ": target must be text");
    }
    return component;
}

// Interleaved MRoPE: rotary pair p belongs to axis 1 or 2 when p%3 selects that axis inside its
// section, and to axis 0 otherwise.
RopeConfig rope(const Json& value, std::uint32_t head_dim) {
    require_members(value, {"rope_theta", "partial_rotary_factor", "mrope_section"}, {},
                    "text RoPE");
    RopeConfig out;
    out.rope_theta            = positive_float(value, "rope_theta");
    out.partial_rotary_factor = positive_float(value, "partial_rotary_factor");
    if (out.partial_rotary_factor > 1) { throw ArtifactError("partial_rotary_factor exceeds one"); }
    out.rotary_dim = static_cast<std::uint32_t>(double(head_dim) * out.partial_rotary_factor);
    if (!out.rotary_dim || out.rotary_dim % 2) {
        throw ArtifactError("rotary dimension must be positive and even");
    }
    const auto& sections = value.at("mrope_section");
    if (!sections.is_array() || sections.size() != 3) {
        throw ArtifactError("MRoPE requires three sections");
    }
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        out.mrope_section[i] = integer(sections[i], "MRoPE section", false);
        sum += out.mrope_section[i];
    }
    if (sum != out.rotary_dim / 2) {
        throw ArtifactError("MRoPE sections differ from the rotary width");
    }
    std::array<std::uint32_t, 3> counts{};
    for (std::uint32_t pair = 0; pair < out.rotary_dim / 2; ++pair) {
        const std::uint8_t axis = pair % 3 == 1 && pair < 3ULL * out.mrope_section[1]   ? 1
                                  : pair % 3 == 2 && pair < 3ULL * out.mrope_section[2] ? 2
                                                                                        : 0;
        out.pair_axes.push_back(axis);
        ++counts[axis];
    }
    if (counts != out.mrope_section) {
        throw ArtifactError("MRoPE sections exceed interleaved axis ranges");
    }
    return out;
}

TextConfig text(const Json& value) {
    require_members(value,
                    {"architectures",
                     "model_type",
                     "hidden_size",
                     "vocab_size",
                     "num_hidden_layers",
                     "max_position_embeddings",
                     "num_attention_heads",
                     "num_key_value_heads",
                     "head_dim",
                     "linear_num_key_heads",
                     "linear_key_head_dim",
                     "linear_num_value_heads",
                     "linear_value_head_dim",
                     "linear_conv_kernel_dim",
                     "num_experts",
                     "num_experts_per_tok",
                     "moe_intermediate_size",
                     "shared_expert_intermediate_size",
                     "hc_count",
                     "hc_lowrank",
                     "indexer_n_heads",
                     "indexer_kv_heads",
                     "indexer_head_dim",
                     "indexer_compress_ratio",
                     "indexer_budget",
                     "ple_embed_dim",
                     "ple_conv_kernel_size",
                     "ngram_size",
                     "heads_per_ngram",
                     "split_ngram_parts",
                     "tie_word_embeddings",
                     "rms_norm_eps",
                     "layer_types",
                     "rope_parameters",
                     "ple_layer_ids"},
                    {}, "text config");
    if (resolve_architecture(architecture(value),
                             artifact::require_id(value.at("model_type"), "model_type")) !=
        Architecture::Qwen4Exp) {
        throw ArtifactError("text config is not the Qwen4Exp architecture");
    }
    if (value.at("tie_word_embeddings") != false) {
        throw ArtifactError("Qwen4Exp keeps an independent output head");
    }
    TextConfig out;
    out.hidden_size             = dimension(value, "hidden_size");
    out.vocab_size              = dimension(value, "vocab_size");
    out.num_hidden_layers       = dimension(value, "num_hidden_layers");
    out.max_position_embeddings = dimension(value, "max_position_embeddings");
    out.rms_norm_eps            = positive_float(value, "rms_norm_eps");
    const auto& layers          = value.at("layer_types");
    if (!layers.is_array() || layers.size() != out.num_hidden_layers) {
        throw ArtifactError("layer_types must cover every text block");
    }
    for (const auto& layer : layers) {
        if (layer == "full_attention") {
            out.layer_types.push_back(MixerKind::FullAttention);
            out.compact_layer_indices.push_back(out.full_attention_layers++);
        } else if (layer == "linear_attention") {
            out.layer_types.push_back(MixerKind::LinearAttention);
            out.compact_layer_indices.push_back(out.linear_attention_layers++);
        } else {
            throw ArtifactError("unknown text layer_type");
        }
    }
    out.attention = {dimension(value, "num_attention_heads"),
                     dimension(value, "num_key_value_heads"), dimension(value, "head_dim")};
    if (out.attention.num_attention_heads % out.attention.num_key_value_heads) {
        throw ArtifactError("attention query heads must be divisible by KV heads");
    }
    out.rope_parameters = rope(value.at("rope_parameters"), out.attention.head_dim);
    out.gdn = {dimension(value, "linear_num_key_heads"), dimension(value, "linear_key_head_dim"),
               dimension(value, "linear_num_value_heads"),
               dimension(value, "linear_value_head_dim"),
               dimension(value, "linear_conv_kernel_dim")};
    if (out.gdn.linear_num_value_heads % out.gdn.linear_num_key_heads) {
        throw ArtifactError("GDN value heads must be divisible by key heads");
    }
    (void)out.gdn.conv_channels();
    out.moe = {dimension(value, "num_experts"), dimension(value, "num_experts_per_tok"),
               dimension(value, "moe_intermediate_size"),
               dimension(value, "shared_expert_intermediate_size")};
    if (out.moe.num_experts_per_tok > out.moe.num_experts) {
        throw ArtifactError("selected experts exceed expert count");
    }
    out.hyper_connection = {dimension(value, "hc_count"), dimension(value, "hc_lowrank")};
    out.indexer = {dimension(value, "indexer_n_heads"), dimension(value, "indexer_kv_heads"),
                   dimension(value, "indexer_head_dim"), dimension(value, "indexer_compress_ratio"),
                   dimension(value, "indexer_budget")};
    if (out.indexer.num_heads % out.indexer.num_key_value_heads) {
        throw ArtifactError("indexer heads must be divisible by indexer KV heads");
    }
    const auto& ple_layers = value.at("ple_layer_ids");
    if (!ple_layers.is_array() || ple_layers.size() != 1) {
        throw ArtifactError("ple_layer_ids must name exactly one decoder layer");
    }
    // The source counts decoder layers from 1.
    const auto ple_layer = integer(ple_layers[0], "ple_layer_ids");
    if (ple_layer > out.num_hidden_layers) {
        throw ArtifactError("ple_layer_ids exceeds the decoder depth");
    }
    out.ple = {ple_layer - 1,
               dimension(value, "ple_embed_dim"),
               dimension(value, "ple_conv_kernel_size"),
               dimension(value, "ngram_size"),
               dimension(value, "heads_per_ngram"),
               dimension(value, "split_ngram_parts")};
    if (out.ple.ngram_size < 2 || out.ple.embed_dim % out.ple.heads()) {
        throw ArtifactError("PLE heads must be n-grams of two or more tokens dividing its width");
    }
    // The normalized text config does not carry eos_token_id; the PLE segment boundary is the
    // pinned source config's text_config.eos_token_id (tools/convert/qwen4_exp/source_config.json).
    out.ple.boundary_token = kPleBoundaryToken;
    if (out.ple.boundary_token < 0 ||
        static_cast<std::uint32_t>(out.ple.boundary_token) >= out.vocab_size) {
        throw ArtifactError("PLE boundary token is outside the vocabulary");
    }
    return out;
}

VisionConfig vision(const Json& value, const TextConfig& target) {
    require_members(value,
                    {"model_type", "depth", "hidden_size", "intermediate_size", "num_heads",
                     "patch_size", "temporal_patch_size", "spatial_merge_size",
                     "num_position_embeddings", "out_hidden_size"},
                    {}, "vision config");
    if (value.at("model_type") != "qwen4_exp_vision") {
        throw ArtifactError("unknown Vision model_type");
    }
    if (dimension(value, "out_hidden_size") != target.hidden_size) {
        throw ArtifactError("Vision merger width differs from the text hidden size");
    }
    VisionConfig out;
    out.depth                   = dimension(value, "depth");
    out.hidden_size             = dimension(value, "hidden_size");
    out.intermediate_size       = dimension(value, "intermediate_size");
    out.num_heads               = dimension(value, "num_heads");
    out.patch_size              = dimension(value, "patch_size");
    out.temporal_patch_size     = dimension(value, "temporal_patch_size");
    out.spatial_merge_size      = dimension(value, "spatial_merge_size");
    out.num_position_embeddings = dimension(value, "num_position_embeddings");
    out.position_grid_side =
        static_cast<std::uint32_t>(std::sqrt(double(out.num_position_embeddings)));
    if (out.hidden_size % out.num_heads || (out.hidden_size / out.num_heads) % 4 ||
        std::uint64_t(out.position_grid_side) * out.position_grid_side !=
            out.num_position_embeddings) {
        throw ArtifactError("invalid Vision head or square position geometry");
    }
    (void)out.patch_width();
    (void)out.merger_width();
    return out;
}

} // namespace

Config parse_config(const artifact::Directory& directory, const LoadOptions& options) {
    try {
        if (options.purpose != EnginePurpose::Generation &&
            options.purpose != EnginePurpose::CausalScoring) {
            throw ArtifactError("unknown loading purpose");
        }
        if (options.purpose == EnginePurpose::CausalScoring &&
            (options.vision || options.speculative != SpeculativeBackend::None)) {
            throw ArtifactError("CausalScoring loads the Text backbone only");
        }
        if (options.speculative != SpeculativeBackend::None &&
            options.speculative != SpeculativeBackend::Mtp) {
            throw ArtifactError("Qwen4Exp supports ordinary decoding and MTP only");
        }
        if (options.proposal_enabled()) {
            throw ArtifactError("Qwen4Exp artifacts carry no optimized proposal head");
        }
        Config out;
        out.mtp  = options.mtp();
        out.text = text(directory.component("text").config);
        if (options.vision) {
            out.vision = vision(companion(directory, "vision").config, out.text);
        }
        if (out.mtp) {
            const auto& config = companion(directory, "mtp").config;
            require_members(config, {"architectures"}, {}, "MTP config");
            if (architecture(config) != "Qwen4ExpMTP") {
                throw ArtifactError("MTP architecture differs from target mathematics");
            }
        }
        return out;
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("Qwen4Exp config: ") + error.what());
    }
}

} // namespace ninfer::models::qwen4_exp
