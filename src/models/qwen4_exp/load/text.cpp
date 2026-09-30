#include "models/qwen4_exp/load/bindings.h"

#include <string>

namespace ninfer::models::qwen4_exp::loading {
namespace {

HyperConnectionWeights bind_hyper(Bindings& b, const TextConfig& config, const std::string& p) {
    const auto width = config.stream_width();
    const auto low   = config.hyper_connection.low_rank;
    const auto state = p + "normalized_state";
    HyperConnectionWeights out;
    out.norm           = b.direct(p + "norm", {width});
    out.input_mix_down = b.parameter(p + "input_mix/down", {low, width}, {state});
    out.input_mix_up   = b.parameter(p + "input_mix/up", {width, low}, {p + "low_rank"});
    out.block_inject =
        b.parameter(p + "block_inject", {config.hyper_connection.streams, width}, {state});
    return out;
}

AttentionWeights bind_attention(Bindings& b, const TextConfig& config, const std::string& p) {
    const auto& a       = config.attention;
    const auto& indexer = config.indexer;
    const auto h        = config.hidden_size;
    const auto q        = a.query_width();
    const auto input    = p + "mixer_input";
    AttentionWeights out;
    out.query_gate_key_value =
        b.parameter(p + "attention/query_gate_key_value", {2 * q + 2 * a.key_width(), h}, {input});
    out.query_norm = b.direct(p + "attention/query_norm", {a.head_dim});
    out.key_norm   = b.direct(p + "attention/key_norm", {a.head_dim});
    out.output     = b.parameter(p + "attention/output", {h, q}, {p + "attention/gated_output"});
    out.indexer_query_key = b.parameter(
        p + "attention/indexer/query_key",
        {std::uint64_t(indexer.num_heads + indexer.num_key_value_heads) * indexer.head_dim, h},
        {input});
    out.indexer_query_norm = b.direct(p + "attention/indexer/query_norm", {indexer.head_dim});
    out.indexer_key_norm   = b.direct(p + "attention/indexer/key_norm", {indexer.head_dim});
    return out;
}

GdnWeights bind_gdn(Bindings& b, const TextConfig& config, const std::string& p) {
    const auto& g    = config.gdn;
    const auto h     = config.hidden_size;
    const auto heads = g.linear_num_value_heads;
    const auto input = p + "mixer_input";
    GdnWeights out;
    out.query_key_value_z = b.parameter(p + "gdn/query_key_value_z",
                                        {2 * g.key_width() + 2 * g.value_width(), h}, {input});
    out.a_b_projection    = b.parameter(p + "gdn/a_b_projection", {2ULL * heads, h}, {input});
    out.a_log             = b.direct(p + "gdn/a_log", {heads});
    out.dt_bias           = b.direct(p + "gdn/dt_bias", {heads});
    out.convolution =
        b.direct(p + "gdn/convolution", {g.linear_conv_kernel_dim, g.conv_channels()});
    out.norm   = b.direct(p + "gdn/norm", {g.linear_value_head_dim});
    out.output = b.parameter(p + "gdn/output", {h, g.value_width()}, {p + "gdn/gated_output"});
    return out;
}

MoeWeights bind_moe(Bindings& b, const TextConfig& config, const std::string& prefix) {
    const auto& moe   = config.moe;
    const auto p      = prefix + "mlp/";
    const auto input  = prefix + "ffn_input";
    const auto h      = config.hidden_size;
    const auto ir     = moe.moe_intermediate_size;
    const auto shared = moe.shared_expert_intermediate_size;
    MoeWeights out;
    out.router             = b.parameter(p + "router", {moe.num_experts, h}, {input});
    out.shared_expert_gate = b.parameter(p + "shared_expert_gate", {1, h}, {input});
    out.shared_gate        = b.parameter(p + "shared_expert/gate", {shared, h}, {input});
    out.shared_up          = b.parameter(p + "shared_expert/up", {shared, h}, {input});
    out.shared_down =
        b.parameter(p + "shared_expert/down", {h, shared}, {p + "shared_expert/product"});
    out.experts_gate_up =
        b.parameter(p + "experts/gate_up", {moe.num_experts, 2ULL * ir, h}, {input});
    out.experts_down =
        b.parameter(p + "experts/down", {moe.num_experts, h, ir}, {p + "experts/product"});
    return out;
}

} // namespace

BlockWeights bind_block(Bindings& b, const TextConfig& config, const std::string& p,
                        MixerKind mixer) {
    BlockWeights out;
    out.attention_hyper = bind_hyper(b, config, p + "attention/hyper_connection/");
    if (mixer == MixerKind::FullAttention) {
        out.mixer = bind_attention(b, config, p);
    } else {
        out.mixer = bind_gdn(b, config, p);
    }
    out.mlp_hyper = bind_hyper(b, config, p + "mlp/hyper_connection/");
    out.moe       = bind_moe(b, config, p);
    return out;
}

HyperMixerWeights bind_mixer(Bindings& b, const TextConfig& config, const std::string& p) {
    const auto width = config.stream_width();
    const auto low   = config.hyper_connection.low_rank;
    return {b.direct(p + "norm", {width}),
            b.parameter(p + "input_mix/down", {low, width}, {p + "normalized_state"}),
            b.parameter(p + "input_mix/up", {width, low}, {p + "low_rank"})};
}

TextWeights bind_text(Bindings& b, const TextConfig& config, bool mtp) {
    const auto h = config.hidden_size;
    TextWeights out;
    out.token_embedding = b.parameter("text/token_embedding", {config.vocab_size, h});
    std::vector<std::string> head_inputs{"text/final_hidden"};
    if (mtp) { head_inputs.emplace_back("mtp/final_hidden"); }
    out.output_head =
        b.parameter("text/output_head", {config.vocab_size, h}, std::move(head_inputs));
    out.output_head_use = b.use(out.output_head, "text/final_hidden");
    out.layers.reserve(config.num_hidden_layers);
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        out.layers.push_back(
            bind_block(b, config, "text/layers/" + std::to_string(i) + "/", config.layer_types[i]));
    }
    const auto p     = "text/layers/" + std::to_string(config.ple.layer) + "/ple/";
    const auto width = config.stream_width();
    out.ple.key_projection =
        b.parameter(p + "key_projection", {width, config.ple.embed_dim}, {p + "embedding"});
    out.ple.value_projection =
        b.parameter(p + "value_projection", {h, config.ple.embed_dim}, {p + "embedding"});
    out.ple.query_norm  = b.direct(p + "query_norm", {width});
    out.ple.key_norm    = b.direct(p + "key_norm", {width});
    out.ple.conv_norm   = b.direct(p + "conv_norm", {width});
    out.ple.convolution = b.direct(p + "convolution", {config.ple.conv_kernel_size, width});
    out.final_mixer     = bind_mixer(b, config, "text/hyper_connection/");
    return out;
}

MtpWeights bind_mtp(Bindings& b, const TextConfig& config, const TextWeights& target) {
    const auto h = config.hidden_size;
    MtpWeights out;
    out.embedding_projection =
        b.parameter("mtp/embedding_projection", {h, h}, {"mtp/normalized_embedding"});
    out.hidden_projection = b.parameter("mtp/hidden_projection", {h, h}, {"mtp/normalized_hidden"});
    out.embedding_norm    = b.direct("mtp/embedding_norm", {h});
    out.hidden_norm       = b.direct("mtp/hidden_norm", {config.stream_width()});
    out.layer             = bind_block(b, config, "mtp/layers/0/", MixerKind::FullAttention);
    out.final_mixer       = bind_mixer(b, config, "mtp/hyper_connection/");
    out.token_embedding   = target.token_embedding;
    out.output_head       = target.output_head;
    out.output_head_use   = b.use(target.output_head, "mtp/final_hidden");
    return out;
}

} // namespace ninfer::models::qwen4_exp::loading
