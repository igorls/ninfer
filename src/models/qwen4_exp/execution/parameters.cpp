#include "models/qwen4_exp/execution/parameters.h"

#include "core/weight_view.h"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen4_exp::execution {
namespace {

template <class Function>
auto with_context(const std::string& context, Function&& function) {
    try {
        return function();
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(context + ": " + error.what());
    }
}

class Prepare {
public:
    explicit Prepare(const Model& model) : model_(model) {}

    LinearParameters linear(WeightId id) const {
        return with_context(model_.weight(id).name,
                            [&] { return ops::prepare_linear_weight(model_.input(id)); });
    }

    LinearParameters linear(WeightUseId id) const {
        return with_context(model_.weight(id.parameter).name,
                            [&] { return ops::prepare_linear_weight(model_.input(id)); });
    }

    // A BF16 matrix consumed whole by a non-linear Op (hyper mixing, MoE router and shared
    // expert, GDN controls). Its single Use carries the A16 policy those Ops require.
    Weight matrix(WeightId id) const {
        return with_context(model_.weight(id).name, [&] {
            const auto prepared = ops::prepare_linear_weight(model_.input(id));
            if (prepared.policy != ops::LinearPolicy::A16Only) {
                throw std::invalid_argument("expected an A16-only BF16 parameter");
            }
            return prepared.weight;
        });
    }

    Tensor tensor(WeightId id) const {
        const auto& bound = model_.weight(id);
        return with_context(bound.name, [&] {
            const auto& view = bound.view;
            if (view.shape.size() > 4) {
                throw std::invalid_argument("direct parameter exceeds Tensor rank");
            }
            std::array<std::int32_t, 4> axes{1, 1, 1, 1};
            for (std::size_t i = 0; i < view.shape.size(); ++i) {
                const auto extent = view.shape[view.shape.size() - 1 - i];
                if (extent > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
                    throw std::invalid_argument("direct parameter exceeds Tensor extent");
                }
                axes[i] = static_cast<std::int32_t>(extent);
            }
            return weight_tensor(view, {axes[0], axes[1], axes[2], axes[3]});
        });
    }

    ops::HyperConnectionWeights mixing(WeightId norm, WeightId down, WeightId up) const {
        return {tensor(norm), matrix(down), matrix(up)};
    }

    HyperParameters hyper(const HyperConnectionWeights& w) const {
        return {mixing(w.norm, w.input_mix_down, w.input_mix_up), matrix(w.block_inject)};
    }

    ops::SparseMoeNvfp4BankWeights moe(const MoeWeights& w) const {
        return with_context(model_.weight(w.router).name, [&] {
            return ops::SparseMoeNvfp4BankWeights{
                .router             = matrix(w.router),
                .shared_expert_gate = matrix(w.shared_expert_gate),
                .gate_up = ops::prepare_nvfp4_expert_bank_weight(model_.input(w.experts_gate_up)),
                .down    = ops::prepare_nvfp4_expert_bank_weight(model_.input(w.experts_down)),
                .shared_gate = matrix(w.shared_gate),
                .shared_up   = matrix(w.shared_up),
                .shared_down = matrix(w.shared_down),
            };
        });
    }

    BlockParameters block(const BlockWeights& w) const {
        BlockParameters out;
        out.attention_hyper = hyper(w.attention_hyper);
        out.mlp_hyper       = hyper(w.mlp_hyper);
        if (const auto* a = std::get_if<AttentionWeights>(&w.mixer)) {
            out.mixer = AttentionParameters{
                .projection         = linear(a->query_gate_key_value),
                .query_norm         = tensor(a->query_norm),
                .key_norm           = tensor(a->key_norm),
                .indexer_projection = linear(a->indexer_query_key),
                .indexer_query_norm = tensor(a->indexer_query_norm),
                .indexer_key_norm   = tensor(a->indexer_key_norm),
                .output             = linear(a->output),
            };
        } else {
            const auto& g = std::get<GdnWeights>(w.mixer);
            out.mixer     = GdnParameters{
                    .projection  = linear(g.query_key_value_z),
                    .control     = matrix(g.a_b_projection),
                    .a_log       = tensor(g.a_log),
                    .dt_bias     = tensor(g.dt_bias),
                    .convolution = tensor(g.convolution),
                    .norm        = tensor(g.norm),
                    .output      = linear(g.output),
            };
        }
        out.moe = moe(w.moe);
        return out;
    }

    NormParameters norm(const NormWeights& w) const { return {tensor(w.weight), tensor(w.bias)}; }

    std::optional<Tensor> joined_bias(const std::array<WeightId, 3>& ids) const {
        WeightView view;
        std::uint64_t count = 0;
        for (const auto id : ids) {
            const auto& input = model_.weight(id).view;
            count += weight_element_count(input.shape);
            for (const auto& part : input.parts) {
                if (!view.parts.empty() && (view.parts.back().parent != part.parent ||
                                            view.parts.back().end != part.begin)) {
                    return std::nullopt;
                }
                view.parts.push_back(part);
            }
        }
        if (count > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("Vision bias exceeds Tensor extent");
        }
        view.shape = {count};
        return weight_tensor(view, {static_cast<std::int32_t>(count)});
    }

    VisionParameters vision(const VisionWeights& w) const {
        VisionParameters out;
        out.patch_embedding      = linear(w.patch_embedding);
        out.patch_embedding_bias = tensor(w.patch_embedding_bias);
        out.position_embedding   = tensor(w.position_embedding);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context("vision/layers/" + std::to_string(i), [&] {
                const auto& layer = w.layers[i];
                const std::array qkv{model_.input(layer.query), model_.input(layer.key),
                                     model_.input(layer.value)};
                const std::array ids{layer.query_bias, layer.key_bias, layer.value_bias};
                const auto bias = joined_bias(ids);
                if (!bias) {
                    throw std::invalid_argument(
                        "Vision QKV bias: this fixed call requires a contiguous bias bank");
                }
                return VisionBlockParameters{norm(layer.norm1),
                                             norm(layer.norm2),
                                             ops::prepare_linear_weight(qkv),
                                             *bias,
                                             linear(layer.output),
                                             linear(layer.fc1),
                                             linear(layer.fc2),
                                             tensor(layer.output_bias),
                                             tensor(layer.fc1_bias),
                                             tensor(layer.fc2_bias)};
            }));
        }
        out.merger_norm     = norm(w.merger_norm);
        out.merger_fc1      = linear(w.merger_fc1);
        out.merger_fc2      = linear(w.merger_fc2);
        out.merger_fc1_bias = tensor(w.merger_fc1_bias);
        out.merger_fc2_bias = tensor(w.merger_fc2_bias);
        return out;
    }

private:
    const Model& model_;
};

} // namespace

Parameters::Parameters(const Model& source) : model(source) {
    const Prepare prepare(model);
    const auto& w        = model.weights().text;
    text.token_embedding = native_weight(model.weight(w.token_embedding).view);
    text.output_head     = prepare.linear(w.output_head_use);
    text.layers.reserve(w.layers.size());
    for (std::size_t i = 0; i < w.layers.size(); ++i) {
        text.layers.push_back(with_context("text/layers/" + std::to_string(i),
                                           [&] { return prepare.block(w.layers[i]); }));
    }
    text.ple = with_context("text/ple", [&] {
        return PleParameters{
            .layer            = model.config().text.ple.layer,
            .key_projection   = prepare.linear(w.ple.key_projection),
            .value_projection = prepare.linear(w.ple.value_projection),
            .query_norm       = prepare.tensor(w.ple.query_norm),
            .key_norm         = prepare.tensor(w.ple.key_norm),
            .conv_norm        = prepare.tensor(w.ple.conv_norm),
            .convolution      = prepare.tensor(w.ple.convolution),
        };
    });
    text.final_mixer = with_context("text/hyper_connection", [&] {
        return prepare.mixing(w.final_mixer.norm, w.final_mixer.input_mix_down,
                              w.final_mixer.input_mix_up);
    });
    if (model.weights().vision) {
        vision = with_context("vision", [&] { return prepare.vision(*model.weights().vision); });
    }
    if (model.weights().mtp) {
        const auto& m = *model.weights().mtp;
        mtp = with_context("mtp", [&] {
            return MtpParameters{
                .embedding_projection = prepare.linear(m.embedding_projection),
                .hidden_projection = prepare.linear(m.hidden_projection),
                .embedding_norm = prepare.tensor(m.embedding_norm),
                .hidden_norm = prepare.tensor(m.hidden_norm),
                .layer = prepare.block(m.layer),
                .final_mixer = prepare.mixing(m.final_mixer.norm, m.final_mixer.input_mix_down,
                                              m.final_mixer.input_mix_up),
                .token_embedding = native_weight(model.weight(m.token_embedding).view),
                .output_head = prepare.linear(m.output_head_use),
            };
        });
    }
}

} // namespace ninfer::models::qwen4_exp::execution
