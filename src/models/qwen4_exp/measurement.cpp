#include "models/qwen4_exp/measurement.h"

#include "models/qwen3_5/frontend/digest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <map>
#include <vector>

namespace ninfer::models::qwen4_exp {

std::string prefill_signature(const Model& model) {
    using Json         = nlohmann::json;
    const auto& config = model.config();
    const auto& t      = config.text;
    Json facts{{"implementation", "qwen4_exp-prefill-1"},
               {"architecture", architecture_name(Architecture::Qwen4Exp)},
               {"hidden_size", t.hidden_size},
               {"vocab_size", t.vocab_size},
               {"public_token_count", model.resources().public_token_count},
               {"num_hidden_layers", t.num_hidden_layers},
               {"rms_norm_eps", t.rms_norm_eps},
               {"layer_types", t.layer_types},
               {"attention",
                {t.attention.num_attention_heads, t.attention.num_key_value_heads,
                 t.attention.head_dim}},
               {"rope",
                {t.rope_parameters.rope_theta, t.rope_parameters.rotary_dim,
                 t.rope_parameters.pair_axes}},
               {"gdn",
                {t.gdn.linear_num_key_heads, t.gdn.linear_key_head_dim,
                 t.gdn.linear_num_value_heads, t.gdn.linear_value_head_dim,
                 t.gdn.linear_conv_kernel_dim}},
               {"moe",
                {t.moe.num_experts, t.moe.num_experts_per_tok, t.moe.moe_intermediate_size,
                 t.moe.shared_expert_intermediate_size}},
               {"hyper_connection", {t.hyper_connection.streams, t.hyper_connection.low_rank}},
               {"indexer",
                {t.indexer.num_heads, t.indexer.num_key_value_heads, t.indexer.head_dim,
                 t.indexer.compress_ratio, t.indexer.budget}},
               {"ple",
                {t.ple.layer, t.ple.embed_dim, t.ple.conv_kernel_size, t.ple.ngram_size,
                 t.ple.heads_per_ngram, t.ple.boundary_token}}};
    if (config.vision) {
        const auto& v   = *config.vision;
        facts["vision"] = {
            v.depth,      v.hidden_size,         v.intermediate_size,  v.num_heads,
            v.patch_size, v.temporal_patch_size, v.spatial_merge_size, v.num_position_embeddings};
    }
    std::vector<const BoundWeight*> weights;
    // PrefillWork prices primary Text/Vision reconstruction; the speculative head does not
    // change the baseline's applicability.
    for (const auto& weight : model.weight_data()) {
        if (weight.name.starts_with("text/") || weight.name.starts_with("vision/")) {
            weights.push_back(&weight);
        }
    }
    std::sort(weights.begin(), weights.end(),
              [](const auto* a, const auto* b) { return a->name < b->name; });
    std::map<const WeightParent*, std::size_t> parents;
    auto& inventory = facts["weights"] = Json::array();
    for (const auto* weight : weights) {
        Json item{{"role", weight->name},
                  {"shape", weight->view.shape},
                  {"parts", Json::array()},
                  {"uses", Json::array()}};
        for (const auto& part : weight->view.parts) {
            const auto [it, inserted] = parents.emplace(part.parent, parents.size());
            const auto& geometry      = part.parent->geometry;
            item["parts"].push_back({it->second, part.begin, part.end, geometry.format,
                                     geometry.layout, geometry.shape, geometry.padded_columns,
                                     geometry.group_size});
        }
        for (const auto& use : weight->uses) {
            if (!use.input.starts_with("text/") && !use.input.starts_with("vision/")) { continue; }
            item["uses"].push_back(
                {use.input, use.policy, use.activation_input_divisor.has_value()});
        }
        inventory.push_back(std::move(item));
    }
    return qwen3_5::frontend::sha256_hex(qwen3_5::frontend::sha256(facts.dump()));
}

} // namespace ninfer::models::qwen4_exp
