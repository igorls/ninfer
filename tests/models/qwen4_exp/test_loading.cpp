// Qwen4Exp configuration reads and binding plans: PLE layer lists, the PLE boundary token, the
// derived PLE row and history geometry, companion identities and the PLE Op domain check.

#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "artifact/reader.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/load.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace q4 = ninfer::models::qwen4_exp;
using namespace ninfer::test::artifact_fixture;

// The text config tools/convert/qwen4_exp writes for the pinned Qwen/Qwen3.8-Flash-Next
// config.json, without the eos_token_id that derivations before it omitted.
Json flash_next_text() {
    Json layers = Json::array();
    for (int i = 0; i < 48; ++i) {
        layers.push_back(i % 4 == 3 ? "full_attention" : "linear_attention");
    }
    return {{"architectures", {"Qwen4ExpForCausalLM"}},
            {"model_type", "qwen4_exp_text"},
            {"hidden_size", 2560},
            {"vocab_size", 248320},
            {"num_hidden_layers", 48},
            {"max_position_embeddings", 262144},
            {"num_attention_heads", 24},
            {"num_key_value_heads", 2},
            {"head_dim", 256},
            {"linear_num_key_heads", 16},
            {"linear_key_head_dim", 128},
            {"linear_num_value_heads", 48},
            {"linear_value_head_dim", 128},
            {"linear_conv_kernel_dim", 4},
            {"num_experts", 512},
            {"num_experts_per_tok", 10},
            {"moe_intermediate_size", 640},
            {"shared_expert_intermediate_size", 640},
            {"hc_count", 4},
            {"hc_lowrank", 320},
            {"indexer_n_heads", 4},
            {"indexer_kv_heads", 1},
            {"indexer_head_dim", 128},
            {"indexer_compress_ratio", 4},
            {"indexer_budget", 2048},
            {"ple_embed_dim", 2560},
            {"ple_conv_kernel_size", 4},
            {"ngram_size", 3},
            {"heads_per_ngram", 8},
            {"split_ngram_parts", 128},
            {"tie_word_embeddings", false},
            {"rms_norm_eps", 9.999999974752427e-07},
            {"layer_types", layers},
            {"rope_parameters",
             {{"rope_theta", 10000000.0},
              {"partial_rotary_factor", 0.25},
              {"mrope_section", {11, 11, 10}}}},
            {"ple_layer_ids", {2}}};
}

q4::Config parse(const Json& text, LoadOptions options = {},
                 const std::optional<Json>& mtp    = std::nullopt,
                 const std::optional<Json>& vision = std::nullopt) {
    artifact::Directory directory;
    directory.components["text"].config = text;
    if (mtp) {
        auto& component  = directory.components["mtp"];
        component.config = *mtp;
        component.target = "text";
    }
    if (vision) {
        auto& component  = directory.components["vision"];
        component.config = *vision;
        component.target = "text";
    }
    return q4::parse_config(directory, options);
}

void published_flash_next_config() {
    auto text            = flash_next_text();
    text["eos_token_id"] = 248044;
    const auto config    = parse(text).text;
    const auto& ple      = config.ple;
    require(ple.layers == std::vector<std::uint32_t>{1}, "ple_layer_ids [2] is not layer 1");
    require(ple.boundary_token == 248044, "eos_token_id is not the PLE boundary");
    require(ple.heads() == 16 && ple.head_width() == 160 && ple.code_row_bytes() == 80 &&
                ple.scale_row_bytes() == 20 && ple.history_columns() == 9,
            "Flash-Next PLE row or history geometry changed");
    q4::execution::require_supported_ple(config);

    auto explicit_eos            = flash_next_text();
    explicit_eos["eos_token_id"] = 248046;
    require(parse(explicit_eos).text.ple.boundary_token == 248046,
            "eos_token_id did not select the PLE boundary");

    rejects([&] { (void)parse(flash_next_text()); }, "a config without eos_token_id was accepted");
    explicit_eos["eos_token_id"] = 248320;
    rejects([&] { (void)parse(explicit_eos); }, "a boundary outside the vocabulary was accepted");
    explicit_eos["eos_token_id"] = -1;
    rejects([&] { (void)parse(explicit_eos); }, "a negative boundary token was accepted");
}

void ple_layer_lists() {
    auto text             = flash_next_text();
    text["eos_token_id"]  = 248044;
    text["ple_layer_ids"] = {2, 6, 47};
    const auto config     = parse(text).text;
    require(config.ple.layers == std::vector<std::uint32_t>{1, 5, 46} &&
                config.ple.index_of(5) == 1U && !config.ple.index_of(3),
            "a PLE layer list was not read as ascending 0-based GDN layers");

    const auto bad = [&](const Json& ids, const char* message) {
        auto changed             = text;
        changed["ple_layer_ids"] = ids;
        rejects([&] { (void)parse(changed); }, message);
    };
    bad(Json::array(), "an empty PLE layer list was accepted");
    bad({0}, "a zero one-based PLE layer was accepted");
    bad({49}, "a PLE layer beyond the decoder depth was accepted");
    bad({4}, "PLE before a full-attention block was accepted");
    bad({6, 2}, "an unsorted PLE layer list was accepted");
    bad({2, 2}, "a repeated PLE layer was accepted");
    bad(2, "a scalar ple_layer_ids was accepted");
}

void unsupported_geometry() {
    // Each configuration is valid mathematics; ops::ple_ngram executes only Flash-Next's PLE.
    const auto rejected = [](const Json& patch, const char* message) {
        auto text            = flash_next_text();
        text["eos_token_id"] = 248044;
        text.update(patch);
        const auto config = parse(text).text;
        rejects<std::invalid_argument>([&] { q4::execution::require_supported_ple(config); },
                                       message);
    };
    rejected({{"hidden_size", 5120}}, "a 5120-wide stream reached the 2560-wide PLE Op");
    rejected({{"ple_conv_kernel_size", 3}}, "a three-tap PLE convolution was accepted");
    rejected({{"ngram_size", 5}, {"heads_per_ngram", 4}},
             "a dilation-5 PLE convolution was accepted");
    rejected({{"heads_per_ngram", 4}}, "a 320-wide n-gram head was accepted");
    rejected({{"rms_norm_eps", 1e-5}}, "a PLE norm epsilon other than 1e-6 was accepted");
}

void companions_and_dense_configs() {
    auto text            = flash_next_text();
    text["eos_token_id"] = 248044;
    const LoadOptions mtp{.speculative = SpeculativeBackend::Mtp};
    require(parse(text, mtp, Json{{"architectures", {"Qwen4ExpMTP"}}}).mtp,
            "the Qwen4ExpMTP companion was not selected");
    rejects([&] { (void)parse(text, mtp, Json{{"architectures", {"Qwen3_5MTP"}}}); },
            "an MTP companion of other mathematics was accepted");
    const Json vision{{"model_type", "qwen4_exp_vision"},
                      {"depth", 1},
                      {"hidden_size", 64},
                      {"intermediate_size", 128},
                      {"num_heads", 4},
                      {"patch_size", 16},
                      {"temporal_patch_size", 2},
                      {"spatial_merge_size", 2},
                      {"num_position_embeddings", 16},
                      {"out_hidden_size", 2560}};
    require(parse(text, {.vision = true}, std::nullopt, vision).vision.has_value(),
            "the qwen4_exp_vision companion was not selected");
    auto other_vision          = vision;
    other_vision["model_type"] = "qwen3_5_vision";
    rejects([&] { (void)parse(text, {.vision = true}, std::nullopt, other_vision); },
            "a Vision companion of another model_type was accepted");

    // Dense FFN blocks are not executable yet: a config without the MoE keys is rejected.
    auto dense = text;
    for (const char* key : {"num_experts", "num_experts_per_tok", "moe_intermediate_size",
                            "shared_expert_intermediate_size"}) {
        dense.erase(key);
    }
    dense["intermediate_size"] = 17408;
    rejects([&] { (void)parse(dense); }, "a dense FFN config was accepted");
}

Json added_token(int id, const std::string& content) {
    return {{"id", id},        {"content", content}, {"special", true},    {"single_word", false},
            {"lstrip", false}, {"rstrip", false},    {"normalized", false}};
}

// A complete four-block artifact (three GDN blocks and one QSA block) small enough to write in a
// test, with PLE tables before decoder layers 0 and 2.
struct TinyArtifact {
    static constexpr std::uint64_t kHidden = 64, kStreams = 4, kLow = 8, kVocab = 272;
    static constexpr std::uint64_t kPleHeads = 4, kPleWidth = 16, kShardRows = 8, kShards = 2;

    Fixture file;
    std::optional<artifact::Reader> reader;

    void tensor(const std::string& id, std::vector<std::uint64_t> shape, QType format = QType::BF16,
                QuantLayout layout = QuantLayout::Contiguous) {
        const auto geometry = weight_geometry(format, layout, shape);
        const auto offset   = (file.payload.size() + 255) / 256 * 256;
        file.payload.resize(offset + geometry.bytes);
        file.root["objects"].push_back({{"id", id},
                                        {"kind", "tensor"},
                                        {"shape", shape},
                                        {"format", artifact::format_name(format)},
                                        {"layout", artifact::layout_name(layout)},
                                        {"offset", offset},
                                        {"bytes", geometry.bytes}});
        file.root["bindings"][id] = {{"object", id}};
    }

    void parameter(const std::string& name, std::vector<std::uint64_t> shape,
                   const std::vector<std::string>& inputs = {}) {
        tensor(name, std::move(shape));
        for (const auto& input : inputs) {
            file.root["uses"].push_back(
                {{"parameter", name}, {"input", input}, {"activation_policy", "A16Only"}});
        }
    }

    void int64_table(const std::string& name, const std::vector<std::int64_t>& values) {
        tensor(name, {values.size()}, QType::INT64);
        const auto offset = file.root["objects"].back()["offset"].get<std::size_t>();
        for (std::size_t i = 0; i < values.size(); ++i) {
            put_word(file.payload, offset + 8 * i, static_cast<std::uint64_t>(values[i]), 8);
        }
    }

    void resource(const std::string& role, const std::string& bytes) {
        const auto offset = file.payload.size();
        for (const auto byte : bytes) { file.payload.push_back(std::byte(byte)); }
        file.root["objects"].push_back({{"id", role},
                                        {"kind", "resource"},
                                        {"encoding", "raw_bytes_v1"},
                                        {"offset", offset},
                                        {"bytes", bytes.size()}});
        file.root["components"]["text"]["resources"][role] = role;
    }

    // A stream mixer; a hyper-connection around a block also injects the block output.
    void hyper(const std::string& p, bool inject = true) {
        const auto width = kHidden * kStreams;
        parameter(p + "norm", {width});
        parameter(p + "input_mix/down", {kLow, width}, {p + "normalized_state"});
        parameter(p + "input_mix/up", {width, kLow}, {p + "low_rank"});
        if (inject) { parameter(p + "block_inject", {kStreams, width}, {p + "normalized_state"}); }
    }

    void block(std::uint32_t layer, bool attention) {
        const auto p = "text/layers/" + std::to_string(layer) + "/";
        hyper(p + "attention/hyper_connection/");
        hyper(p + "mlp/hyper_connection/");
        if (attention) {
            parameter(p + "attention/query_gate_key_value", {96, kHidden}, {p + "mixer_input"});
            parameter(p + "attention/query_norm", {16});
            parameter(p + "attention/key_norm", {16});
            parameter(p + "attention/output", {kHidden, 32}, {p + "attention/gated_output"});
            parameter(p + "attention/indexer/query_key", {48, kHidden}, {p + "mixer_input"});
            parameter(p + "attention/indexer/query_norm", {16});
            parameter(p + "attention/indexer/key_norm", {16});
        } else {
            parameter(p + "gdn/query_key_value_z", {96, kHidden}, {p + "mixer_input"});
            parameter(p + "gdn/a_b_projection", {4, kHidden}, {p + "mixer_input"});
            parameter(p + "gdn/a_log", {2});
            parameter(p + "gdn/dt_bias", {2});
            parameter(p + "gdn/convolution", {4, 64});
            parameter(p + "gdn/norm", {16});
            parameter(p + "gdn/output", {kHidden, 32}, {p + "gdn/gated_output"});
        }
        const auto ffn = p + "ffn_input";
        parameter(p + "mlp/router", {4, kHidden}, {ffn});
        parameter(p + "mlp/shared_expert_gate", {1, kHidden}, {ffn});
        parameter(p + "mlp/shared_expert/gate", {16, kHidden}, {ffn});
        parameter(p + "mlp/shared_expert/up", {16, kHidden}, {ffn});
        parameter(p + "mlp/shared_expert/down", {kHidden, 16}, {p + "mlp/shared_expert/product"});
        parameter(p + "mlp/experts/gate_up", {4, 32, kHidden}, {ffn});
        parameter(p + "mlp/experts/down", {4, kHidden, 16}, {p + "mlp/experts/product"});
    }

    void ple(std::uint32_t layer) {
        const auto p     = "text/layers/" + std::to_string(layer) + "/ple/";
        const auto width = kHidden * kStreams;
        parameter(p + "key_projection", {width, kHidden}, {p + "embedding"});
        parameter(p + "value_projection", {kHidden, kHidden}, {p + "embedding"});
        parameter(p + "query_norm", {width});
        parameter(p + "key_norm", {width});
        parameter(p + "conv_norm", {width});
        parameter(p + "convolution", {4, width});
        int64_table(p + "embedding/layer_multipliers", {3, 5, 7});
        int64_table(p + "embedding/ngram_head_offsets", {0, 3, 7, 11});
        int64_table(p + "embedding/ngram_head_vocab_sizes", {3, 4, 4, 5});
        for (std::uint64_t shard = 0; shard < kShards; ++shard) {
            tensor(p + "embedding/shards/" + std::to_string(shard), {kShardRows, kPleWidth},
                   QType::U4Z8_G16_FP16, QuantLayout::PackedU4G16);
        }
    }

    TinyArtifact() {
        file.payload.clear();
        Json layers = {"linear_attention", "linear_attention", "linear_attention",
                       "full_attention"};
        file.root   = {{"components",
                        {{"text",
                          {{"config",
                            {{"architectures", {"Qwen4ExpForCausalLM"}},
                             {"model_type", "qwen4_exp_text"},
                             {"hidden_size", kHidden},
                             {"vocab_size", kVocab},
                             {"num_hidden_layers", 4},
                             {"max_position_embeddings", 128},
                             {"num_attention_heads", 2},
                             {"num_key_value_heads", 1},
                             {"head_dim", 16},
                             {"linear_num_key_heads", 1},
                             {"linear_key_head_dim", 16},
                             {"linear_num_value_heads", 2},
                             {"linear_value_head_dim", 16},
                             {"linear_conv_kernel_dim", 4},
                             {"num_experts", 4},
                             {"num_experts_per_tok", 2},
                             {"moe_intermediate_size", 16},
                             {"shared_expert_intermediate_size", 16},
                             {"hc_count", kStreams},
                             {"hc_lowrank", kLow},
                             {"indexer_n_heads", 2},
                             {"indexer_kv_heads", 1},
                             {"indexer_head_dim", 16},
                             {"indexer_compress_ratio", 4},
                             {"indexer_budget", 16},
                             {"ple_embed_dim", kHidden},
                             {"ple_conv_kernel_size", 4},
                             {"ngram_size", 3},
                             {"heads_per_ngram", 2},
                             {"split_ngram_parts", kShards},
                             {"tie_word_embeddings", false},
                             {"rms_norm_eps", 1e-6},
                             {"layer_types", layers},
                             {"rope_parameters",
                              {{"rope_theta", 10000},
                               {"partial_rotary_factor", 0.5},
                               {"mrope_section", {2, 1, 1}}}},
                             {"ple_layer_ids", {1, 3}},
                             {"eos_token_id", 256}}}}}}},
                       {"objects", Json::array()},
                       {"bindings", Json::object()},
                       {"uses", Json::array()},
                       {"metadata", {{"name", "a-qwen4-exp-test-instance"}}}};
        parameter("text/token_embedding", {kVocab, kHidden});
        parameter("text/output_head", {kVocab, kHidden}, {"text/final_hidden"});
        hyper("text/hyper_connection/", false);
        for (std::uint32_t layer = 0; layer < 4; ++layer) { block(layer, layer == 3); }
        ple(0);
        ple(2);

        Json vocab    = Json::object();
        unsigned next = 256;
        for (unsigned byte = 0; byte < 256; ++byte) {
            const bool visible =
                (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
            const unsigned cp = visible ? byte : next++;
            std::string token;
            if (cp < 128) {
                token.push_back(static_cast<char>(cp));
            } else {
                token.push_back(static_cast<char>(0xc0 | (cp >> 6)));
                token.push_back(static_cast<char>(0x80 | (cp & 63)));
            }
            vocab[token] = byte;
        }
        Json added                = Json::array();
        Json decoder              = Json::object();
        const std::array specials = {"<|endoftext|>",  "<|im_start|>",  "<|im_end|>",
                                     "<think>",        "</think>",      "<|vision_start|>",
                                     "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"};
        for (std::size_t i = 0; i < specials.size(); ++i) {
            const auto token = added_token(static_cast<int>(256 + i), specials[i]);
            if (i < 8) { added.push_back(token); }
            decoder[std::to_string(256 + i)] = token;
        }
        resource("tokenizer.json",
                 Json{{"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", Json::array()}}},
                      {"added_tokens", added}}
                     .dump());
        resource("tokenizer_config.json", Json{{"added_tokens_decoder", decoder}}.dump());
        resource("generation_config.json", Json{{"eos_token_id", {258, 256}}}.dump());
        resource("chat_template.jinja", "a template carried as data");
    }

    q4::LoadPlan plan() {
        file.root["files"] =
            Json::array({{{"path", nullptr}, {"payload_bytes", file.payload.size()}}});
        file.write();
        reader.emplace(file.entry);
        return q4::plan_load(*reader, {});
    }

    void erase_prefix(const std::string& prefix) {
        for (auto it = file.root["bindings"].begin(); it != file.root["bindings"].end();) {
            it = it.key().starts_with(prefix) ? file.root["bindings"].erase(it) : std::next(it);
        }
        auto& uses = file.root["uses"];
        for (auto it = uses.begin(); it != uses.end();) {
            it = (*it)["parameter"].get<std::string>().starts_with(prefix) ? uses.erase(it)
                                                                           : std::next(it);
        }
    }
};

void ple_layer_bindings() {
    // Each listed PLE layer binds its own projections, norms, convolution and table.
    {
        TinyArtifact artifact;
        const auto plan = artifact.plan();
        require(plan.config().text.ple.layers == std::vector<std::uint32_t>{0, 2} &&
                    plan.config().text.ple.boundary_token == 256,
                "the tiny artifact's PLE configuration was not read");
        require(plan.weights().text.ple.size() == 2, "a PLE layer's weights were not bound");
        std::size_t device_parameters = 0;
        for (const auto& item : artifact.file.root["bindings"].items()) {
            if (item.key().find("/embedding/") == std::string::npos) { ++device_parameters; }
        }
        require(plan.binding_count() == device_parameters + 2 * (TinyArtifact::kShards + 3),
                "PLE shards and index tables were not bound per layer");
    }
    // A listed PLE layer without stored weights fails to bind.
    {
        TinyArtifact artifact;
        artifact.erase_prefix("text/layers/2/ple/");
        rejects([&] { (void)artifact.plan(); }, "a listed PLE layer without weights was accepted");
    }
    // Stored PLE weights at a layer the config does not list are an unconsumed contract.
    {
        TinyArtifact artifact;
        artifact.file.root["components"]["text"]["config"]["ple_layer_ids"] = {1};
        rejects([&] { (void)artifact.plan(); }, "unlisted stored PLE weights were ignored");
    }
}

} // namespace

int main() {
    try {
        published_flash_next_config();
        ple_layer_lists();
        unsupported_geometry();
        companions_and_dense_configs();
        ple_layer_bindings();
        std::cout << "Qwen4Exp config reads, PLE layer bindings and PLE Op domain checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
