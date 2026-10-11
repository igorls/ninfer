#include "models/qwen4_exp/load.h"

#include "artifact/reader.h"
#include "artifact/views.h"
#include "models/qwen4_exp/load/bindings.h"

#include <utility>

namespace ninfer::models::qwen4_exp {

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    ModelWeights weights;
    std::vector<loading::PendingWeight> pending;
    std::vector<loading::PendingPleTable> ple;
    artifact::MaterializationPlan materialization;
    FrontendResources resources;
    InstanceInfo info;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const FrontendResources& LoadPlan::resources() const { return impl_->resources; }

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->materialization;
}

std::size_t LoadPlan::binding_count() const {
    // Device parameters, and per PLE layer its mapped shards and three owning index tables.
    std::size_t count = impl_->pending.size();
    for (const auto& table : impl_->ple) { count += table.shards.size() + 3; }
    return count;
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    artifact::Binder binder(reader);
    out->resources = loading::bind_resources(binder, out->config);
    loading::Bindings bindings(binder);
    const auto& text  = out->config.text;
    out->weights.text = loading::bind_text(bindings, text, out->config.mtp);
    out->ple          = loading::bind_ple_tables(bindings, text);
    if (out->config.vision) {
        out->weights.vision = loading::bind_vision(bindings, *out->config.vision, text);
    }
    if (out->config.mtp) {
        out->weights.mtp = loading::bind_mtp(bindings, text, out->weights.text);
    }
    bindings.require_complete(out->config);
    out->pending         = std::move(bindings.weights);
    out->materialization = std::move(binder).finish();
    out->info.name       = reader.directory().metadata.value(
        "name", std::string(architecture_name(Architecture::Qwen4Exp)));
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    auto data    = std::move(plan.impl_);
    auto backing = artifact::materialize(*data->materialization.source,
                                         std::move(data->materialization), device, observer);
    auto bound   = loading::resolve_weights(std::move(data->pending), backing);
    std::vector<PleTable> ple;
    ple.reserve(data->ple.size());
    for (auto& pending : data->ple) {
        PleTable& table = ple.emplace_back();
        table.shards.reserve(pending.shards.size());
        for (const auto& shard : pending.shards) {
            table.shards.push_back(artifact::bind_view(shard, backing));
        }
        table.layer_multipliers      = std::move(pending.layer_multipliers);
        table.ngram_head_offsets     = std::move(pending.ngram_head_offsets);
        table.ngram_head_vocab_sizes = std::move(pending.ngram_head_vocab_sizes);
    }
    return std::unique_ptr<Model>(new Model(
        std::move(data->config), data->options, std::move(data->weights), std::move(bound),
        std::move(ple), std::move(data->resources), std::move(data->info), std::move(backing)));
}

} // namespace ninfer::models::qwen4_exp
