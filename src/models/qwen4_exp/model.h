#pragma once

#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/weights.h"
#include "ninfer/ops/weight_input.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen4_exp {

// The Qwen3.5 Frontend serves Qwen4Exp prompts: same tokenizer family, chat-template mechanics,
// MRoPE prompt layout and Vision preprocessing.
using qwen3_5::FrontendResources;

struct InstanceInfo {
    std::string name;
    std::string metadata_json;
    std::string provenance_json;
    artifact::ArtifactId artifact_id{};
};

// Host half of the per-layer n-gram embedding. Global row r of the table is row r % rows of shard
// r / rows; head h of an n-gram addresses rows [head_offsets[h], head_offsets[h] + vocab[h]).
// Shards are read-only file mappings warmed before readiness, never Device weights.
struct PleTable {
    std::vector<WeightView> shards;
    std::vector<std::int64_t> layer_multipliers;
    std::vector<std::int64_t> ngram_head_offsets;
    std::vector<std::int64_t> ngram_head_vocab_sizes;

    [[nodiscard]] std::uint64_t rows_per_shard() const { return shards.at(0).shape.at(0); }
};

class LoadPlan;

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&)                 = delete;
    Model& operator=(Model&&)      = delete;

    [[nodiscard]] const Config& config() const noexcept { return config_; }

    [[nodiscard]] const LoadOptions& options() const noexcept { return options_; }

    [[nodiscard]] const ModelWeights& weights() const noexcept { return weights_; }

    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }

    [[nodiscard]] ops::WeightInput input(WeightUseId id) const;
    [[nodiscard]] ops::WeightInput input(WeightId id) const;

    // Device parameters only; the mapped PLE shards are in ple_table().
    [[nodiscard]] std::span<const BoundWeight> weight_data() const noexcept { return bound_; }

    [[nodiscard]] const PleTable& ple_table() const noexcept { return ple_table_; }

    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }

    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    Model(Config config, LoadOptions options, ModelWeights weights, std::vector<BoundWeight> bound,
          PleTable ple_table, FrontendResources resources, InstanceInfo info,
          artifact::MaterializedArtifact backing);

    // Destroy all borrowers before backing. The caller keeps DeviceContext alive through cleanup.
    artifact::MaterializedArtifact backing_;
    Config config_;
    LoadOptions options_;
    ModelWeights weights_;
    std::vector<BoundWeight> bound_;
    PleTable ple_table_;
    FrontendResources resources_;
    InstanceInfo info_;
};

} // namespace ninfer::models::qwen4_exp
