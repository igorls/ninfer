#pragma once

#include "artifact/binder.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/model.h"
#include "models/qwen4_exp/weights.h"

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen4_exp::loading {

struct PendingWeight {
    artifact::ParameterReference reference;
    std::vector<WeightUse> uses;
    std::vector<std::string> source_objects;
};

struct PendingPleTable {
    std::vector<artifact::ParameterReference> shards;
    std::vector<std::int64_t> layer_multipliers;
    std::vector<std::int64_t> ngram_head_offsets;
    std::vector<std::int64_t> ngram_head_vocab_sizes;
};

// Declares the selected logical parameters and records every Binding and Use it consumes, so a
// load can prove it follows the artifact contract exactly.
class Bindings {
public:
    explicit Bindings(artifact::Binder& binder) : binder(binder) {}

    [[nodiscard]] WeightId parameter(std::string name, artifact::Shape shape,
                                     std::vector<std::string> inputs   = {},
                                     std::optional<QType> exact_format = {});
    [[nodiscard]] WeightId direct(std::string name, artifact::Shape shape,
                                  QType format = QType::BF16);
    [[nodiscard]] artifact::ParameterReference mapped(std::string name, artifact::Shape shape,
                                                      QType format);
    [[nodiscard]] std::vector<std::int64_t> int64_values(std::string name, std::uint64_t elements);

    [[nodiscard]] const PendingWeight& at(WeightId id) const { return weights.at(id.index); }

    [[nodiscard]] WeightUseId use(WeightId id, std::string_view input) const;

    // Every artifact Binding and Use is consumed exactly once, except those that belong only to
    // components this load did not select.
    void require_complete(const Config& config) const;

    artifact::Binder& binder;
    std::vector<PendingWeight> weights;

private:
    void consume(const std::string& name);

    std::map<std::string, WeightId, std::less<>> parameters_;
    std::set<std::string, std::less<>> consumed_;
    std::set<std::pair<std::string, std::string>> consumed_uses_;
};

[[nodiscard]] FrontendResources bind_resources(artifact::Binder& binder, const Config& config);
[[nodiscard]] BlockWeights bind_block(Bindings& bindings, const TextConfig& config,
                                      const std::string& prefix, MixerKind mixer);
[[nodiscard]] HyperMixerWeights bind_mixer(Bindings& bindings, const TextConfig& config,
                                           const std::string& prefix);
[[nodiscard]] TextWeights bind_text(Bindings& bindings, const TextConfig& config, bool mtp);
[[nodiscard]] PendingPleTable bind_ple_table(Bindings& bindings, const TextConfig& config);
[[nodiscard]] MtpWeights bind_mtp(Bindings& bindings, const TextConfig& config,
                                  const TextWeights& target);
[[nodiscard]] VisionWeights bind_vision(Bindings& bindings, const VisionConfig& config,
                                        const TextConfig& target);
[[nodiscard]] std::vector<BoundWeight>
resolve_weights(std::vector<PendingWeight>&& pending,
                const artifact::MaterializedArtifact& materialized);

} // namespace ninfer::models::qwen4_exp::loading
