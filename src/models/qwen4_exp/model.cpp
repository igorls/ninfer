#include "models/qwen4_exp/model.h"

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen4_exp {

Model::Model(Config config, LoadOptions options, ModelWeights weights,
             std::vector<BoundWeight> bound, std::vector<PleTable> ple_tables,
             FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), options_(options),
      weights_(std::move(weights)), bound_(std::move(bound)), ple_tables_(std::move(ple_tables)),
      resources_(std::move(resources)), info_(std::move(info)) {
    if (ple_tables_.size() != config_.text.ple.layers.size() ||
        weights_.text.ple.size() != config_.text.ple.layers.size()) {
        throw std::invalid_argument("Qwen4Exp PLE tables do not match the configured PLE layers");
    }
}

Model::~Model() = default;

ops::WeightInput Model::input(WeightUseId id) const {
    const auto& parameter = weight(id.parameter);
    const auto& use       = parameter.uses.at(id.use_index);
    return {parameter.view, use.policy, use.activation_input_divisor};
}

ops::WeightInput Model::input(WeightId id) const {
    if (weight(id).uses.size() != 1) {
        throw std::invalid_argument("weight input requires an explicit mathematical use");
    }
    return input(WeightUseId{id, 0});
}

} // namespace ninfer::models::qwen4_exp
