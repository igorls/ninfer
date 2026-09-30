#include "models/registry.h"

#include "artifact/schema.h"

#include <stdexcept>
#include <string>

namespace ninfer::models {

Architecture resolve_architecture(std::string_view architecture, std::string_view model_type) {
    if (architecture == "Qwen3_5ForCausalLM" && model_type == "qwen3_5_text") {
        return Architecture::Qwen3_5;
    }
    if (architecture == "Qwen3_5MoeForCausalLM" && model_type == "qwen3_5_moe_text") {
        return Architecture::Qwen3_5Moe;
    }
    if (architecture == "Qwen4ExpForCausalLM" && model_type == "qwen4_exp_text") {
        return Architecture::Qwen4Exp;
    }
    throw std::invalid_argument("unsupported architecture/config pair " +
                                std::string(architecture) + "/" + std::string(model_type));
}

Architecture resolve_architecture(const artifact::Directory& directory) {
    const auto& config = directory.component("text").config;
    if (!config.is_object() || !config.contains("architectures") ||
        !config.contains("model_type")) {
        throw artifact::ArtifactError("text config must name its architecture and model_type");
    }
    const auto& names = config.at("architectures");
    if (!names.is_array() || names.size() != 1) {
        throw artifact::ArtifactError("architectures must name one implementation");
    }
    return resolve_architecture(artifact::require_id(names[0], "architecture"),
                                artifact::require_id(config.at("model_type"), "model_type"));
}

std::string_view architecture_name(Architecture architecture) noexcept {
    switch (architecture) {
    case Architecture::Qwen3_5:
        return "Qwen3_5ForCausalLM";
    case Architecture::Qwen3_5Moe:
        return "Qwen3_5MoeForCausalLM";
    case Architecture::Qwen4Exp:
        return "Qwen4ExpForCausalLM";
    }
    return {};
}

} // namespace ninfer::models
