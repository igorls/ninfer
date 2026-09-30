#include "models/qwen4_exp/load/bindings.h"

#include "artifact/views.h"

#include <cmath>

namespace ninfer::models::qwen4_exp::loading {

void Bindings::consume(const std::string& name) {
    if (!consumed_.insert(name).second) {
        throw artifact::ArtifactError(name + ": duplicate model parameter declaration");
    }
}

WeightUseId Bindings::use(WeightId id, std::string_view input) const {
    const auto& parameter = at(id);
    for (std::size_t i = 0; i < parameter.uses.size(); ++i) {
        if (parameter.uses[i].input == input) { return {id, i}; }
    }
    throw artifact::ArtifactError(parameter.reference.name + ": unresolved Use " +
                                  std::string(input));
}

WeightId Bindings::parameter(std::string name, artifact::Shape shape,
                             std::vector<std::string> inputs, std::optional<QType> exact_format) {
    consume(name);
    PendingWeight pending;
    pending.reference =
        binder.parameter(name, std::move(shape), artifact::Residency::Device, exact_format);
    for (auto& input : inputs) {
        const auto& use = binder.use(name, input);
        if (!use.activation_policy) {
            throw artifact::ArtifactError(name + "@" + input + ": missing activation policy");
        }
        WeightUse result;
        switch (*use.activation_policy) {
        case artifact::ActivationPolicy::A16Only:
            result.policy = ops::LinearPolicy::A16Only;
            break;
        case artifact::ActivationPolicy::AllowA8:
            result.policy = ops::LinearPolicy::AllowA8;
            break;
        case artifact::ActivationPolicy::AllowA4:
            result.policy = ops::LinearPolicy::AllowA4;
            break;
        }
        for (const auto& [role, binding] : use.auxiliaries) {
            if (role != "activation_input_divisor") {
                throw artifact::ArtifactError(name + "@" + input + ": unknown auxiliary " + role);
            }
            const auto value = binder.values(binding, QType::FP32).scalar_f32();
            if (!std::isfinite(value) || value <= 0) {
                throw artifact::ArtifactError(name + "@" + input +
                                              ": activation divisor must be positive finite FP32");
            }
            result.activation_input_divisor = value;
        }
        consumed_uses_.emplace(name, input);
        result.input = std::move(input);
        pending.uses.push_back(std::move(result));
    }
    for (const auto& part : pending.reference.binding.parts) {
        pending.source_objects.push_back(
            artifact::object_id(binder.reader().directory().object(part.object)));
    }
    const WeightId id{weights.size()};
    parameters_.emplace(std::move(name), id);
    weights.push_back(std::move(pending));
    return id;
}

WeightId Bindings::direct(std::string name, artifact::Shape shape, QType format) {
    return parameter(std::move(name), std::move(shape), {}, format);
}

artifact::ParameterReference Bindings::mapped(std::string name, artifact::Shape shape,
                                              QType format) {
    consume(name);
    return binder.parameter(name, std::move(shape), artifact::Residency::Mapped, format);
}

std::vector<std::int64_t> Bindings::int64_values(std::string name, std::uint64_t elements) {
    consume(name);
    const auto found = binder.reader().directory().bindings.find(name);
    if (found == binder.reader().directory().bindings.end()) {
        throw artifact::ArtifactError("missing logical parameter " + name);
    }
    if (found->second.elements != elements) {
        throw artifact::ArtifactError(name + ": logical shape differs from Binding coverage");
    }
    return binder.values(found->second, QType::INT64).integers64();
}

void Bindings::require_complete(const Config& config) const {
    const auto& directory = binder.reader().directory();
    const auto unselected = [&](std::string_view name) {
        return (!config.vision && name.starts_with("vision/")) ||
               (!config.mtp && name.starts_with("mtp/"));
    };
    for (const auto& [name, binding] : directory.bindings) {
        if (consumed_.contains(name) == unselected(name)) {
            throw artifact::ArtifactError(
                name + (unselected(name) ? ": bound for an unselected component"
                                         : ": artifact parameter is not bound by Qwen4Exp"));
        }
    }
    for (const auto& name : consumed_) {
        if (!directory.bindings.contains(name)) {
            throw artifact::ArtifactError(name + ": bound without an artifact Binding");
        }
    }
    for (const auto& [key, use] : directory.uses) {
        const bool skipped = unselected(key.first) || unselected(key.second);
        if (consumed_uses_.contains(key) == skipped) {
            throw artifact::ArtifactError(key.first + "@" + key.second +
                                          (skipped ? ": Use consumed for an unselected component"
                                                   : ": artifact Use is not consumed by Qwen4Exp"));
        }
    }
}

FrontendResources bind_resources(artifact::Binder& binder, const Config& config) {
    const auto resource = [&](std::string_view component, std::string_view role) {
        const auto bytes = binder.host_object(binder.resource(component, role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    FrontendResources out;
    out.tokenizer_json         = resource("text", "tokenizer.json");
    out.tokenizer_config_json  = resource("text", "tokenizer_config.json");
    out.chat_template_jinja    = resource("text", "chat_template.jinja");
    out.generation_config_json = resource("text", "generation_config.json");
    qwen3_5::FrontendGeometry geometry{.embedding_rows = config.text.vocab_size};
    if (config.vision) {
        out.preprocessor_config_json       = resource("vision", "preprocessor_config.json");
        out.video_preprocessor_config_json = resource("vision", "video_preprocessor_config.json");
        geometry.vision = qwen3_5::FrontendGeometry::VisionPatch{config.vision->patch_size,
                                                                 config.vision->temporal_patch_size,
                                                                 config.vision->spatial_merge_size};
    }
    qwen3_5::parse_resources(out, geometry);
    return out;
}

std::vector<BoundWeight> resolve_weights(std::vector<PendingWeight>&& pending,
                                         const artifact::MaterializedArtifact& materialized) {
    std::vector<BoundWeight> out;
    out.reserve(pending.size());
    for (auto& item : pending) {
        auto view = artifact::bind_view(item.reference, materialized);
        out.push_back({std::move(item.reference.name), std::move(item.source_objects),
                       std::move(view), std::move(item.uses)});
    }
    return out;
}

} // namespace ninfer::models::qwen4_exp::loading
