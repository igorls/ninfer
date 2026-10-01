#pragma once

#include "models/qwen3_5/model.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/runtime_types.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/model.h"
#include "models/qwen4_exp/program/runtime_types.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/kv_capacity.h"

#include <memory>
#include <variant>

namespace ninfer::runtime {

[[nodiscard]] EngineOptions normalize_engine_options(EngineOptions options);

// One resident model of an architecture package: its immutable Model, the native Parameters
// borrowed by planning and execution, the Frontend and the package's Program.
template <class Package>
struct ModelInstanceOf {
    using ModelContract = typename Package::RuntimeTypes;
    using Model         = typename Package::Model;
    using Parameters    = typename Package::Parameters;

    std::unique_ptr<Model> model;
    const Parameters parameters;
    typename ModelContract::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<typename ModelContract::Program> program;

    ModelInstanceOf(std::unique_ptr<Model> model, const EngineOptions& options);
    ~ModelInstanceOf();
    ModelInstanceOf(const ModelInstanceOf&)            = delete;
    ModelInstanceOf& operator=(const ModelInstanceOf&) = delete;
};

struct Qwen3_5Package {
    using RuntimeTypes = models::qwen3_5::RuntimeTypes;
    using Model        = models::qwen3_5::Model;
    using Parameters   = models::qwen3_5::execution::Parameters;
};

struct Qwen4ExpPackage {
    using RuntimeTypes = models::qwen4_exp::RuntimeTypes;
    using Model        = models::qwen4_exp::Model;
    using Parameters   = models::qwen4_exp::execution::Parameters;
};

using Qwen3_5Instance  = ModelInstanceOf<Qwen3_5Package>;
using Qwen4ExpInstance = ModelInstanceOf<Qwen4ExpPackage>;

// The architecture is resolved once from the artifact; the Engine holds exactly one package's
// instance for its lifetime.
using ModelInstance =
    std::variant<std::unique_ptr<Qwen3_5Instance>, std::unique_ptr<Qwen4ExpInstance>>;

struct ConstructedModel {
    ModelInstance instance;
    LoadSummary load;
    ContextMachineCostModel context_cost;
};

// Loads and plans the model. options.max_concurrency may be lowered to what the KV pool backs
// (clamp_concurrency_to_pool); the caller keeps the effective options.
[[nodiscard]] ConstructedModel construct_model(EngineOptions& options, DeviceContext& device);

} // namespace ninfer::runtime
