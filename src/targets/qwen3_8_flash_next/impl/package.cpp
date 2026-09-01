#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include <ninfer/targets/qwen3_8_flash_next/runtime.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/loader.h"
#include "targets/qwen3_8_flash_next/impl/load/materialized.h"
#include "targets/qwen3_8_flash_next/impl/program_impl.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_8_flash_next::detail {

class LoadPlan::Impl {
public:
    Impl(WeightsProfile weights_profile_in, ArtifactLoadPlan target_plan,
         bool quantize_output_head_fp8_in)
        : weights_profile(weights_profile_in), plan(std::move(target_plan)),
          quantize_output_head_fp8(quantize_output_head_fp8_in) {}

    WeightsProfile weights_profile;
    ArtifactLoadPlan plan;
    bool quantize_output_head_fp8 = false;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadPlan::LoadPlan(LoadPlan&&) noexcept                 = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept      = default;
LoadPlan::~LoadPlan()                                   = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->plan.materialization;
}

class LoadedModel::Impl {
public:
    Impl(BindingPlan plan, artifact::MaterializedArtifact materialized,
         bool quantize_output_head_fp8)
        : data(std::move(plan), std::move(materialized), quantize_output_head_fp8) {}

    LoadedModelData data;
};

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadedModel::~LoadedModel()                                   = default;

} // namespace ninfer::targets::qwen3_8_flash_next::detail

namespace ninfer::targets::qwen3_8_flash_next {

// ---------------------------------------------------------------------------
// Package Static Definitions
// ---------------------------------------------------------------------------

namespace {

constexpr ModelSamplingDefaults kFlashNextDefaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kFlashNextDefaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    detail::validate_identity(identity);
    if (identity.model_id == model_id && identity.weights_id == "mixed-nvfp4-fp8-ple-int4") {
        return WeightsProfile::MixedNvfp4Fp8PleInt4;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    (void)weights_profile;
    auto target_plan = detail::bind_artifact(
        binder, detail::LoadFeatures{.vision = options.enable_vision, .mtp = false});
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile, std::move(target_plan), options.quantize_output_head_fp8));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("load plan is empty"); }

    auto impl = std::make_unique<detail::LoadedModel::Impl>(
        std::move(plan.impl_->plan.bindings), std::move(materialized),
        plan.impl_->quantize_output_head_fp8);
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::make_frontend(
        model.impl_->data.frontend,
        qwen3_6::FrontendOptions{
            .vision_enabled           = model.impl_->data.vision.has_value(),
            .max_context              = options.max_context,
            .media_cache_bytes        = options.media_cache_bytes,
            .media_live_bytes         = options.media_live_bytes,
            .media_preprocess_threads = options.media_preprocess_threads,
        });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile) {
    (void)device;
    (void)weights_profile;
    const std::uint32_t max_concurrency = std::clamp(options.max_concurrency, 1u, 8u);
    // Every active lane needs a source and a destination recurrent-state slot, so the plan's
    // floor is 2 * max_concurrency. Context-cache continuation capacity adds slots on top of that floor.
    const std::uint32_t floor_slots = 2u * max_concurrency;
    const std::uint32_t cont_cap =
        options.context_cache.enabled && options.context_cache.max_private_continuations
            ? *options.context_cache.max_private_continuations
            : 0u;
    const std::uint32_t total_state_slots = floor_slots + cont_cap;
    detail::FlashNextRuntimeConfig config{
        .max_concurrency       = max_concurrency,
        .max_context           = options.max_context,
        .state_slot_capacity   = std::min(64u, total_state_slots),
        .continuation_capacity = cont_cap,
        .prefill_chunk         = options.prefill_chunk,
        .use_cuda_graph        = options.use_cuda_graph,
        .vision_enabled        = options.enable_vision,
        .max_vision_tokens     = 4096,
        .use_qsa_prefill_mma   = options.use_qsa_prefill_mma,
    };
    return SequencePlanner(std::make_unique<detail::SequencePlannerImpl>(config));
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }

    auto program_impl = std::make_unique<detail::ProgramImpl>(
        &model.impl_->data, std::move(plan.impl_->plan), device);
    plan.impl_.reset();
    return std::unique_ptr<Program>(new Program(std::move(program_impl)));
}

} // namespace ninfer::targets::qwen3_8_flash_next
