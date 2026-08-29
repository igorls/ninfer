#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include <ninfer/targets/qwen3_8_flash_next/runtime.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/loader.h"
#include "targets/qwen3_8_flash_next/impl/load/materialized.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"
#include "targets/qwen3_8_flash_next/impl/runtime_state.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_8_flash_next::detail {

class LoadPlan::Impl {
public:
    Impl(WeightsProfile weights_profile_in, ArtifactLoadPlan target_plan)
        : weights_profile(weights_profile_in), plan(std::move(target_plan)) {}

    WeightsProfile weights_profile;
    ArtifactLoadPlan plan;
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
    Impl(BindingPlan plan, artifact::MaterializedArtifact materialized)
        : data(std::move(plan), std::move(materialized)) {}

    LoadedModelData data;
};

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadedModel::~LoadedModel()                                   = default;

class RequestBasePlanImpl {
public:
    runtime::RequestPlanSummary summary;
    qwen3_6::PreparedContextCache context_cache;
};

class AdmissionCandidateImpl {
public:
    runtime::RequestPlanSummary summary;
    runtime::IdentityMaterializationAssessment assessment;
};

class PressurePlanningSessionImpl {
public:
};

class SequencePlanImpl {
public:
    explicit SequencePlanImpl(FlashNextRuntimePlan target_plan) : plan(std::move(target_plan)) {}
    FlashNextRuntimePlan plan;
};

class SequencePlannerImpl {
public:
    explicit SequencePlannerImpl(FlashNextRuntimeConfig cfg) : config(cfg) {}
    FlashNextRuntimeConfig config;
};

class ProgramImpl {
public:
    ProgramImpl(LoadedModel::Impl* model, FlashNextRuntimePlan plan_in, DeviceContext& dev)
        : model_(model), plan_(std::move(plan_in)), device_(dev), allocation_(plan_) {}

    LoadedModel::Impl* model_ = nullptr;
    FlashNextRuntimePlan plan_;
    DeviceContext& device_;
    FlashNextRuntimeAllocation allocation_;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail

namespace ninfer::targets::qwen3_8_flash_next {

// ---------------------------------------------------------------------------
// RequestBasePlan
// ---------------------------------------------------------------------------

RequestBasePlan::RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}
RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept            = default;
RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;
RequestBasePlan::~RequestBasePlan()                                     = default;

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept {
    static const runtime::RequestPlanSummary empty{};
    return impl_ != nullptr ? impl_->summary : empty;
}

const qwen3_6::PreparedContextCache& RequestBasePlan::context_cache() const noexcept {
    static const qwen3_6::PreparedContextCache empty{};
    return impl_ != nullptr ? impl_->context_cache : empty;
}

std::optional<PrefixShortlistKey>
RequestBasePlan::prefix_shortlist_key(std::uint32_t /*frontier*/) const noexcept {
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// AdmissionCandidate
// ---------------------------------------------------------------------------

AdmissionCandidate::AdmissionCandidate(
    std::unique_ptr<detail::AdmissionCandidateImpl> impl) noexcept
    : impl_(std::move(impl)) {}
AdmissionCandidate::AdmissionCandidate(AdmissionCandidate&&) noexcept            = default;
AdmissionCandidate& AdmissionCandidate::operator=(AdmissionCandidate&&) noexcept = default;
AdmissionCandidate::~AdmissionCandidate()                                         = default;

const runtime::RequestPlanSummary& AdmissionCandidate::summary() const noexcept {
    static const runtime::RequestPlanSummary empty{};
    return impl_ != nullptr ? impl_->summary : empty;
}

const runtime::IdentityMaterializationAssessment&
AdmissionCandidate::identity_assessment() const noexcept {
    static const runtime::IdentityMaterializationAssessment empty{};
    return impl_ != nullptr ? impl_->assessment : empty;
}

// ---------------------------------------------------------------------------
// ResourcePlan
// ---------------------------------------------------------------------------

ResourcePlan::ResourcePlan(AdmissionCandidate&& admission, std::uint64_t revision,
                           bool needs_transfer) noexcept
    : admission_(std::move(admission)), revision_(revision), needs_transfer_(needs_transfer) {}
ResourcePlan::ResourcePlan(ResourcePlan&&) noexcept            = default;
ResourcePlan& ResourcePlan::operator=(ResourcePlan&&) noexcept = default;
ResourcePlan::~ResourcePlan()                                  = default;

const runtime::RequestPlanSummary& ResourcePlan::summary() const noexcept {
    return admission_.summary();
}

// ---------------------------------------------------------------------------
// PressurePlanningSession
// ---------------------------------------------------------------------------

PressurePlanningSession::PressurePlanningSession(
    std::unique_ptr<detail::PressurePlanningSessionImpl> impl) noexcept
    : impl_(std::move(impl)) {}
PressurePlanningSession::PressurePlanningSession(PressurePlanningSession&&) noexcept            = default;
PressurePlanningSession& PressurePlanningSession::operator=(PressurePlanningSession&&) noexcept = default;
PressurePlanningSession::~PressurePlanningSession()                                             = default;

PressureTargetHandle
PressurePlanningSession::identity_target(const AdmissionCandidate& /*candidate*/) const {
    return {};
}

PressureTargetHandle
PressurePlanningSession::root_maximal_target(const AdmissionCandidate& /*root_candidate*/) {
    return {};
}

runtime::PressureTargetAssessment
PressurePlanningSession::assess(PressureTargetHandle /*target*/) {
    return {};
}

PreparedPressureExpansion
PressurePlanningSession::prepare_expansion(PressureTargetHandle /*parent*/) {
    return PreparedPressureExpansion(nullptr, 0, 0, 0, 0);
}

PressureExpansionView
PressurePlanningSession::commit_expansion(PreparedPressureExpansion&& /*prepared*/) {
    return {};
}

void PressurePlanningSession::discard_expansion(PreparedPressureExpansion&& /*prepared*/) noexcept {}

std::optional<ResourcePlan>
PressurePlanningSession::seal(PressureTargetHandle /*target*/,
                              const qwen3_6::PreparedPrompt& /*prompt*/) {
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// SequencePlan & SequencePlanner
// ---------------------------------------------------------------------------

SequencePlan::SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}
SequencePlan::SequencePlan(SequencePlan&&) noexcept            = default;
SequencePlan& SequencePlan::operator=(SequencePlan&&) noexcept = default;
SequencePlan::~SequencePlan()                                  = default;

std::uint32_t SequencePlan::capacity() const noexcept {
    return impl_ != nullptr ? impl_->plan.resolved_tokens : 0;
}

std::uint32_t SequencePlan::kv_capacity() const noexcept {
    return impl_ != nullptr ? impl_->plan.resolved_tokens : 0;
}

std::uint32_t SequencePlan::max_concurrency() const noexcept {
    return impl_ != nullptr ? impl_->plan.config.max_concurrency : 0;
}

std::size_t SequencePlan::device_reservation_bytes() const noexcept {
    return impl_ != nullptr ? impl_->plan.total_device_bytes : 0;
}

std::size_t SequencePlan::workspace_capacity_bytes() const noexcept {
    return impl_ != nullptr ? impl_->plan.workspace_bytes : 0;
}

SequencePlanner::SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept
    : impl_(std::move(impl)) {}
SequencePlanner::SequencePlanner(SequencePlanner&&) noexcept            = default;
SequencePlanner& SequencePlanner::operator=(SequencePlanner&&) noexcept = default;
SequencePlanner::~SequencePlanner()                                     = default;

runtime::SequenceCapacityCurve SequencePlanner::capacity_curve() const {
    if (impl_ == nullptr) { throw std::logic_error("SequencePlanner is empty"); }
    return detail::flash_next_capacity_curve(impl_->config);
}

SequencePlan SequencePlanner::finalize(std::uint32_t main_page_groups) && {
    if (impl_ == nullptr) { throw std::logic_error("SequencePlanner is empty"); }
    auto plan = detail::finalize_flash_next_runtime_plan(impl_->config, main_page_groups);
    impl_.reset();
    return SequencePlan(std::make_unique<detail::SequencePlanImpl>(std::move(plan)));
}

// ---------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------

Program::Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept : impl_(std::move(impl)) {}
Program::~Program() noexcept                                          = default;

RequestBasePlan Program::plan_request(const qwen3_6::PreparedPrompt& /*prompt*/,
                                      const runtime::ResolvedExecutionOptions& /*options*/) {
    throw std::logic_error("Flash-Next Program: plan_request not implemented");
}

std::optional<AdmissionCandidate>
Program::inspect_admission(const qwen3_6::PreparedPrompt& /*prompt*/,
                           const RequestBasePlan& /*base*/, runtime::LaneId /*destination*/,
                           const ContinuationHandle* /*source*/,
                           const SharedPrefixHandle* /*shared_source*/,
                           std::optional<runtime::CheckpointRef> /*checkpoint*/,
                           bool /*must_retain_private_source*/,
                           const runtime::ContextMachineCostModel& /*machine_cost*/) {
    throw std::logic_error("Flash-Next Program: inspect_admission not implemented");
}

std::optional<ResourcePlan>
Program::seal_identity(const AdmissionCandidate& /*candidate*/,
                       const qwen3_6::PreparedPrompt& /*prompt*/) {
    throw std::logic_error("Flash-Next Program: seal_identity not implemented");
}

PressurePlanningSession
Program::begin_pressure_planning(const runtime::ContextMachineCostModel& /*machine_cost*/,
                                 std::span<const AdmissionCandidate* const> /*candidates*/,
                                 std::span<const ContinuationHandle* const> /*private_owners*/,
                                 std::span<const std::uint32_t> /*private_owner_ordinals*/,
                                 std::span<const SharedPrefixHandle* const> /*shared_owners*/,
                                 std::span<const std::uint32_t> /*shared_owner_ordinals*/) {
    throw std::logic_error("Flash-Next Program: begin_pressure_planning not implemented");
}

runtime::ContextTransactionReserveStatus
Program::start_resource_transaction(ResourcePlan&& /*plan*/,
                                    qwen3_6::PreparedPrompt&& /*prompt*/,
                                    runtime::CancellationFlagView /*cancellation*/) {
    throw std::logic_error("Flash-Next Program: start_resource_transaction not implemented");
}

std::optional<PersistentBackfillProof>
Program::prove_persistent_backfill(const RequestBasePlan& /*blocked_head*/,
                                   const ResourcePlan& /*candidate*/,
                                   std::span<const SequenceHandle> /*persistent_borrowers*/) const {
    return std::nullopt;
}

ContextTransactionProgress
Program::progress_context_transaction(runtime::CancellationFlagView /*cancellation*/) {
    throw std::logic_error("Flash-Next Program: progress_context_transaction not implemented");
}

void Program::finalize_context_transaction() noexcept {}

bool Program::has_context_transaction() const noexcept {
    return false;
}

PrefillProgress
Program::advance_prefill(SequenceHandle /*sequence*/,
                         runtime::ExecutionTiming* /*failed_timing*/) {
    throw std::logic_error("Flash-Next Program: advance_prefill not implemented");
}

CaptureAssessment
Program::inspect_capture(const CaptureOffer& /*offer*/,
                         const SharedPrefixHandle* /*exact_shared*/,
                         const SharedPrefixHandle* /*replacement*/,
                         std::optional<runtime::CheckpointRef> /*private_replacement*/) const {
    return CaptureAssessment{};
}

bool Program::shared_capture_matches(const CaptureOffer& /*offer*/,
                                     const SharedPrefixHandle& /*shared*/) const {
    return false;
}

void Program::skip_capture(CaptureOffer&& /*offer*/) {}

runtime::ContextTransactionReserveStatus
Program::reserve_active_capture(CaptureOffer&& /*offer*/,
                                const SharedPrefixHandle* /*exact_shared*/,
                                const SharedPrefixHandle* /*replacement*/,
                                std::optional<runtime::CheckpointRef> /*private_replacement*/,
                                runtime::CancellationFlagView /*cancellation*/) {
    return runtime::ContextTransactionReserveStatus::Aborted;
}

PendingBatch Program::decode(std::span<const SequenceHandle> /*sequences*/,
                             std::span<const runtime::RoundBudget> /*budgets*/,
                             runtime::ExecutionTiming* /*failed_timing*/) {
    throw std::logic_error("Flash-Next Program: decode not implemented");
}

runtime::ExecutionTiming
Program::append_forced_tokens(std::span<const SequenceHandle> /*sequences*/,
                              std::span<const TokenId> /*row_major_tokens*/,
                              std::uint32_t /*row_stride*/,
                              runtime::ExecutionTiming* /*failed_timing*/) {
    throw std::logic_error("Flash-Next Program: append_forced_tokens not implemented");
}

CommitResult Program::commit(PendingBatch&& /*pending*/,
                             std::span<const runtime::CommitDecision> /*decisions*/,
                             runtime::CommitObservation /*observation*/,
                             runtime::ExecutionTiming* /*failed_timing*/) {
    throw std::logic_error("Flash-Next Program: commit not implemented");
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept {
    return DiscardResult{
        .status    = runtime::ConsumeStatus::Consumed,
        .row_count = pending.row_count(),
    };
}

FinishResult Program::finish(SequenceHandle /*sequence*/) noexcept {
    return FinishResult{
        .status      = runtime::ConsumeStatus::Consumed,
        .disposition = runtime::FinishDisposition::Released,
    };
}

AbortResult Program::abort(SequenceHandle /*sequence*/) noexcept {
    return AbortResult{
        .status = runtime::ConsumeStatus::Consumed,
    };
}

ReleaseResult Program::release_continuation(ContinuationHandle&& /*continuation*/) noexcept {
    return ReleaseResult{
        .status = runtime::ConsumeStatus::Consumed,
    };
}

ReleaseResult Program::release_shared_prefix(SharedPrefixHandle&& /*shared*/) noexcept {
    return ReleaseResult{
        .status = runtime::ConsumeStatus::Consumed,
    };
}

void Program::fail_all_cleanup() noexcept {}

bool Program::isolated_request_feasible(const RequestBasePlan& /*base*/) const noexcept {
    return false;
}

std::uint64_t Program::resource_revision() const noexcept {
    return 0;
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept {
    if (impl_ == nullptr) { return {}; }
    return PhysicalUsageSnapshot{
        .resource_revision       = 0,
        .device_state_slots      = impl_->plan_.state_slots,
        .host_state_slots        = 0,
        .device_main_kv_pages    = impl_->plan_.attention_physical_pages,
        .device_backend_kv_pages = impl_->plan_.indexer_physical_pages,
        .host_kv_bytes           = 0,
    };
}

MemorySummary Program::memory_summary() const noexcept {
    if (impl_ == nullptr) { return {}; }
    MemorySummary out{};
    out.device                          = impl_->device_.device;
    out.max_context                     = impl_->plan_.config.max_context;
    out.kv_capacity                     = impl_->plan_.resolved_tokens;
    out.kv_cache                        = KvCacheStorage::BFloat16;
    out.runtime_reservation_bytes       = impl_->plan_.total_device_bytes;
    out.minimum_runtime_reservation_bytes =
        impl_->plan_.capacity_curve.minimum_device_reservation_bytes;
    out.kv_capacity_increment_bytes =
        impl_->plan_.capacity_curve.bytes_per_additional_main_page_group;
    const std::size_t weights_bytes =
        impl_->model_ != nullptr ? impl_->model_->data.backing.stats().device_capacity_bytes : 0;
    out.weights = ArenaMemorySummary{weights_bytes, weights_bytes, weights_bytes};
    out.sequence =
        ArenaMemorySummary{impl_->plan_.total_device_bytes - impl_->plan_.workspace_bytes,
                           impl_->plan_.total_device_bytes - impl_->plan_.workspace_bytes,
                           impl_->plan_.total_device_bytes - impl_->plan_.workspace_bytes};
    out.workspace = ArenaMemorySummary{impl_->plan_.workspace_bytes, 0, 0};
    return out;
}

void Program::reset_memory_peaks() noexcept {}

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
    return LoadPlan(std::make_unique<detail::LoadPlan::Impl>(
        weights_profile,
        detail::bind_artifact(binder, detail::LoadFeatures{.vision = options.enable_vision,
                                                           .mtp    = false})));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    auto impl = std::make_unique<detail::LoadedModel::Impl>(
        std::move(plan.impl_->plan.bindings), std::move(materialized));
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::make_frontend(
        model.impl_->data.frontend,
        qwen3_6::FrontendOptions{
            .vision_enabled                = model.impl_->data.vision.has_value(),
            .max_context                   = options.max_context,
            .media_cache_bytes             = options.media_cache_bytes,
            .media_live_bytes              = options.media_live_bytes,
            .media_preprocess_threads      = options.media_preprocess_threads,
            .max_cache_markers_per_request = options.context_cache.max_cache_markers_per_request.value_or(0),
        });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile) {
    (void)device;
    (void)weights_profile;
    detail::FlashNextRuntimeConfig config{
        .max_concurrency     = std::clamp(options.max_concurrency, 1u, 8u),
        .max_context         = options.max_context,
        .state_slot_capacity = options.context_cache.max_private_continuations
                                   ? *options.context_cache.max_private_continuations
                                   : 0u,
    };
    return SequencePlanner(std::make_unique<detail::SequencePlannerImpl>(config));
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }

    auto program_impl = std::make_unique<detail::ProgramImpl>(
        model.impl_.get(), std::move(plan.impl_->plan), device);
    plan.impl_.reset();
    return std::unique_ptr<Program>(new Program(std::move(program_impl)));
}

} // namespace ninfer::targets::qwen3_8_flash_next
