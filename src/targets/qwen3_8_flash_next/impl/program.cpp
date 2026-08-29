#include "targets/qwen3_8_flash_next/impl/program_impl.h"
#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include "targets/qwen3_8_flash_next/impl/load/materialized.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_8_flash_next::detail {

ProgramImpl::ProgramImpl(const LoadedModelData* model_data, FlashNextRuntimePlan plan_in,
                         DeviceContext& dev, TextModelView text_override)
    : model_data_(model_data), plan_(std::move(plan_in)), device_(dev), allocation_(plan_),
      executor_(model_data_ != nullptr ? model_data_->text : text_override, kPleIndexMetadata,
                device_, allocation_),
      lane_states_(plan_.config.max_concurrency),
      sampling_workspace_(std::max<std::size_t>(
          256,
          ops::sampling_workspace_capacity_bytes(
              248'077, 1, static_cast<std::int32_t>(plan_.config.max_concurrency)))),
      device_sampling_configs_(plan_.config.max_concurrency * sizeof(ops::SamplingConfig)),
      device_sampling_positions_(plan_.config.max_concurrency * sizeof(std::int32_t)),
      device_sampled_tokens_(plan_.config.max_concurrency * sizeof(std::int32_t)),
      host_sampled_tokens_(plan_.config.max_concurrency, 0),
      host_sampling_configs_(plan_.config.max_concurrency),
      host_sampling_positions_(plan_.config.max_concurrency, 0) {
    allocation_.initialize(device_.stream);
}

void ProgramImpl::sample_tokens(const Tensor& logits,
                                std::span<const std::uint32_t> lane_indices,
                                std::span<std::int32_t> out_tokens) {
    const std::size_t B = lane_indices.size();
    if (B == 0 || B > plan_.config.max_concurrency || out_tokens.size() < B) {
        throw std::invalid_argument("sample_tokens: batch size is invalid");
    }

    for (std::size_t b = 0; b < B; ++b) {
        const std::uint32_t lane_idx = lane_indices[b];
        const auto& st               = lane_states_[lane_idx];
        host_sampling_configs_[b]   = st.sampling_config;
        host_sampling_positions_[b] = st.last_token_pos;
    }

    CUDA_CHECK(cudaMemcpyAsync(device_sampling_configs_.p, host_sampling_configs_.data(),
                               B * sizeof(ops::SamplingConfig), cudaMemcpyHostToDevice,
                               device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(device_sampling_positions_.p, host_sampling_positions_.data(),
                               B * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                               device_.stream));

    Tensor out_tensor(device_sampled_tokens_.p, DType::I32, {static_cast<std::int32_t>(B)});
    Tensor pos_tensor(device_sampling_positions_.p, DType::I32, {static_cast<std::int32_t>(B)});

    constexpr std::int32_t kSemanticTokenDomain = 248'077;
    ops::sample(logits, out_tensor, kSemanticTokenDomain,
                static_cast<const ops::SamplingConfig*>(device_sampling_configs_.p), pos_tensor,
                ops::kSamplePurposeDecode, sampling_workspace_, device_.stream);

    CUDA_CHECK(cudaMemcpyAsync(out_tokens.data(), device_sampled_tokens_.p,
                               B * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                               device_.stream));
    CUDA_CHECK(cudaStreamSynchronize(device_.stream));
}

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
    return PressureTargetHandle{};
}

PressureTargetHandle
PressurePlanningSession::root_maximal_target(const AdmissionCandidate& /*root_candidate*/) {
    return PressureTargetHandle{};
}

runtime::PressureTargetAssessment
PressurePlanningSession::assess(PressureTargetHandle /*target*/) {
    return runtime::PressureTargetAssessment{};
}

PreparedPressureExpansion
PressurePlanningSession::prepare_expansion(PressureTargetHandle /*parent*/) {
    return PreparedPressureExpansion(nullptr, 0, 0, 0, 0);
}

PressureExpansionView
PressurePlanningSession::commit_expansion(PreparedPressureExpansion&& /*prepared*/) {
    return PressureExpansionView{};
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
    if (impl_ == nullptr) { throw std::logic_error("sequence planner is empty"); }
    return detail::flash_next_capacity_curve(impl_->config);
}

SequencePlan SequencePlanner::finalize(std::uint32_t main_page_groups) && {
    if (impl_ == nullptr) { throw std::logic_error("sequence planner is empty"); }
    auto plan = detail::finalize_flash_next_runtime_plan(impl_->config, main_page_groups);
    impl_.reset();
    return SequencePlan(std::make_unique<detail::SequencePlanImpl>(std::move(plan)));
}

// ---------------------------------------------------------------------------
// Program Implementation
// ---------------------------------------------------------------------------

Program::Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept : impl_(std::move(impl)) {}
Program::~Program() noexcept = default;

RequestBasePlan
Program::plan_request(const qwen3_6::PreparedPrompt& prompt,
                      const runtime::ResolvedExecutionOptions& options) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }

    const auto& prompt_data = qwen3_6::PreparedPromptAccess::view(prompt);
    if (prompt_data.token_ids.empty()) {
        throw std::invalid_argument("prompt must contain tokens");
    }
    if (prompt_data.has_media()) {
        throw std::invalid_argument("Vision requests are not supported in Flash-Next cold path");
    }
    if (prompt_data.token_ids.size() > impl_->plan_.resolved_tokens) {
        throw std::invalid_argument("prompt exceeds configured context capacity");
    }
    for (const TokenId id : prompt_data.token_ids) {
        if (id < 0 || id >= 248'077) {
            throw std::invalid_argument("prompt contains token outside the 248077-token domain");
        }
    }
    if (!std::isfinite(options.sampling.temperature) || !std::isfinite(options.sampling.top_p) ||
        !std::isfinite(options.sampling.min_p) || !std::isfinite(options.sampling.presence_penalty) ||
        !std::isfinite(options.sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (options.sampling.top_p < 0.0F || options.sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (options.sampling.min_p < 0.0F || options.sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }

    auto base                   = std::make_unique<detail::RequestBasePlanImpl>();
    base->context_cache         = prompt_data.context_cache;
    base->summary.prompt_tokens = static_cast<std::uint32_t>(prompt_data.token_ids.size());
    base->summary.reusable_prompt_tokens = 0;
    base->summary.requested_output_tokens = options.requested_output_tokens;

    const std::uint32_t capacity_output =
        impl_->plan_.resolved_tokens >= base->summary.prompt_tokens
            ? (impl_->plan_.resolved_tokens - base->summary.prompt_tokens + 1U)
            : 0U;
    base->summary.effective_output_tokens =
        std::min(options.requested_output_tokens, capacity_output);
    base->summary.effective_limit_reason =
        options.requested_output_tokens <= capacity_output
            ? FinishReason::OutputLimit
            : FinishReason::ContextCapacity;
    base->summary.prefix_reuse_path    = PrefixReusePath::Root;
    base->summary.publish_continuation = false;

    const runtime::PrefillWork prefill_work =
        runtime::make_prefill_work(0, base->summary.prompt_tokens, 0, 0, 1);
    base->summary.service_work_quanta =
        prefill_work.tokens + base->summary.effective_output_tokens;

    base->requested_output_tokens = options.requested_output_tokens;
    base->effective_output_tokens = base->summary.effective_output_tokens;
    base->allow_prefix_reuse      = options.allow_prefix_reuse;
    base->thinking                = options.thinking;

    base->sampling_config.temperature       = options.sampling.temperature;
    base->sampling_config.top_k             = options.sampling.top_k;
    base->sampling_config.top_p             = options.sampling.top_p;
    base->sampling_config.min_p             = options.sampling.min_p;
    base->sampling_config.presence_penalty  = options.sampling.presence_penalty;
    base->sampling_config.frequency_penalty = options.sampling.frequency_penalty;
    base->sampling_config.seed              = options.sampling.seed;
    base->sampling_config.token_counts      = nullptr;

    return RequestBasePlan(std::move(base));
}

std::optional<AdmissionCandidate>
Program::inspect_admission(const qwen3_6::PreparedPrompt& /*prompt*/, const RequestBasePlan& base,
                          runtime::LaneId destination, const ContinuationHandle* /*source*/,
                          const SharedPrefixHandle* /*shared_source*/,
                          std::optional<runtime::CheckpointRef> /*checkpoint*/,
                          bool /*must_retain_private_source*/,
                          const runtime::ContextMachineCostModel& /*machine_cost*/) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }

    const std::uint32_t prompt_tokens = base.summary().prompt_tokens;
    const std::uint32_t effective_out = base.summary().effective_output_tokens;
    const std::uint32_t total_tokens  = prompt_tokens + (effective_out > 0 ? effective_out - 1U : 0U);
    const std::uint32_t required_groups = (total_tokens + 255U) / 256U;

    bool lane_available = false;
    if (destination.value < impl_->plan_.config.max_concurrency) {
        lane_available = !impl_->lane_states_[destination.value].active;
    } else {
        lane_available = (impl_->executor_.active_lanes_count() < impl_->plan_.config.max_concurrency);
    }

    const bool groups_available = (impl_->executor_.available_physical_groups() >= required_groups);
    const bool feasible = (total_tokens <= impl_->plan_.resolved_tokens) && lane_available && groups_available;

    const runtime::PrefillWork prefill_work =
        runtime::make_prefill_work(0, prompt_tokens, 0, 0, 1);

    auto cand_impl                              = std::make_unique<detail::AdmissionCandidateImpl>();
    cand_impl->summary                          = base.summary();
    cand_impl->assessment.physical_status       = feasible ? runtime::MaterializationPhysicalStatus::Feasible
                                                           : runtime::MaterializationPhysicalStatus::Infeasible;
    cand_impl->assessment.source_disposition    = runtime::ClaimDisposition::ConsumedToActive;
    cand_impl->assessment.machine.remaining_prefill_work = prefill_work;
    cand_impl->assessment.machine.reused_prompt_tokens   = 0;
    cand_impl->base_plan                        = std::make_unique<detail::RequestBasePlanImpl>(*base.impl_);

    return AdmissionCandidate(std::move(cand_impl));
}

std::optional<ResourcePlan>
Program::seal_identity(const AdmissionCandidate& candidate,
                       const qwen3_6::PreparedPrompt& /*prompt*/) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }
    return ResourcePlan(std::move(const_cast<AdmissionCandidate&>(candidate)),
                        impl_->resource_revision_,
                        false);
}

PressurePlanningSession
Program::begin_pressure_planning(const runtime::ContextMachineCostModel& /*machine_cost*/,
                                 std::span<const AdmissionCandidate* const> /*candidates*/,
                                 std::span<const ContinuationHandle* const> /*private_owners*/,
                                 std::span<const std::uint32_t> /*private_owner_ordinals*/,
                                 std::span<const SharedPrefixHandle* const> /*shared_owners*/,
                                 std::span<const std::uint32_t> /*shared_owner_ordinals*/) {
    return PressurePlanningSession(std::make_unique<detail::PressurePlanningSessionImpl>());
}

runtime::ContextTransactionReserveStatus
Program::start_resource_transaction(ResourcePlan&& plan, qwen3_6::PreparedPrompt&& prompt,
                                    runtime::CancellationFlagView cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }
    if (cancellation.requested()) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    if (impl_->has_context_transaction_) {
        throw std::logic_error("Program: a context transaction is already in progress");
    }

    detail::LaneHandle handle;
    try {
        handle = impl_->executor_.allocate_lane();
    } catch (const std::exception&) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }

    const std::uint32_t lane_idx = handle.lane_index();
    auto& st                     = impl_->lane_states_[lane_idx];
    st.active                    = true;
    st.epoch                     = handle.epoch();
    st.lane_handle               = handle;

    const auto& prompt_data = qwen3_6::PreparedPromptAccess::view(prompt);
    st.prompt_tokens        = prompt_data.token_ids;
    st.mrope_pos0.assign(prompt_data.position_axis(0).begin(), prompt_data.position_axis(0).end());
    st.mrope_pos1.assign(prompt_data.position_axis(1).begin(), prompt_data.position_axis(1).end());
    st.mrope_pos2.assign(prompt_data.position_axis(2).begin(), prompt_data.position_axis(2).end());
    st.prompt_tokens_processed = 0;
    st.last_token_id           = 0;
    st.last_token_pos          = 0;
    st.last_token_index        = 0;
    st.total_generated_tokens  = 0;
    st.requested_output_tokens = plan.summary().requested_output_tokens;
    st.effective_output_tokens = plan.summary().effective_output_tokens;

    auto adm = ContractAccess::take_admission(plan);
    if (adm.impl_ != nullptr && adm.impl_->base_plan != nullptr) {
        st.sampling_config = adm.impl_->base_plan->sampling_config;
    }

    st.prefill_completed = false;
    st.finished          = false;

    impl_->has_context_transaction_ = true;
    impl_->transaction_lane_        = runtime::LaneId(lane_idx);
    impl_->transaction_epoch_       = handle.epoch();
    ++impl_->resource_revision_;

    return runtime::ContextTransactionReserveStatus::Reserved;
}

std::optional<PersistentBackfillProof>
Program::prove_persistent_backfill(const RequestBasePlan& /*blocked_head*/,
                                  const ResourcePlan& /*candidate*/,
                                  std::span<const SequenceHandle> /*persistent_borrowers*/) const {
    return std::nullopt;
}

ContextTransactionProgress
Program::progress_context_transaction(runtime::CancellationFlagView cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }
    if (!impl_->has_context_transaction_) {
        throw std::logic_error("Program: no active context transaction to progress");
    }

    if (cancellation.requested()) {
        if (impl_->transaction_lane_) {
            const std::uint32_t lane_idx = impl_->transaction_lane_->value;
            if (impl_->lane_states_[lane_idx].active) {
                impl_->executor_.release_lane(impl_->lane_states_[lane_idx].lane_handle);
                impl_->lane_states_[lane_idx].active = false;
                ++impl_->resource_revision_;
            }
        }
        impl_->has_context_transaction_ = false;
        impl_->transaction_lane_.reset();
        impl_->transaction_epoch_.reset();
        MaterializationResult res;
        res.status = runtime::ContextTransactionStatus::Aborted;
        return res;
    }

    MaterializationResult res;
    res.status = runtime::ContextTransactionStatus::Published;
    res.published = StartResult{
        .sequence = SequenceHandle(this, impl_->transaction_lane_.value(),
                                   impl_->transaction_epoch_.value()),
    };
    return res;
}

void Program::finalize_context_transaction() noexcept {
    if (impl_ != nullptr) {
        impl_->has_context_transaction_ = false;
        impl_->transaction_lane_.reset();
        impl_->transaction_epoch_.reset();
    }
}

bool Program::has_context_transaction() const noexcept {
    return impl_ != nullptr && impl_->has_context_transaction_;
}

PrefillProgress
Program::advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming* failed_timing) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }

    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (sequence.owner() != this) {
        throw std::logic_error("SequenceHandle does not belong to this Program");
    }
    const std::uint32_t lane_idx = sequence.lane().value;
    if (lane_idx >= impl_->plan_.config.max_concurrency) {
        throw std::out_of_range("sequence lane out of range");
    }
    auto& st = impl_->lane_states_[lane_idx];
    if (!st.active || st.epoch != sequence.epoch()) {
        throw std::logic_error("sequence lane is not active or epoch mismatch");
    }
    if (st.prefill_completed) {
        throw std::logic_error("prefill is already completed for this sequence");
    }

    // Sequence 5c will replace token-by-token loop with chunked prefill kernels
    const std::size_t N        = st.prompt_tokens.size();
    const std::uint32_t start_i = st.prompt_tokens_processed;
    const std::uint32_t end_i   = std::min(static_cast<std::uint32_t>(N), start_i + 16U);

    for (std::uint32_t i = start_i; i < end_i; ++i) {
        if (i + 1 < N) {
            detail::LaneStepRequest req{
                .handle          = st.lane_handle,
                .token_id        = st.prompt_tokens[i],
                .token_index     = static_cast<std::int32_t>(i),
                .mrope_positions = {st.mrope_pos0[i], st.mrope_pos1[i], st.mrope_pos2[i]},
            };
            auto round = impl_->executor_.execute_round(std::span(&req, 1));
            std::array<detail::LaneCommitDecision, 1> decision = {{{.accept = true}}};
            round.commit(decision);
            st.last_token_id    = req.token_id;
            st.last_token_pos   = req.mrope_positions[0];
            st.last_token_index = req.token_index;
        } else {
            // Final prompt token (i == N - 1)
            detail::LaneStepRequest req{
                .handle          = st.lane_handle,
                .token_id        = st.prompt_tokens[i],
                .token_index     = static_cast<std::int32_t>(i),
                .mrope_positions = {st.mrope_pos0[i], st.mrope_pos1[i], st.mrope_pos2[i]},
            };
            impl_->pending_round_ = impl_->executor_.execute_round(std::span(&req, 1));
            st.last_token_id      = req.token_id;
            st.last_token_pos     = req.mrope_positions[0];
            st.last_token_index   = req.token_index;

            // Sample the first output token from logits
            impl_->sample_tokens(impl_->pending_round_.logits(), std::span(&lane_idx, 1),
                                 std::span(impl_->host_sampled_tokens_.data(), 1));

            impl_->pending_batch_tokens_.resize(1);
            impl_->pending_batch_row_counts_.resize(1);
            impl_->pending_batch_tokens_[0]     = static_cast<TokenId>(impl_->host_sampled_tokens_[0]);
            impl_->pending_batch_row_counts_[0] = 1;

            st.prefill_completed = true;
        }
    }
    st.prompt_tokens_processed = end_i;

    PrefillProgress progress;
    progress.summary.prompt_tokens        = static_cast<std::uint32_t>(N);
    progress.summary.reused_prompt_tokens = 0;
    progress.summary.prefix_reuse_path    = PrefixReusePath::Root;
    progress.processed_prompt_tokens      = end_i - start_i;
    progress.complete                     = (end_i == N);
    progress.timing                       = timing.finish();

    if (progress.complete) {
        progress.pending = ContractAccess::make_pending(
            this, impl_->pending_round_.valid() ? 1 : 0, std::span(&sequence, 1),
            std::span(impl_->pending_batch_tokens_.data(), 1),
            std::span(impl_->pending_batch_row_counts_.data(), 1), 1, progress.timing);
    }
    return progress;
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

PendingBatch Program::decode(std::span<const SequenceHandle> sequences,
                             std::span<const runtime::RoundBudget> /*budgets*/,
                             runtime::ExecutionTiming* failed_timing) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }

    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    const std::size_t B = sequences.size();
    if (B == 0 || B > impl_->plan_.config.max_concurrency) {
        throw std::invalid_argument("decode batch size invalid");
    }
    if (impl_->pending_round_.valid()) {
        throw std::logic_error("cannot decode while a pending round is uncommitted");
    }

    std::vector<detail::LaneStepRequest> requests(B);
    std::vector<std::uint32_t> lane_indices(B);

    for (std::size_t b = 0; b < B; ++b) {
        const auto& seq = sequences[b];
        if (seq.owner() != this) {
            throw std::logic_error("SequenceHandle does not belong to this Program");
        }
        const std::uint32_t lane_idx = seq.lane().value;
        if (lane_idx >= impl_->plan_.config.max_concurrency) {
            throw std::out_of_range("sequence lane out of range");
        }
        auto& st = impl_->lane_states_[lane_idx];
        if (!st.active || st.epoch != seq.epoch()) {
            throw std::logic_error("sequence lane is not active or epoch mismatch");
        }
        lane_indices[b] = lane_idx;

        requests[b] = detail::LaneStepRequest{
            .handle          = st.lane_handle,
            .token_id        = st.last_token_id,
            .token_index     = st.last_token_index,
            .mrope_positions = {st.last_token_pos, st.last_token_pos, st.last_token_pos},
        };
    }

    impl_->pending_round_ = impl_->executor_.execute_round(requests);

    impl_->sample_tokens(impl_->pending_round_.logits(), lane_indices,
                         std::span(impl_->host_sampled_tokens_.data(), B));

    impl_->pending_batch_tokens_.resize(B);
    impl_->pending_batch_row_counts_.resize(B);
    for (std::size_t b = 0; b < B; ++b) {
        impl_->pending_batch_tokens_[b]     = static_cast<TokenId>(impl_->host_sampled_tokens_[b]);
        impl_->pending_batch_row_counts_[b] = 1;
    }

    const auto exec_timing = timing.finish();
    return ContractAccess::make_pending(
        this, impl_->pending_round_.valid() ? 1 : 0, sequences,
        std::span(impl_->pending_batch_tokens_.data(), B),
        std::span(impl_->pending_batch_row_counts_.data(), B), 1, exec_timing);
}

runtime::ExecutionTiming
Program::append_forced_tokens(std::span<const SequenceHandle> sequences,
                              std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                              runtime::ExecutionTiming* failed_timing) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }

    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    const std::size_t B = sequences.size();
    if (B == 0 || B > impl_->plan_.config.max_concurrency || row_stride == 0 ||
        row_major_tokens.size() != B * row_stride) {
        throw std::invalid_argument("append_forced_tokens arguments invalid");
    }
    if (impl_->pending_round_.valid()) {
        throw std::logic_error("cannot append forced tokens while pending round is uncommitted");
    }

    for (std::size_t b = 0; b < B; ++b) {
        const auto& seq = sequences[b];
        if (seq.owner() != this) {
            throw std::logic_error("SequenceHandle does not belong to this Program");
        }
        const std::uint32_t lane_idx = seq.lane().value;
        auto& st                     = impl_->lane_states_[lane_idx];
        if (!st.active || st.epoch != seq.epoch()) {
            throw std::logic_error("sequence lane not active or epoch mismatch");
        }

        for (std::uint32_t s = 0; s < row_stride; ++s) {
            const TokenId tok = row_major_tokens[b * row_stride + s];
            detail::LaneStepRequest req{
                .handle          = st.lane_handle,
                .token_id        = static_cast<std::int32_t>(tok),
                .token_index     = st.last_token_index + 1,
                .mrope_positions = {st.last_token_pos + 1, st.last_token_pos + 1,
                                    st.last_token_pos + 1},
            };
            auto round = impl_->executor_.execute_round(std::span(&req, 1));
            std::array<detail::LaneCommitDecision, 1> decision = {{{.accept = true}}};
            round.commit(decision);

            st.last_token_id    = req.token_id;
            st.last_token_pos   = req.mrope_positions[0];
            st.last_token_index = req.token_index;
            ++st.total_generated_tokens;
        }
    }

    return timing.finish();
}

CommitResult
Program::commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
                runtime::CommitObservation /*observation*/,
                runtime::ExecutionTiming* failed_timing) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }

    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    const auto rows     = ContractAccess::rows(pending);
    const std::size_t B = rows.size();
    if (B != decisions.size()) {
        throw std::invalid_argument("decisions size mismatch with pending rows");
    }

    std::vector<detail::LaneCommitDecision> lane_decisions(B);
    CommitResult result;
    result.row_count = B;

    for (std::size_t b = 0; b < B; ++b) {
        const auto& dec          = decisions[b];
        lane_decisions[b].accept = (dec.accepted_tokens > 0);
    }

    if (impl_->pending_round_.valid()) {
        impl_->pending_round_.commit(lane_decisions);
    }

    for (std::size_t b = 0; b < B; ++b) {
        const auto& seq              = rows[b];
        const auto& dec              = decisions[b];
        const std::uint32_t lane_idx = seq.lane().value;
        auto& st                     = impl_->lane_states_[lane_idx];

        if (dec.accepted_tokens > 0) {
            const TokenId sampled = pending.tokens()[b];
            st.last_token_id      = static_cast<std::int32_t>(sampled);
            st.last_token_pos += 1;
            st.last_token_index += 1;
            ++st.total_generated_tokens;

            if (dec.terminal) {
                impl_->executor_.release_lane(st.lane_handle);
                st.active   = false;
                st.finished = true;
                ++impl_->resource_revision_;
                result.rows[b].disposition = runtime::CommitDisposition::Finishable;
            } else {
                result.rows[b].disposition = runtime::CommitDisposition::Active;
            }
        } else {
            impl_->executor_.release_lane(st.lane_handle);
            st.active   = false;
            st.finished = true;
            ++impl_->resource_revision_;
            result.rows[b].disposition = runtime::CommitDisposition::CancelledReleased;
        }
    }

    ContractAccess::consume(pending);
    result.timing = timing.finish();
    return result;
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept {
    if (impl_ != nullptr && impl_->pending_round_.valid()) {
        impl_->pending_round_.abort();
    }
    ContractAccess::consume(pending);
    return DiscardResult{
        .status    = runtime::ConsumeStatus::Consumed,
        .row_count = 0,
    };
}

FinishResult Program::finish(SequenceHandle sequence) noexcept {
    if (impl_ != nullptr) {
        const std::uint32_t lane_idx = sequence.lane().value;
        if (lane_idx < impl_->plan_.config.max_concurrency) {
            auto& st = impl_->lane_states_[lane_idx];
            if (st.active && st.epoch == sequence.epoch()) {
                impl_->executor_.release_lane(st.lane_handle);
                st.active   = false;
                st.finished = true;
                ++impl_->resource_revision_;
            }
        }
    }
    return FinishResult{
        .status      = runtime::ConsumeStatus::Consumed,
        .disposition = runtime::FinishDisposition::Released,
    };
}

AbortResult Program::abort(SequenceHandle sequence) noexcept {
    if (impl_ != nullptr) {
        const std::uint32_t lane_idx = sequence.lane().value;
        if (lane_idx < impl_->plan_.config.max_concurrency) {
            auto& st = impl_->lane_states_[lane_idx];
            if (st.active && st.epoch == sequence.epoch()) {
                impl_->executor_.release_lane(st.lane_handle);
                st.active   = false;
                st.finished = true;
                ++impl_->resource_revision_;
            }
        }
    }
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

void Program::fail_all_cleanup() noexcept {
    if (impl_ == nullptr) { return; }
    if (impl_->pending_round_.valid()) {
        impl_->pending_round_.abort();
    }
    for (std::uint32_t l = 0; l < impl_->plan_.config.max_concurrency; ++l) {
        auto& st = impl_->lane_states_[l];
        if (st.active) {
            impl_->executor_.release_lane(st.lane_handle);
            st.active   = false;
            st.finished = true;
        }
    }
    impl_->has_context_transaction_ = false;
    impl_->transaction_lane_.reset();
    impl_->transaction_epoch_.reset();
    ++impl_->resource_revision_;
}

bool Program::isolated_request_feasible(const RequestBasePlan& base) const noexcept {
    if (impl_ == nullptr) { return false; }
    const std::uint32_t prompt_tokens = base.summary().prompt_tokens;
    const std::uint32_t effective_out = base.summary().effective_output_tokens;
    const std::uint32_t total_tokens  = prompt_tokens + (effective_out > 0 ? effective_out - 1U : 0U);
    return total_tokens <= impl_->plan_.resolved_tokens &&
           (impl_->plan_.main_page_groups >= (total_tokens + 255U) / 256U) &&
           (impl_->executor_.active_lanes_count() < impl_->plan_.config.max_concurrency);
}

std::uint64_t Program::resource_revision() const noexcept {
    return impl_ != nullptr ? impl_->resource_revision_ : 0;
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept {
    if (impl_ == nullptr) { return {}; }
    return PhysicalUsageSnapshot{
        .resource_revision       = impl_->resource_revision_,
        .device_state_slots      = static_cast<std::uint32_t>(impl_->executor_.active_lanes_count()),
        .host_state_slots        = 0,
        .device_main_kv_pages    = static_cast<std::uint32_t>(
            (impl_->plan_.main_page_groups - impl_->executor_.available_physical_groups()) * 4),
        .device_backend_kv_pages = 0,
        .host_kv_bytes           = 0,
    };
}

MemorySummary Program::memory_summary() const noexcept {
    if (impl_ == nullptr) { return {}; }
    MemorySummary out{};
    out.device                            = impl_->device_.device;
    out.max_context                       = impl_->plan_.config.max_context;
    out.kv_capacity                       = impl_->plan_.resolved_tokens;
    out.kv_cache                          = KvCacheStorage::BFloat16;
    out.runtime_reservation_bytes         = impl_->plan_.total_device_bytes;
    out.minimum_runtime_reservation_bytes =
        impl_->plan_.capacity_curve.minimum_device_reservation_bytes;
    out.kv_capacity_increment_bytes =
        impl_->plan_.capacity_curve.bytes_per_additional_main_page_group;
    const std::size_t weights_bytes =
        impl_->model_data_ != nullptr ? impl_->model_data_->backing.stats().device_capacity_bytes : 0;
    out.weights = ArenaMemorySummary{weights_bytes, weights_bytes, weights_bytes};
    out.sequence =
        ArenaMemorySummary{impl_->plan_.total_device_bytes - impl_->plan_.workspace_bytes,
                           impl_->plan_.total_device_bytes - impl_->plan_.workspace_bytes,
                           impl_->plan_.total_device_bytes - impl_->plan_.workspace_bytes};
    out.workspace = ArenaMemorySummary{impl_->plan_.workspace_bytes, 0, 0};
    return out;
}

void Program::reset_memory_peaks() noexcept {}

} // namespace ninfer::targets::qwen3_8_flash_next
