#include "targets/qwen3_8_flash_next/impl/program_impl.h"
#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include "targets/qwen3_8_flash_next/impl/load/materialized.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_8_flash_next::detail {

ProgramImpl::ProgramImpl(const LoadedModelData* model_data, FlashNextRuntimePlan plan_in,
                         DeviceContext& dev, TextModelView text_override,
                         std::optional<VisionModelView> vision_override,
                         PleIndexMetadata ple_override)
    : model_data_(model_data), text_override_(std::move(text_override)),
      vision_override_(std::move(vision_override)),
      plan_(std::move(plan_in)), device_(dev), allocation_(plan_),
      executor_(model_data_ != nullptr ? model_data_->text : text_override_,
                model_data_ != nullptr ? kPleIndexMetadata : ple_override,
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
    const std::uint32_t cont_cap = plan_.config.continuation_capacity;
    continuation_slots_.resize(cont_cap);
    for (std::uint32_t c = 0; c < cont_cap; ++c) {
        continuation_slots_[c].cache_slot = 2U * plan_.config.max_concurrency + c;
    }
    if (model_data_ != nullptr && model_data_->vision.has_value() && plan_.config.vision_enabled) {
        vision_session_.emplace(*model_data_->vision, device_, plan_.config.max_vision_tokens);
    } else if (vision_override_.has_value() && plan_.config.vision_enabled) {
        vision_session_.emplace(*vision_override_, device_, plan_.config.max_vision_tokens);
    }
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
RequestBasePlan::prefix_shortlist_key(std::uint32_t frontier) const noexcept {
    if (impl_ == nullptr || frontier == 0 || frontier > impl_->prefix_digests.size()) {
        return std::nullopt;
    }
    return PrefixShortlistKey{
        .digest       = impl_->prefix_digests.at(frontier),
        .frontier     = frontier,
        .identity_tag = impl_->prefix_identity_tag,
    };
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
        if (!impl_->vision_session_.has_value() &&
            (impl_->model_data_ == nullptr || !impl_->model_data_->vision.has_value())) {
            throw std::invalid_argument("Vision requests require vision features enabled in model");
        }
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
    base->summary.publish_continuation =
        options.allow_prefix_reuse && prompt_data.identity.reusable &&
        (impl_->plan_.config.continuation_capacity > 0);

    const runtime::PrefillWork prefill_work =
        runtime::make_prefill_work(0, base->summary.prompt_tokens, 0, 0,
                                   impl_->plan_.config.prefill_chunk);
    base->summary.service_work_quanta =
        prefill_work.tokens + base->summary.effective_output_tokens;

    base->requested_output_tokens = options.requested_output_tokens;
    base->effective_output_tokens = base->summary.effective_output_tokens;
    base->allow_prefix_reuse      = options.allow_prefix_reuse;
    base->thinking                = options.thinking;
    base->prefix_digests.assign(prompt_data);
    base->prefix_identity_tag     = 0;

    base->sampling_config.temperature       = options.sampling.temperature;
    base->sampling_config.top_k             = options.sampling.top_k;
    base->sampling_config.top_p             = options.sampling.top_p;
    base->sampling_config.min_p             = options.sampling.min_p;
    base->sampling_config.presence_penalty  = options.sampling.presence_penalty;
    base->sampling_config.frequency_penalty = options.sampling.frequency_penalty;
    base->sampling_config.seed              = options.sampling.seed;
    base->sampling_config.token_counts      = nullptr;

    if (prompt_data.has_media()) {
        auto control_plan         = qwen3_6::plan_vision_control(prompt_data);
        auto control              = qwen3_6::build_vision_control(prompt_data, control_plan, 0);
        base->vision_control_plan = std::move(control_plan);
        base->vision_control      = std::move(control);
    }

    return RequestBasePlan(std::move(base));
}

std::optional<AdmissionCandidate>
Program::inspect_admission(const qwen3_6::PreparedPrompt& prompt, const RequestBasePlan& base,
                          runtime::LaneId destination, const ContinuationHandle* source,
                          const SharedPrefixHandle* shared_source,
                          std::optional<runtime::CheckpointRef> /*checkpoint*/,
                          bool must_retain_private_source,
                          const runtime::ContextMachineCostModel& /*machine_cost*/) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }

    const auto& prompt_data = qwen3_6::PreparedPromptAccess::view(prompt);
    const std::uint32_t prompt_tokens = base.summary().prompt_tokens;
    const std::uint32_t effective_out = base.summary().effective_output_tokens;

    std::uint32_t reusable_tokens = 0;
    const detail::ContinuationSlot* cont_slot = nullptr;

    if (shared_source != nullptr) {
        return std::nullopt;
    }

    std::uint32_t matched_slot_index = 0;
    if (source != nullptr) {
        if (source->owner() != this) {
            return std::nullopt;
        }
        const std::uint32_t c_idx = source->index();
        const std::uint64_t gen   = source->generation();
        if (c_idx >= impl_->continuation_slots_.size()) {
            return std::nullopt;
        }
        const auto& c = impl_->continuation_slots_[c_idx];
        if (c.role != detail::ContinuationSlotRole::Catalogued || c.generation != gen) {
            return std::nullopt;
        }

        // Try matching endpoint continuation
        const std::size_t K = static_cast<std::size_t>(c.committed_frontier);
        if (K > 0 && K <= prompt_data.token_ids.size() && K <= c.committed_tokens.size()) {
            bool match = true;
            for (std::size_t i = 0; i < K; ++i) {
                if (prompt_data.token_ids[i] != c.committed_tokens[i]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                reusable_tokens    = static_cast<std::uint32_t>(K);
                cont_slot          = &c;
                matched_slot_index = c_idx;
            }
        }

        // If endpoint didn't match (e.g. omitted reasoning / divergence in multi-turn), check TurnClosure checkpoints
        if (cont_slot == nullptr) {
            for (std::size_t tc_idx = 0; tc_idx < impl_->continuation_slots_.size(); ++tc_idx) {
                const auto& tc = impl_->continuation_slots_[tc_idx];
                if (tc.role == detail::ContinuationSlotRole::Catalogued &&
                    tc.kind == runtime::CheckpointKind::TurnClosure) {
                    const std::size_t tc_K = static_cast<std::size_t>(tc.committed_frontier);
                    if (tc_K > 0 && tc_K <= prompt_data.token_ids.size() && tc_K <= tc.committed_tokens.size()) {
                        bool match = true;
                        for (std::size_t i = 0; i < tc_K; ++i) {
                            if (prompt_data.token_ids[i] != tc.committed_tokens[i]) {
                                match = false;
                                break;
                            }
                        }
                        if (match && tc_K > reusable_tokens) {
                            reusable_tokens    = static_cast<std::uint32_t>(tc_K);
                            cont_slot          = &tc;
                            matched_slot_index = static_cast<std::uint32_t>(tc_idx);
                        }
                    }
                }
            }
        }

        // If neither endpoint nor TurnClosure matched, admission against this source fails
        if (cont_slot == nullptr) {
            return std::nullopt;
        }
    }

    const std::uint32_t total_tokens = prompt_tokens + (effective_out > 0 ? effective_out - 1U : 0U);
    const std::uint32_t total_required_groups = (total_tokens + 255U) / 256U;
    const std::uint32_t cont_groups =
        cont_slot != nullptr ? static_cast<std::uint32_t>(cont_slot->physical_groups.size()) : 0U;
    const std::uint32_t additional_groups_needed =
        total_required_groups > cont_groups ? (total_required_groups - cont_groups) : 0U;

    bool lane_available = false;
    if (destination.value < impl_->plan_.config.max_concurrency) {
        lane_available = !impl_->lane_states_[destination.value].active;
    } else {
        lane_available = (impl_->executor_.active_lanes_count() < impl_->plan_.config.max_concurrency);
    }

    std::uint32_t evictable_groups = 0;
    for (const auto& c : impl_->continuation_slots_) {
        if (c.role == detail::ContinuationSlotRole::Catalogued && &c != cont_slot) {
            evictable_groups += static_cast<std::uint32_t>(c.physical_groups.size());
        }
    }
    const std::uint32_t total_freeable_groups =
        static_cast<std::uint32_t>(impl_->executor_.available_physical_groups()) + evictable_groups;
    const bool groups_available = (total_freeable_groups >= additional_groups_needed);
    const bool feasible = (total_tokens <= impl_->plan_.resolved_tokens) && lane_available && groups_available;

    const runtime::PrefillWork prefill_work =
        runtime::make_prefill_work(reusable_tokens, prompt_tokens, 0, 0, impl_->plan_.config.prefill_chunk);

    auto cand_impl                            = std::make_unique<detail::AdmissionCandidateImpl>();
    cand_impl->summary                        = base.summary();
    cand_impl->summary.reusable_prompt_tokens = reusable_tokens;
    cand_impl->summary.prefix_reuse_path =
        reusable_tokens > 0
            ? (cont_slot && cont_slot->kind == runtime::CheckpointKind::TurnClosure
                   ? PrefixReusePath::PrivateTurnClosure
                   : PrefixReusePath::PrivateEndpoint)
            : PrefixReusePath::Root;
    cand_impl->assessment.physical_status = feasible ? runtime::MaterializationPhysicalStatus::Feasible
                                                   : runtime::MaterializationPhysicalStatus::Infeasible;
    cand_impl->assessment.source_disposition =
        (source != nullptr && (must_retain_private_source ||
                               (cont_slot != nullptr && cont_slot->kind == runtime::CheckpointKind::TurnClosure)))
            ? runtime::ClaimDisposition::Retained
            : runtime::ClaimDisposition::ConsumedToActive;
    cand_impl->assessment.expandable                     = false;
    cand_impl->assessment.projection_work                = 1;
    cand_impl->assessment.machine.remaining_prefill_work = prefill_work;
    cand_impl->assessment.machine.reused_prompt_tokens   = reusable_tokens;
    const std::uint64_t est_ns                           = static_cast<std::uint64_t>(prefill_work.tokens) * 1000ULL;
    cand_impl->assessment.machine.immediate_ns           = est_ns;
    cand_impl->assessment.machine.minimum_request_ns     = est_ns;
    cand_impl->base_plan                        = std::make_unique<detail::RequestBasePlanImpl>(*base.impl_);
    cand_impl->reusable_tokens                  = reusable_tokens;
    if (cont_slot != nullptr) {
        cand_impl->source_continuation_index      = matched_slot_index;
        cand_impl->source_continuation_generation = cont_slot->generation;
    }

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

static std::optional<std::uint32_t>
derive_turn_closure_frontier(const qwen3_6::PreparedPromptData& prompt_data) {
    if (prompt_data.identity.rewrite_checkpoint &&
        prompt_data.identity.rewrite_checkpoint->frontier > 0 &&
        prompt_data.identity.rewrite_checkpoint->frontier < prompt_data.token_ids.size()) {
        return prompt_data.identity.rewrite_checkpoint->frontier;
    }
    std::optional<std::uint32_t> last_opp;
    for (const auto& opp : prompt_data.context_cache.opportunities) {
        if (opp.frontier > 0 && opp.frontier < prompt_data.token_ids.size()) {
            if (!last_opp || opp.frontier > *last_opp) {
                last_opp = opp.frontier;
            }
        }
    }
    return last_opp;
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

    const runtime::RequestPlanSummary summary = plan.summary();
    auto adm = ContractAccess::take_admission(plan);
    const std::uint32_t prompt_tokens = summary.prompt_tokens;
    const std::uint32_t effective_out  = summary.effective_output_tokens;
    const std::uint32_t total_tokens   = prompt_tokens + (effective_out > 0 ? effective_out - 1U : 0U);
    const std::uint32_t total_req_groups = (total_tokens + 255U) / 256U;
    const std::uint32_t cont_groups =
        (adm.impl_ != nullptr && adm.impl_->reusable_tokens > 0 &&
         adm.impl_->source_continuation_index < impl_->continuation_slots_.size())
            ? static_cast<std::uint32_t>(
                  impl_->continuation_slots_[adm.impl_->source_continuation_index].physical_groups.size())
            : 0U;
    const std::uint32_t additional_needed =
        total_req_groups > cont_groups ? (total_req_groups - cont_groups) : 0U;

    while (impl_->executor_.available_physical_groups() < additional_needed) {
        std::uint32_t oldest_idx = impl_->plan_.config.continuation_capacity;
        std::uint64_t oldest_epoch = std::numeric_limits<std::uint64_t>::max();
        for (std::uint32_t c = 0; c < impl_->plan_.config.continuation_capacity; ++c) {
            auto& slot = impl_->continuation_slots_[c];
            if (slot.role == detail::ContinuationSlotRole::Catalogued) {
                if (adm.impl_ != nullptr && adm.impl_->reusable_tokens > 0 &&
                    c == adm.impl_->source_continuation_index) {
                    continue;
                }
                if (slot.last_used_epoch < oldest_epoch) {
                    oldest_epoch = slot.last_used_epoch;
                    oldest_idx = c;
                }
            }
        }
        if (oldest_idx == impl_->plan_.config.continuation_capacity) {
            break;
        }
        auto& victim = impl_->continuation_slots_[oldest_idx];
        impl_->executor_.release_physical_groups(victim.physical_groups);
        victim.physical_groups.clear();
        victim.committed_tokens.clear();
        victim.committed_frontier = 0;
        victim.role = detail::ContinuationSlotRole::Vacant;
        victim.generation++;
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

    auto prompt_data        = qwen3_6::PreparedPromptAccess::take(std::move(prompt));
    st.capture_frontier     = derive_turn_closure_frontier(prompt_data);
    st.capture_offered      = false;
    st.mrope_pos0.assign(prompt_data.position_axis(0).begin(), prompt_data.position_axis(0).end());
    st.mrope_pos1.assign(prompt_data.position_axis(1).begin(), prompt_data.position_axis(1).end());
    st.mrope_pos2.assign(prompt_data.position_axis(2).begin(), prompt_data.position_axis(2).end());
    st.prefix_digests.assign(prompt_data);
    st.prompt_tokens        = std::move(prompt_data.token_ids);
    st.media_payloads       = std::move(prompt_data.media_payloads);
    st.vision_items         = std::move(prompt_data.vision_items);
    st.turn_closure_continuation_index = std::nullopt;
    st.pending_capture_offer = 0;
    st.reused_from_turn_closure = (adm.impl_ != nullptr && adm.impl_->summary.prefix_reuse_path == PrefixReusePath::PrivateTurnClosure);
    st.prompt_tokens_processed = 0;
    st.reused_prompt_tokens    = 0;
    st.committed_frontier      = 0;
    st.last_token_id           = 0;
    st.last_token_pos          = 0;
    st.last_token_index        = 0;
    st.total_generated_tokens  = 0;
    st.requested_output_tokens = summary.requested_output_tokens;
    st.effective_output_tokens = summary.effective_output_tokens;
    st.publish_continuation    = summary.publish_continuation;

    if (adm.impl_ != nullptr && adm.impl_->base_plan != nullptr) {
        st.sampling_config = adm.impl_->base_plan->sampling_config;
        st.vision_control  = std::move(adm.impl_->base_plan->vision_control);
    }

    if (adm.impl_ != nullptr && adm.impl_->reusable_tokens > 0) {
        const std::uint32_t c_idx = adm.impl_->source_continuation_index;
        const std::uint64_t gen   = adm.impl_->source_continuation_generation;
        if (c_idx < impl_->continuation_slots_.size()) {
            auto& c_slot = impl_->continuation_slots_[c_idx];
            if (c_slot.role == detail::ContinuationSlotRole::Catalogued && c_slot.generation == gen) {
                // Attach physical groups
                impl_->executor_.attach_physical_groups(
                    handle, c_slot.physical_groups, c_slot.committed_frontier, c_slot.history);

                // Copy-on-Resume recurrent state from cache slot to active lane slot
                const std::int32_t active_slot = impl_->executor_.allocation().current_source_slot(lane_idx);
                impl_->executor_.copy_state_slot(c_slot.cache_slot, static_cast<std::uint32_t>(active_slot));

                st.prompt_tokens_processed = adm.impl_->reusable_tokens;
                st.reused_prompt_tokens    = adm.impl_->reusable_tokens;
                st.last_token_pos          = static_cast<std::int32_t>(adm.impl_->reusable_tokens) - 1;
                st.last_token_index        = static_cast<std::int32_t>(adm.impl_->reusable_tokens) - 1;
                st.committed_frontier      = static_cast<std::int32_t>(adm.impl_->reusable_tokens);

                if (c_slot.kind != runtime::CheckpointKind::TurnClosure) {
                    // Consumed to active: release continuation cache entry (page groups transferred to active lane)
                    impl_->executor_.release_physical_groups(c_slot.physical_groups);
                    c_slot.physical_groups.clear();
                    c_slot.committed_tokens.clear();
                    c_slot.committed_frontier = 0;
                    c_slot.role               = detail::ContinuationSlotRole::Vacant;
                    c_slot.generation++;
                } else {
                    // TurnClosure checkpoint: stays catalogued and immutable for future turns / sibling requests!
                    c_slot.last_used_epoch = ++impl_->continuation_epoch_;
                }
            }
        }
    }

    st.prefill_completed = false;
    st.finished          = false;

    impl_->has_context_transaction_ = true;
    impl_->transaction_lane_        = runtime::LaneId(lane_idx);
    impl_->transaction_epoch_       = handle.epoch();
    impl_->transaction_has_source_  = (adm.impl_ != nullptr && adm.impl_->reusable_tokens > 0);
    impl_->transaction_source_disposition_ =
        adm.impl_ != nullptr ? adm.impl_->assessment.source_disposition : runtime::ClaimDisposition::ConsumedToActive;
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

    if (impl_->is_capture_transaction_) {
        ActiveCaptureResult res = std::move(impl_->pending_capture_result_);
        return res;
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
        impl_->transaction_has_source_ = false;
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
    if (impl_->transaction_has_source_) {
        res.source.emplace(qwen3_6::MaterializationSourceResult{
            .disposition = impl_->transaction_source_disposition_,
        });
    }
    return res;
}

void Program::finalize_context_transaction() noexcept {
    if (impl_ != nullptr) {
        impl_->has_context_transaction_ = false;
        impl_->is_capture_transaction_  = false;
        impl_->transaction_lane_.reset();
        impl_->transaction_epoch_.reset();
        impl_->transaction_has_source_ = false;
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

    const std::size_t N         = st.prompt_tokens.size();
    const std::uint32_t start_i = st.prompt_tokens_processed;

    if (start_i == N) {
        if (!impl_->pending_round_.valid()) {
            throw std::logic_error("advance_prefill called on completed prompt without pending round");
        }
        // Sample the first output token from logits (step 2: completion)
        impl_->sample_tokens(impl_->pending_round_.logits(), std::span(&lane_idx, 1),
                             std::span(impl_->host_sampled_tokens_.data(), 1));

        impl_->pending_batch_tokens_.resize(1);
        impl_->pending_batch_row_counts_.resize(1);
        impl_->pending_batch_tokens_[0]     = static_cast<TokenId>(impl_->host_sampled_tokens_[0]);
        impl_->pending_batch_row_counts_[0] = 1;

        st.prefill_completed  = true;
        st.committed_frontier = static_cast<std::int32_t>(N);
        st.media_payloads.clear();
        if (impl_->vision_session_.has_value()) {
            impl_->vision_session_->retire_handoff();
        }

        PrefillProgress progress;
        progress.summary.prompt_tokens        = static_cast<std::uint32_t>(N);
        progress.summary.reused_prompt_tokens = st.reused_prompt_tokens;
        progress.summary.prefix_reuse_path =
            st.reused_prompt_tokens > 0
                ? (st.reused_from_turn_closure
                       ? PrefixReusePath::PrivateTurnClosure
                       : PrefixReusePath::PrivateEndpoint)
                : PrefixReusePath::Root;
        progress.processed_prompt_tokens      = 0;
        progress.complete                     = true;
        progress.timing                       = timing.finish();
        progress.pending                      = ContractAccess::make_pending(
            this, 1, std::span(&sequence, 1),
            std::span(impl_->pending_batch_tokens_.data(), 1),
            std::span(impl_->pending_batch_row_counts_.data(), 1), 1, progress.timing);
        return progress;
    }

    std::uint32_t end_i =
        std::min<std::uint32_t>(start_i + impl_->plan_.config.prefill_chunk,
                                static_cast<std::uint32_t>(N));

    // Clip at capture frontier F if not yet offered
    const bool can_offer_capture =
        st.publish_continuation && (impl_->plan_.config.continuation_capacity > 0) &&
        st.capture_frontier.has_value() && !st.capture_offered;

    if (can_offer_capture) {
        const std::uint32_t F = *st.capture_frontier;
        if (start_i < F && end_i > F) {
            end_i = F;
        }
    }

    // Chunk clipping for vision items: a chunk never crosses into a new vision item
    if (st.vision_control.has_value() && !st.vision_control->items.empty()) {
        for (std::size_t k = 0; k < st.vision_control->items.size(); ++k) {
            const auto& item = st.vision_control->items[k];
            if (item.scatter_indices.empty()) { continue; }
            const std::uint32_t item_begin = static_cast<std::uint32_t>(item.scatter_indices.front());
            const std::uint32_t item_end   = static_cast<std::uint32_t>(item.scatter_indices.back() + 1);

            if (item_end <= start_i) {
                continue;
            }
            if (start_i < item_begin && item_begin < end_i) {
                end_i = item_begin;
                break;
            }
            if (item_begin <= start_i && start_i < item_end) {
                end_i = std::min(end_i, item_end);
                break;
            }
        }
    }

    const std::uint32_t chunk_size = end_i - start_i;
    std::span<const std::int32_t> chunk_token_ids(st.prompt_tokens.data() + start_i, chunk_size);
    std::vector<std::array<std::int32_t, 3>> chunk_positions(chunk_size);
    for (std::uint32_t i = 0; i < chunk_size; ++i) {
        chunk_positions[i] = {st.mrope_pos0[start_i + i], st.mrope_pos1[start_i + i],
                              st.mrope_pos2[start_i + i]};
    }

    const Tensor* visual_embeddings = nullptr;
    std::vector<std::int32_t> local_scatter_indices;
    Tensor chunk_visual_tensor;

    if (st.vision_control.has_value() && !st.vision_control->items.empty()) {
        for (std::size_t k = 0; k < st.vision_control->items.size(); ++k) {
            const auto& item = st.vision_control->items[k];
            if (item.scatter_indices.empty()) { continue; }
            const auto& scatter = item.scatter_indices;
            auto begin_it = std::lower_bound(scatter.begin(), scatter.end(), static_cast<std::int32_t>(start_i));
            auto end_it   = std::lower_bound(begin_it, scatter.end(), static_cast<std::int32_t>(end_i));

            if (begin_it < end_it) {
                if (!st.encoded_item_index.has_value() || *st.encoded_item_index != k) {
                    if (!impl_->vision_session_.has_value()) {
                        throw std::logic_error("Vision item in prefill but vision session not initialized");
                    }
                    if (k >= st.media_payloads.size() || !st.media_payloads[k]) {
                        throw std::logic_error("Missing media payload for vision item");
                    }
                    impl_->vision_session_->encode(item, st.media_payloads[k]->span(), impl_->device_.stream);
                    st.encoded_item_index = k;
                }
                const std::size_t visual_begin = begin_it - scatter.begin();
                const std::size_t visual_count = end_it - begin_it;
                local_scatter_indices.resize(visual_count);
                for (std::size_t i = 0; i < visual_count; ++i) {
                    local_scatter_indices[i] = begin_it[i] - static_cast<std::int32_t>(start_i);
                }
                chunk_visual_tensor = impl_->vision_session_->output_tensor().slice(
                    1, static_cast<std::int32_t>(visual_begin), static_cast<std::int32_t>(visual_count));
                visual_embeddings = &chunk_visual_tensor;
                break;
            }
        }
    }

    const bool is_capture_split = can_offer_capture && (end_i == *st.capture_frontier);

    if (end_i < N) {
        auto round = impl_->executor_.execute_prefill_chunk(
            st.lane_handle, chunk_token_ids, chunk_positions, static_cast<std::int32_t>(start_i),
            nullptr, visual_embeddings, local_scatter_indices);
        std::array<detail::LaneCommitDecision, 1> decision = {{{.accept = true}}};
        round.commit(decision);
        st.last_token_id    = chunk_token_ids.back();
        st.last_token_pos   = chunk_positions.back()[0];
        st.last_token_index = static_cast<std::int32_t>(end_i - 1);
        st.prompt_tokens_processed = end_i;

        PrefillProgress progress;
        progress.summary.prompt_tokens        = static_cast<std::uint32_t>(N);
        progress.summary.reused_prompt_tokens = st.reused_prompt_tokens;
        progress.summary.prefix_reuse_path =
            st.reused_prompt_tokens > 0
                ? (st.reused_from_turn_closure
                       ? PrefixReusePath::PrivateTurnClosure
                       : PrefixReusePath::PrivateEndpoint)
                : PrefixReusePath::Root;
        progress.processed_prompt_tokens      = chunk_size;
        progress.complete                     = false;
        progress.timing                       = timing.finish();

        if (is_capture_split) {
            st.capture_offered             = true;
            const std::uint64_t capture_id = ++impl_->capture_counter_;
            st.pending_capture_offer       = capture_id;
            progress.capture               = ContractAccess::make_capture_offer(
                this, sequence.lane(), sequence.epoch(), capture_id);
        }
        return progress;
    }

    // Final prefill chunk ending at N
    if (is_capture_split) {
        // F == N: offer capture at N before sampling
        impl_->pending_round_ = impl_->executor_.execute_prefill_chunk(
            st.lane_handle, chunk_token_ids, chunk_positions, static_cast<std::int32_t>(start_i),
            nullptr, visual_embeddings, local_scatter_indices);
        st.last_token_id    = chunk_token_ids.back();
        st.last_token_pos   = chunk_positions.back()[0];
        st.last_token_index = static_cast<std::int32_t>(end_i - 1);
        st.prompt_tokens_processed = end_i;
        st.capture_offered         = true;

        const std::uint64_t capture_id = ++impl_->capture_counter_;
        st.pending_capture_offer       = capture_id;

        PrefillProgress progress;
        progress.summary.prompt_tokens        = static_cast<std::uint32_t>(N);
        progress.summary.reused_prompt_tokens = st.reused_prompt_tokens;
        progress.summary.prefix_reuse_path =
            st.reused_prompt_tokens > 0
                ? (st.reused_from_turn_closure
                       ? PrefixReusePath::PrivateTurnClosure
                       : PrefixReusePath::PrivateEndpoint)
                : PrefixReusePath::Root;
        progress.processed_prompt_tokens      = chunk_size;
        progress.complete                     = false;
        progress.timing                       = timing.finish();
        progress.capture                      = ContractAccess::make_capture_offer(
            this, sequence.lane(), sequence.epoch(), capture_id);
        return progress;
    }

    // Normal completion at N without capture offer at N
    impl_->pending_round_ = impl_->executor_.execute_prefill_chunk(
        st.lane_handle, chunk_token_ids, chunk_positions, static_cast<std::int32_t>(start_i),
        nullptr, visual_embeddings, local_scatter_indices);
    st.last_token_id    = chunk_token_ids.back();
    st.last_token_pos   = chunk_positions.back()[0];
    st.last_token_index = static_cast<std::int32_t>(end_i - 1);
    st.prompt_tokens_processed = end_i;

    impl_->sample_tokens(impl_->pending_round_.logits(), std::span(&lane_idx, 1),
                         std::span(impl_->host_sampled_tokens_.data(), 1));

    impl_->pending_batch_tokens_.resize(1);
    impl_->pending_batch_row_counts_.resize(1);
    impl_->pending_batch_tokens_[0]     = static_cast<TokenId>(impl_->host_sampled_tokens_[0]);
    impl_->pending_batch_row_counts_[0] = 1;

    st.prefill_completed  = true;
    st.committed_frontier = static_cast<std::int32_t>(N);
    st.media_payloads.clear();
    if (impl_->vision_session_.has_value()) {
        impl_->vision_session_->retire_handoff();
    }

    PrefillProgress progress;
    progress.summary.prompt_tokens        = static_cast<std::uint32_t>(N);
    progress.summary.reused_prompt_tokens = st.reused_prompt_tokens;
    progress.summary.prefix_reuse_path =
        st.reused_prompt_tokens > 0
            ? (st.reused_from_turn_closure
                   ? PrefixReusePath::PrivateTurnClosure
                   : PrefixReusePath::PrivateEndpoint)
            : PrefixReusePath::Root;
    progress.processed_prompt_tokens      = chunk_size;
    progress.complete                     = true;
    progress.timing                       = timing.finish();
    progress.pending                      = ContractAccess::make_pending(
        this, 1, std::span(&sequence, 1),
        std::span(impl_->pending_batch_tokens_.data(), 1),
        std::span(impl_->pending_batch_row_counts_.data(), 1), 1, progress.timing);
    return progress;
}

CaptureAssessment
Program::inspect_capture(const CaptureOffer& offer,
                         const SharedPrefixHandle* /*exact_shared*/,
                         const SharedPrefixHandle* /*replacement*/,
                         std::optional<runtime::CheckpointRef> /*private_replacement*/) const {
    if (impl_ == nullptr || offer.owner() != this) {
        return CaptureAssessment{};
    }
    const std::uint32_t lane_idx = offer.lane().value;
    if (lane_idx >= impl_->plan_.config.max_concurrency) {
        return CaptureAssessment{};
    }
    const auto& st = impl_->lane_states_[lane_idx];
    if (!st.active || st.epoch != offer.epoch()) {
        return CaptureAssessment{};
    }
    if (impl_->plan_.config.continuation_capacity == 0) {
        return CaptureAssessment{};
    }

    CaptureAssessment assessment;
    const std::uint32_t N        = st.prompt_tokens_processed;
    assessment.frontier          = N;
    assessment.publishes_private = st.publish_continuation && (impl_->plan_.config.continuation_capacity > 0);
    assessment.publishes_shared  = false;
    assessment.shortlist_key     = PrefixShortlistKey{
        .digest       = st.prefix_digests.at(N),
        .frontier     = N,
        .identity_tag = 0,
    };
    return assessment;
}

bool Program::shared_capture_matches(const CaptureOffer& /*offer*/,
                                     const SharedPrefixHandle& /*shared*/) const {
    return false;
}

void Program::skip_capture(CaptureOffer&& offer) {
    if (impl_ == nullptr || offer.owner() != this) { return; }
    const std::uint32_t lane_idx = offer.lane().value;
    if (lane_idx < impl_->plan_.config.max_concurrency) {
        impl_->lane_states_[lane_idx].pending_capture_offer = 0;
    }
    ContractAccess::consume(offer);
}

runtime::ContextTransactionReserveStatus
Program::reserve_active_capture(CaptureOffer&& offer,
                                const SharedPrefixHandle* /*exact_shared*/,
                                const SharedPrefixHandle* /*replacement*/,
                                std::optional<runtime::CheckpointRef> /*private_replacement*/,
                                runtime::CancellationFlagView cancellation) {
    if (impl_ == nullptr || offer.owner() != this) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    if (cancellation.requested()) {
        skip_capture(std::move(offer));
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    const std::uint32_t lane_idx = offer.lane().value;
    if (lane_idx >= impl_->plan_.config.max_concurrency) {
        skip_capture(std::move(offer));
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    auto& st = impl_->lane_states_[lane_idx];
    if (!st.active || st.epoch != offer.epoch()) {
        skip_capture(std::move(offer));
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    if (impl_->has_context_transaction_) {
        skip_capture(std::move(offer));
        return runtime::ContextTransactionReserveStatus::Aborted;
    }

    std::int32_t slot_idx = -1;
    for (std::size_t c = 0; c < impl_->continuation_slots_.size(); ++c) {
        if (impl_->continuation_slots_[c].role == detail::ContinuationSlotRole::Vacant) {
            slot_idx = static_cast<std::int32_t>(c);
            break;
        }
    }
    if (slot_idx < 0) {
        std::uint64_t oldest_epoch = std::numeric_limits<std::uint64_t>::max();
        for (std::size_t c = 0; c < impl_->continuation_slots_.size(); ++c) {
            if (impl_->continuation_slots_[c].role == detail::ContinuationSlotRole::Catalogued &&
                impl_->continuation_slots_[c].last_used_epoch < oldest_epoch) {
                oldest_epoch = impl_->continuation_slots_[c].last_used_epoch;
                slot_idx     = static_cast<std::int32_t>(c);
            }
        }
        if (slot_idx >= 0) {
            impl_->executor_.release_physical_groups(
                impl_->continuation_slots_[slot_idx].physical_groups);
            impl_->continuation_slots_[slot_idx].physical_groups.clear();
            impl_->continuation_slots_[slot_idx].generation++;
        }
    }
    if (slot_idx < 0) {
        skip_capture(std::move(offer));
        return runtime::ContextTransactionReserveStatus::Aborted;
    }

    const std::int32_t capture_frontier = static_cast<std::int32_t>(st.prompt_tokens_processed);
    auto& c_slot              = impl_->continuation_slots_[slot_idx];
    c_slot.role               = detail::ContinuationSlotRole::Catalogued;
    c_slot.generation         = ++impl_->continuation_epoch_;
    c_slot.committed_tokens.assign(st.prompt_tokens.begin(),
                                   st.prompt_tokens.begin() + capture_frontier);
    c_slot.committed_frontier = capture_frontier;
    c_slot.prefix_digest      = st.prefix_digests.at(capture_frontier);
    c_slot.last_used_epoch    = ++impl_->continuation_epoch_;
    c_slot.kind               = runtime::CheckpointKind::TurnClosure;
    const auto groups         = impl_->executor_.lane_physical_groups(st.lane_handle);
    c_slot.physical_groups.assign(groups.begin(), groups.end());
    c_slot.history            = impl_->executor_.lane_history(st.lane_handle);
    impl_->executor_.acquire_physical_groups(c_slot.physical_groups);

    const auto active_slot = impl_->allocation_.current_source_slot(lane_idx);
    impl_->executor_.copy_state_slot(static_cast<std::uint32_t>(active_slot), c_slot.cache_slot);

    st.turn_closure_continuation_index = static_cast<std::uint32_t>(slot_idx);
    st.pending_capture_offer           = 0;
    ContractAccess::consume(offer);

    impl_->has_context_transaction_ = true;
    impl_->is_capture_transaction_  = true;
    impl_->transaction_lane_        = runtime::LaneId(lane_idx);
    impl_->transaction_epoch_       = st.epoch;

    ActiveCaptureResult capture_res;
    capture_res.status = runtime::ContextTransactionStatus::Published;
    capture_res.active_summary.rewrite = CheckpointSummary{
        .ref = runtime::CheckpointRef{
            .kind     = runtime::CheckpointKind::TurnClosure,
            .frontier = static_cast<std::uint32_t>(capture_frontier),
            .ordinal  = 0,
        },
        .shortlist_key = PrefixShortlistKey{
            .digest       = c_slot.prefix_digest,
            .frontier     = static_cast<std::uint32_t>(capture_frontier),
            .identity_tag = 0,
        },
        .state_residency = runtime::ReplicaResidency::DeviceOnly,
        .required_kv = TargetKVRequirement{
            .main_frontier    = static_cast<std::uint32_t>(capture_frontier),
            .backend_frontier = 0,
            .main_pages       = static_cast<std::uint32_t>(c_slot.physical_groups.size() * 4),
            .backend_pages    = 0,
        },
        .rebuild_work = runtime::make_prefill_work(
            0, static_cast<std::uint32_t>(capture_frontier), 0, 0,
            impl_->plan_.config.prefill_chunk),
    };
    impl_->pending_capture_result_ = std::move(capture_res);

    return runtime::ContextTransactionReserveStatus::Reserved;
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
            .sampling        = st.sampling_config,
        };
    }

    impl_->pending_round_ = impl_->executor_.execute_round(requests);

    const auto sampled = impl_->pending_round_.sampled_tokens();

    impl_->pending_batch_tokens_.resize(B);
    impl_->pending_batch_row_counts_.resize(B);
    for (std::size_t b = 0; b < B; ++b) {
        impl_->pending_batch_tokens_[b]     = static_cast<TokenId>(sampled[b]);
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
            st.prompt_tokens.push_back(sampled);
            st.prefix_digests.append_generated(std::span(&sampled, 1), 0);
            st.committed_frontier += 1;
            st.last_token_id      = static_cast<std::int32_t>(sampled);
            st.last_token_pos += 1;
            st.last_token_index += 1;
            ++st.total_generated_tokens;

            if (dec.terminal) {
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
        try {
            if (impl_->pending_round_.valid()) {
                impl_->pending_round_.abort();
            }
            const std::uint32_t lane_idx = sequence.lane().value;
            if (lane_idx < impl_->plan_.config.max_concurrency) {
                auto& st = impl_->lane_states_[lane_idx];
                if (st.active && st.epoch == sequence.epoch()) {
                    const bool should_catalogue =
                        st.publish_continuation && (impl_->plan_.config.continuation_capacity > 0) &&
                        (st.committed_frontier > 0);

                if (should_catalogue) {
                    std::uint32_t target_c_idx = impl_->plan_.config.continuation_capacity;
                    for (std::uint32_t c = 0; c < impl_->plan_.config.continuation_capacity; ++c) {
                        if (impl_->continuation_slots_[c].role == detail::ContinuationSlotRole::Vacant) {
                            target_c_idx = c;
                            break;
                        }
                    }
                    if (target_c_idx == impl_->plan_.config.continuation_capacity) {
                        // Evict oldest (LRU)
                        std::uint64_t oldest_epoch = std::numeric_limits<std::uint64_t>::max();
                        for (std::uint32_t c = 0; c < impl_->plan_.config.continuation_capacity; ++c) {
                            if (impl_->continuation_slots_[c].last_used_epoch < oldest_epoch) {
                                oldest_epoch = impl_->continuation_slots_[c].last_used_epoch;
                                target_c_idx = c;
                            }
                        }
                        if (target_c_idx < impl_->plan_.config.continuation_capacity) {
                            impl_->executor_.release_physical_groups(
                                impl_->continuation_slots_[target_c_idx].physical_groups);
                            impl_->continuation_slots_[target_c_idx].physical_groups.clear();
                            impl_->continuation_slots_[target_c_idx].generation++;
                        }
                    }

                    if (target_c_idx < impl_->plan_.config.continuation_capacity) {
                        auto& c_slot = impl_->continuation_slots_[target_c_idx];
                        c_slot.role = detail::ContinuationSlotRole::Catalogued;
                        c_slot.kind = runtime::CheckpointKind::SessionEndpoint;
                        c_slot.last_used_epoch = ++impl_->continuation_epoch_;
                        c_slot.committed_frontier = st.committed_frontier;
                        c_slot.committed_tokens = st.prompt_tokens;
                        c_slot.history = impl_->executor_.lane_history(st.lane_handle);
                        c_slot.physical_groups = impl_->executor_.take_lane_physical_groups(st.lane_handle);

                        const std::int32_t active_slot = impl_->executor_.allocation().current_source_slot(lane_idx);
                        impl_->executor_.copy_state_slot(static_cast<std::uint32_t>(active_slot), c_slot.cache_slot);

                        const std::uint64_t digest = st.prefix_digests.at(st.committed_frontier);
                        c_slot.prefix_digest = digest;

                        FinishResult out;
                        CheckpointSummary cp;
                        cp.ref = runtime::CheckpointRef{
                            .frontier = static_cast<std::uint32_t>(st.committed_frontier),
                            .ordinal  = 0,
                        };
                        cp.shortlist_key = PrefixShortlistKey{
                            .digest       = digest,
                            .frontier     = static_cast<std::uint32_t>(st.committed_frontier),
                            .identity_tag = 0,
                        };
                        cp.state_residency = runtime::ReplicaResidency::DeviceOnly;
                        cp.required_kv = TargetKVRequirement{
                            .main_frontier    = static_cast<std::uint32_t>(st.committed_frontier),
                            .backend_frontier = 0,
                            .main_pages       = static_cast<std::uint32_t>(c_slot.physical_groups.size() * 4),
                            .backend_pages    = 0,
                        };
                        cp.rebuild_work = runtime::make_prefill_work(
                            0, static_cast<std::uint32_t>(st.committed_frontier), 0, 0,
                            impl_->plan_.config.prefill_chunk);

                        out.summary.endpoint          = cp;
                        out.summary.active_references = 0;

                        if (st.turn_closure_continuation_index.has_value()) {
                            const std::uint32_t turn_idx = *st.turn_closure_continuation_index;
                            if (turn_idx < impl_->continuation_slots_.size()) {
                                const auto& turn_slot = impl_->continuation_slots_[turn_idx];
                                if (turn_slot.role == detail::ContinuationSlotRole::Catalogued) {
                                    CheckpointSummary rw;
                                    rw.ref = runtime::CheckpointRef{
                                        .kind     = runtime::CheckpointKind::TurnClosure,
                                        .frontier = static_cast<std::uint32_t>(turn_slot.committed_frontier),
                                        .ordinal  = 0,
                                    };
                                    rw.shortlist_key = PrefixShortlistKey{
                                        .digest       = turn_slot.prefix_digest,
                                        .frontier     = static_cast<std::uint32_t>(turn_slot.committed_frontier),
                                        .identity_tag = 0,
                                    };
                                    rw.state_residency = runtime::ReplicaResidency::DeviceOnly;
                                    rw.required_kv = TargetKVRequirement{
                                        .main_frontier    = static_cast<std::uint32_t>(turn_slot.committed_frontier),
                                        .backend_frontier = 0,
                                        .main_pages       = static_cast<std::uint32_t>(turn_slot.physical_groups.size() * 4),
                                        .backend_pages    = 0,
                                    };
                                    rw.rebuild_work = runtime::make_prefill_work(
                                        0, static_cast<std::uint32_t>(turn_slot.committed_frontier), 0, 0,
                                        impl_->plan_.config.prefill_chunk);
                                    out.summary.rewrite = rw;
                                }
                            }
                        }

                        out.continuation.emplace(ContinuationHandle(
                            this, target_c_idx, c_slot.generation));
                        out.disposition = runtime::FinishDisposition::Catalogued;
                        out.status      = runtime::ConsumeStatus::Consumed;

                        impl_->executor_.release_lane(st.lane_handle);
                        st.active   = false;
                        st.finished = true;
                        ++impl_->resource_revision_;
                        return out;
                    }
                }

                impl_->executor_.release_lane(st.lane_handle);
                st.active   = false;
                st.finished = true;
                ++impl_->resource_revision_;
            }
        }
        } catch (...) {}
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
                if (impl_->pending_round_.valid()) {
                    impl_->pending_round_.abort();
                }
                impl_->executor_.release_lane(st.lane_handle);
                st.active   = false;
                st.finished = true;
                st.pending_capture_offer = 0;
                ++impl_->resource_revision_;
            }
        }
    }
    return AbortResult{
        .status = runtime::ConsumeStatus::Consumed,
    };
}

ReleaseResult Program::release_continuation(ContinuationHandle&& continuation) noexcept {
    ReleaseResult out{ .status = runtime::ConsumeStatus::Consumed };
    if (impl_ == nullptr) { return out; }
    if (continuation.owner() != this) { return out; }

    const std::uint32_t c_idx = continuation.index();
    const std::uint64_t gen   = continuation.generation();

    if (c_idx < impl_->continuation_slots_.size()) {
        auto& c_slot = impl_->continuation_slots_[c_idx];
        if (c_slot.role == detail::ContinuationSlotRole::Catalogued && c_slot.generation == gen) {
            impl_->executor_.release_physical_groups(c_slot.physical_groups);
            c_slot.physical_groups.clear();
            c_slot.committed_tokens.clear();
            c_slot.committed_frontier = 0;
            c_slot.role = detail::ContinuationSlotRole::Vacant;
            c_slot.generation++;
            ++impl_->resource_revision_;
        }
    }
    return out;
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
    for (auto& c_slot : impl_->continuation_slots_) {
        if (c_slot.role == detail::ContinuationSlotRole::Catalogued) {
            impl_->executor_.release_physical_groups(c_slot.physical_groups);
            c_slot.physical_groups.clear();
            c_slot.committed_tokens.clear();
            c_slot.committed_frontier = 0;
            c_slot.role = detail::ContinuationSlotRole::Vacant;
            c_slot.generation++;
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
    std::uint32_t catalogued_slots = 0;
    for (const auto& c : impl_->continuation_slots_) {
        if (c.role == detail::ContinuationSlotRole::Catalogued) {
            ++catalogued_slots;
        }
    }
    return PhysicalUsageSnapshot{
        .resource_revision       = impl_->resource_revision_,
        .device_state_slots      = static_cast<std::uint32_t>(impl_->executor_.active_lanes_count()) + catalogued_slots,
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
    out.cuda_graph_allowance_bytes = impl_->plan_.cuda_graph_allowance_bytes;
    if (impl_->vision_session_.has_value()) {
        out.vision_workspace = impl_->vision_session_->memory_summary(impl_->plan_.config.max_context);
    } else if (impl_->plan_.vision_workspace.has_value()) {
        out.vision_workspace = VisionWorkspaceMemorySummary{
            .aggregate_prompt_tokens = impl_->plan_.config.max_context,
            .max_item_tokens         = impl_->plan_.vision_workspace->max_merged_tokens,
            .general_capacity_bytes  = impl_->plan_.vision_workspace->general_capacity_bytes,
            .encode_peak_bytes       = impl_->plan_.vision_workspace->encode_peak_bytes,
            .handoff_offset_bytes    = impl_->plan_.vision_workspace->handoff_offset_bytes,
            .handoff_capacity_bytes  = impl_->plan_.vision_workspace->handoff_capacity_bytes,
            .handoff_active_bytes    = 0,
            .handoff_peak_bytes      = 0,
        };
    }
    return out;
}

void Program::reset_memory_peaks() noexcept {}

} // namespace ninfer::targets::qwen3_8_flash_next
