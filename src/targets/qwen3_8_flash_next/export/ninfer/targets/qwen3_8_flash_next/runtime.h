#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6/runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::targets::qwen3_8_flash_next {

namespace detail {
class RequestBasePlanImpl;
class AdmissionCandidateImpl;
class PressurePlanningSessionImpl;
class ProgramImpl;
class SequencePlanImpl;
class SequencePlannerImpl;
} // namespace detail

using PrefixShortlistKey    = qwen3_6::PrefixShortlistKey;
using TargetKVRequirement   = qwen3_6::TargetKVRequirement;
using CheckpointSummary     = qwen3_6::CheckpointSummary;
using ContinuationSummary   = qwen3_6::ContinuationSummary;
using SharedPrefixSummary   = qwen3_6::SharedPrefixSummary;
using CaptureAssessment     = qwen3_6::CaptureAssessment;
using PressureTargetHandle  = qwen3_6::PressureTargetHandle;
using PhysicalUsageSnapshot = qwen3_6::PhysicalUsageSnapshot;
using CommitRowResult       = qwen3_6::CommitRowResult;

class SequenceHandle {
public:
    SequenceHandle() noexcept                                 = default;
    SequenceHandle(const SequenceHandle&) noexcept            = default;
    SequenceHandle& operator=(const SequenceHandle&) noexcept = default;

    explicit SequenceHandle(const void* owner, runtime::LaneId lane, std::uint64_t epoch) noexcept
        : owner_(owner), lane_(lane), epoch_(epoch) {}

    [[nodiscard]] const void* owner() const noexcept { return owner_; }
    [[nodiscard]] runtime::LaneId lane() const noexcept { return lane_; }
    [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;
};

class ContinuationHandle {
public:
    ContinuationHandle() noexcept = default;
    ~ContinuationHandle()         = default;

    ContinuationHandle(ContinuationHandle&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_),
          generation_(std::exchange(other.generation_, 0)) {}

    ContinuationHandle& operator=(ContinuationHandle&& other) noexcept {
        if (this != &other) {
            owner_      = std::exchange(other.owner_, nullptr);
            index_      = other.index_;
            generation_ = std::exchange(other.generation_, 0);
        }
        return *this;
    }

    ContinuationHandle(const ContinuationHandle&)            = delete;
    ContinuationHandle& operator=(const ContinuationHandle&) = delete;

    explicit ContinuationHandle(const void* owner, std::uint32_t index,
                                std::uint64_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    [[nodiscard]] const void* owner() const noexcept { return owner_; }
    [[nodiscard]] std::uint32_t index() const noexcept { return index_; }
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }

private:
    const void* owner_        = nullptr;
    std::uint32_t index_      = 0;
    std::uint64_t generation_ = 0;
};

class SharedPrefixHandle {
public:
    SharedPrefixHandle() noexcept = default;
    ~SharedPrefixHandle()         = default;

    SharedPrefixHandle(SharedPrefixHandle&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_),
          generation_(std::exchange(other.generation_, 0)) {}

    SharedPrefixHandle& operator=(SharedPrefixHandle&& other) noexcept {
        if (this != &other) {
            owner_      = std::exchange(other.owner_, nullptr);
            index_      = other.index_;
            generation_ = std::exchange(other.generation_, 0);
        }
        return *this;
    }

    SharedPrefixHandle(const SharedPrefixHandle&)            = delete;
    SharedPrefixHandle& operator=(const SharedPrefixHandle&) = delete;

    explicit SharedPrefixHandle(const void* owner, std::uint32_t index,
                                std::uint64_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    [[nodiscard]] const void* owner() const noexcept { return owner_; }
    [[nodiscard]] std::uint32_t index() const noexcept { return index_; }
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }

private:
    const void* owner_        = nullptr;
    std::uint32_t index_      = 0;
    std::uint64_t generation_ = 0;
};

class CaptureOffer {
public:
    CaptureOffer() noexcept = default;
    ~CaptureOffer()         = default;

    CaptureOffer(CaptureOffer&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), lane_(other.lane_), epoch_(other.epoch_),
          id_(std::exchange(other.id_, 0)) {}

    CaptureOffer& operator=(CaptureOffer&& other) noexcept {
        if (this != &other) {
            owner_ = std::exchange(other.owner_, nullptr);
            lane_  = other.lane_;
            epoch_ = other.epoch_;
            id_    = std::exchange(other.id_, 0);
        }
        return *this;
    }

    CaptureOffer(const CaptureOffer&)            = delete;
    CaptureOffer& operator=(const CaptureOffer&) = delete;

    explicit CaptureOffer(const void* owner, runtime::LaneId lane, std::uint64_t epoch,
                          std::uint64_t id) noexcept
        : owner_(owner), lane_(lane), epoch_(epoch), id_(id) {}

    [[nodiscard]] const void* owner() const noexcept { return owner_; }
    [[nodiscard]] runtime::LaneId lane() const noexcept { return lane_; }
    [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }
    friend class ContractAccess;

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;
    std::uint64_t id_    = 0;
};

class PendingBatch {
public:
    PendingBatch() noexcept = default;
    ~PendingBatch()         = default;

    PendingBatch(PendingBatch&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          transaction_(std::exchange(other.transaction_, 0)), rows_(other.rows_),
          row_count_(std::exchange(other.row_count_, 0)), tokens_(other.tokens_),
          row_counts_(other.row_counts_), row_stride_(other.row_stride_), timing_(other.timing_) {
        other.tokens_     = {};
        other.row_counts_ = {};
        other.row_stride_ = 0;
        other.timing_     = {};
    }

    PendingBatch& operator=(PendingBatch&& other) noexcept {
        if (this != &other) {
            owner_       = std::exchange(other.owner_, nullptr);
            transaction_ = std::exchange(other.transaction_, 0);
            rows_        = other.rows_;
            row_count_   = std::exchange(other.row_count_, 0);
            tokens_      = other.tokens_;
            row_counts_  = other.row_counts_;
            row_stride_  = other.row_stride_;
            timing_      = other.timing_;
            other.tokens_     = {};
            other.row_counts_ = {};
            other.row_stride_ = 0;
            other.timing_     = {};
        }
        return *this;
    }

    PendingBatch(const PendingBatch&)            = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }
    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }
    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return row_counts_; }
    [[nodiscard]] std::uint32_t row_stride() const noexcept { return row_stride_; }
    [[nodiscard]] runtime::ExecutionTiming execution_timing() const noexcept { return timing_; }

    friend class ContractAccess;

private:
    const void* owner_         = nullptr;
    std::uint64_t transaction_ = 0;
    std::array<SequenceHandle, kMaximumConcurrency> rows_{};
    std::size_t row_count_ = 0;
    std::span<const TokenId> tokens_;
    std::span<const std::int32_t> row_counts_;
    std::uint32_t row_stride_ = 0;
    runtime::ExecutionTiming timing_;
};

class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const qwen3_6::PreparedContextCache& context_cache() const noexcept;
    [[nodiscard]] std::optional<PrefixShortlistKey>
    prefix_shortlist_key(std::uint32_t frontier) const noexcept;
    [[nodiscard]] std::optional<runtime::PrefillWork>
    shared_candidate_rebuild_work(std::uint32_t frontier) const noexcept;

public:
    explicit RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl> impl) noexcept;
    std::unique_ptr<detail::RequestBasePlanImpl> impl_;
};

class AdmissionCandidate {
public:
    AdmissionCandidate(AdmissionCandidate&&) noexcept;
    AdmissionCandidate& operator=(AdmissionCandidate&&) noexcept;
    ~AdmissionCandidate();

    AdmissionCandidate(const AdmissionCandidate&)            = delete;
    AdmissionCandidate& operator=(const AdmissionCandidate&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const runtime::IdentityMaterializationAssessment&
    identity_assessment() const noexcept;

public:
    explicit AdmissionCandidate(std::unique_ptr<detail::AdmissionCandidateImpl> impl) noexcept;
    std::unique_ptr<detail::AdmissionCandidateImpl> impl_;
};

class ResourcePlan {
public:
    ResourcePlan(ResourcePlan&&) noexcept;
    ResourcePlan& operator=(ResourcePlan&&) noexcept;
    ~ResourcePlan();

    ResourcePlan(const ResourcePlan&)            = delete;
    ResourcePlan& operator=(const ResourcePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] bool needs_transfer() const noexcept { return needs_transfer_; }
    [[nodiscard]] std::uint64_t resource_revision() const noexcept { return revision_; }

public:
    ResourcePlan(AdmissionCandidate&& admission, std::uint64_t revision,
                 bool needs_transfer) noexcept;

    friend class ContractAccess;

private:
    AdmissionCandidate admission_;
    std::uint64_t revision_ = 0;
    bool needs_transfer_    = false;
};

class PersistentBackfillProof {
public:
    PersistentBackfillProof(PersistentBackfillProof&&) noexcept            = default;
    PersistentBackfillProof& operator=(PersistentBackfillProof&&) noexcept = default;

    PersistentBackfillProof(const PersistentBackfillProof&)            = delete;
    PersistentBackfillProof& operator=(const PersistentBackfillProof&) = delete;

    explicit PersistentBackfillProof(std::uint64_t revision) noexcept : revision_(revision) {}

    [[nodiscard]] std::uint64_t resource_revision() const noexcept { return revision_; }

private:
    std::uint64_t revision_ = 0;
};

class PreparedPressureExpansion {
public:
    PreparedPressureExpansion(PreparedPressureExpansion&& other) noexcept
        : session_(std::exchange(other.session_, nullptr)),
          session_generation_(std::exchange(other.session_generation_, 0)),
          scratch_generation_(std::exchange(other.scratch_generation_, 0)),
          parent_index_(other.parent_index_), new_canonical_count_(other.new_canonical_count_) {}

    PreparedPressureExpansion& operator=(PreparedPressureExpansion&&)      = delete;
    PreparedPressureExpansion(const PreparedPressureExpansion&)            = delete;
    PreparedPressureExpansion& operator=(const PreparedPressureExpansion&) = delete;

    [[nodiscard]] std::uint32_t new_canonical_count() const noexcept {
        return new_canonical_count_;
    }

public:
    PreparedPressureExpansion(const void* session, std::uint32_t session_generation,
                              std::uint32_t scratch_generation, std::uint32_t parent_index,
                              std::uint32_t new_canonical_count) noexcept
        : session_(session), session_generation_(session_generation),
          scratch_generation_(scratch_generation), parent_index_(parent_index),
          new_canonical_count_(new_canonical_count) {}

private:
    const void* session_               = nullptr;
    std::uint32_t session_generation_  = 0;
    std::uint32_t scratch_generation_  = 0;
    std::uint32_t parent_index_        = 0;
    std::uint32_t new_canonical_count_ = 0;
};

struct PressureExpansionView {
    std::span<const PressureTargetHandle> children;
    std::uint32_t new_canonical_count = 0;
};

class CapturePressurePlan {
public:
    CapturePressurePlan(CapturePressurePlan&&) noexcept            = default;
    CapturePressurePlan& operator=(CapturePressurePlan&&) noexcept = default;
    ~CapturePressurePlan()                                         = default;

    CapturePressurePlan(const CapturePressurePlan&)            = delete;
    CapturePressurePlan& operator=(const CapturePressurePlan&) = delete;

    [[nodiscard]] std::uint64_t resource_revision() const noexcept { return revision_; }

private:
    CapturePressurePlan(AdmissionCandidate&& pressure, std::uint64_t revision) noexcept
        : pressure_(std::move(pressure)), revision_(revision) {}

    AdmissionCandidate pressure_;
    std::uint64_t revision_ = 0;

    friend class Program;
    friend class PressurePlanningSession;
};

class PressurePlanningSession {
public:
    PressurePlanningSession(PressurePlanningSession&&) noexcept;
    PressurePlanningSession& operator=(PressurePlanningSession&&) noexcept;
    ~PressurePlanningSession();

    PressurePlanningSession(const PressurePlanningSession&)            = delete;
    PressurePlanningSession& operator=(const PressurePlanningSession&) = delete;

    [[nodiscard]] PressureTargetHandle
    identity_target(const AdmissionCandidate& candidate) const;
    [[nodiscard]] PressureTargetHandle
    root_maximal_target(const AdmissionCandidate& root_candidate);
    [[nodiscard]] runtime::PressureTargetAssessment assess(PressureTargetHandle target);
    [[nodiscard]] PreparedPressureExpansion prepare_expansion(PressureTargetHandle parent);
    [[nodiscard]] PressureExpansionView
    commit_expansion(PreparedPressureExpansion&& prepared);
    void discard_expansion(PreparedPressureExpansion&& prepared) noexcept;
    [[nodiscard]] std::optional<ResourcePlan> seal(PressureTargetHandle target,
                                                  const qwen3_6::PreparedPrompt& prompt);
    [[nodiscard]] std::optional<CapturePressurePlan> seal_capture(PressureTargetHandle target) {
        (void)target;
        return std::nullopt;
    }

public:
    explicit PressurePlanningSession(
        std::unique_ptr<detail::PressurePlanningSessionImpl> impl) noexcept;
    std::unique_ptr<detail::PressurePlanningSessionImpl> impl_;
};

struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    runtime::ExecutionTiming timing;
    std::optional<PendingBatch> pending;
    std::optional<CaptureOffer> capture;
};

struct SharedPrefixPublication {
    SharedPrefixHandle handle;
    SharedPrefixSummary summary;
};

struct ActiveCaptureResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    bool capacity_preparation_committed      = false;
    ContinuationSummary active_summary;
    std::optional<SharedPrefixPublication> shared;
    std::vector<qwen3_6::MaterializationVictimResult> victims;
    std::vector<qwen3_6::MaterializationSharedVictimResult> shared_victims;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations;
};

struct StartResult {
    SequenceHandle sequence;
};

struct MaterializationResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    std::optional<StartResult> published;
    std::optional<qwen3_6::MaterializationSourceResult> source;
    std::optional<qwen3_6::MaterializationSharedSourceResult> shared_source;
    std::vector<qwen3_6::MaterializationVictimResult> victims;
    std::vector<qwen3_6::MaterializationSharedVictimResult> shared_victims;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations;
};

using ContextTransactionProgress =
    std::variant<runtime::ContextTransactionInProgress, MaterializationResult, ActiveCaptureResult>;

struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    std::array<std::optional<CaptureOffer>, kMaximumConcurrency> captures{};
    std::size_t row_count = 0;
    runtime::ExecutionTiming timing;
};

struct DiscardResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    std::size_t row_count         = 0;
};

struct FinishResult {
    runtime::ConsumeStatus status          = runtime::ConsumeStatus::InvariantMismatch;
    runtime::FinishDisposition disposition = runtime::FinishDisposition::Released;
    GenerationTimings timings;
    SpeculativeStats speculative;
    ContinuationSummary summary;
    std::optional<ContinuationHandle> continuation;
};

struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct ReleaseResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
};

class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;

public:
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl> impl_;
};

class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] runtime::SequenceCapacityCurve capacity_curve() const;
    [[nodiscard]] SequencePlan finalize(std::uint32_t main_page_groups) &&;

public:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl> impl_;
};

class Program {
public:
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    [[nodiscard]] RequestBasePlan
    plan_request(const qwen3_6::PreparedPrompt& prompt,
                 const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] std::optional<AdmissionCandidate>
    inspect_admission(const qwen3_6::PreparedPrompt& prompt, const RequestBasePlan& base,
                      runtime::LaneId destination, const ContinuationHandle* source,
                      const SharedPrefixHandle* shared_source,
                      std::optional<runtime::CheckpointRef> checkpoint,
                      bool must_retain_private_source,
                      const runtime::ContextMachineCostModel& machine_cost);
    [[nodiscard]] std::optional<ResourcePlan>
    seal_identity(const AdmissionCandidate& candidate, const qwen3_6::PreparedPrompt& prompt);
    [[nodiscard]] PressurePlanningSession
    begin_pressure_planning(const runtime::ContextMachineCostModel& machine_cost,
                            std::span<const AdmissionCandidate* const> candidates,
                            std::span<const ContinuationHandle* const> private_owners,
                            std::span<const std::uint32_t> private_owner_ordinals,
                            std::span<const SharedPrefixHandle* const> shared_owners,
                            std::span<const std::uint32_t> shared_owner_ordinals);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    start_resource_transaction(ResourcePlan&& plan, qwen3_6::PreparedPrompt&& prompt,
                               runtime::CancellationFlagView cancellation);
    [[nodiscard]] std::optional<PersistentBackfillProof>
    prove_persistent_backfill(const RequestBasePlan& blocked_head,
                              const ResourcePlan& candidate,
                              std::span<const SequenceHandle> persistent_borrowers) const;
    [[nodiscard]] ContextTransactionProgress
    progress_context_transaction(runtime::CancellationFlagView cancellation);
    void finalize_context_transaction() noexcept;
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] PrefillProgress
    advance_prefill(SequenceHandle sequence,
                    runtime::ExecutionTiming* failed_timing = nullptr);
    void select_shared_captures(ResourcePlan& /*plan*/, const qwen3_6::PreparedPrompt& /*prompt*/,
                                std::span<const std::uint32_t> /*frontiers*/) {}
    [[nodiscard]] std::uint64_t
    shared_capture_split_cost_ns(const ResourcePlan& /*plan*/, const qwen3_6::PreparedPrompt& /*prompt*/,
                                 std::span<const std::uint32_t> /*frontiers*/,
                                 const runtime::ContextMachineCostModel& /*machine_cost*/) {
        return 0;
    }
    [[nodiscard]] CaptureAssessment
    inspect_capture(const CaptureOffer& offer,
                    const SharedPrefixHandle* exact_shared,
                    const SharedPrefixHandle* replacement,
                    std::optional<runtime::CheckpointRef> private_replacement,
                    bool permit_shared_publication,
                    const runtime::ContextMachineCostModel& machine_cost) const;
    [[nodiscard]] CaptureAssessment
    inspect_capture(const CaptureOffer& offer,
                    const SharedPrefixHandle* exact_shared = nullptr,
                    const SharedPrefixHandle* replacement = nullptr,
                    std::optional<runtime::CheckpointRef> private_replacement = std::nullopt) const;
    [[nodiscard]] std::uint64_t
    checkpoint_recovery_ns(const ContinuationHandle& /*owner*/,
                           runtime::CheckpointRef /*checkpoint*/,
                           const runtime::ContextMachineCostModel& /*machine_cost*/) const {
        return 0;
    }
    [[nodiscard]] std::uint64_t
    checkpoint_recovery_ns(const SharedPrefixHandle& /*owner*/,
                           runtime::CheckpointRef /*checkpoint*/,
                           const runtime::ContextMachineCostModel& /*machine_cost*/) const {
        return 0;
    }
    [[nodiscard]] AdmissionCandidate
    make_capture_pressure_candidate(const CaptureAssessment& assessment,
                                    const runtime::ContextMachineCostModel& machine_cost) const;
    [[nodiscard]] bool shared_capture_matches(const CaptureOffer& offer,
                                              const SharedPrefixHandle& shared) const;
    void skip_capture(CaptureOffer&& offer);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_active_capture(CaptureOffer&& offer,
                           const SharedPrefixHandle* exact_shared,
                           const SharedPrefixHandle* replacement,
                           std::optional<runtime::CheckpointRef> private_replacement,
                           bool permit_shared_publication,
                           const runtime::ContextMachineCostModel& machine_cost,
                           runtime::CancellationFlagView cancellation);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_active_capture(CaptureOffer&& offer,
                           const SharedPrefixHandle* exact_shared,
                           const SharedPrefixHandle* replacement,
                           std::optional<runtime::CheckpointRef> private_replacement,
                           runtime::CancellationFlagView cancellation);
    [[nodiscard]] runtime::ContextTransactionReserveStatus reserve_active_capture_with_pressure(
        CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
        const SharedPrefixHandle* replacement,
        std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
        CapturePressurePlan&& /*pressure*/,
        const runtime::ContextMachineCostModel& machine_cost,
        runtime::CancellationFlagView cancellation) {
        return reserve_active_capture(std::move(offer), exact_shared, replacement,
                                      private_replacement, permit_shared_publication,
                                      machine_cost, cancellation);
    }
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences,
                         std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                         runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] CommitResult
    commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
           runtime::CommitObservation observation  = runtime::CommitObservation::AllRows,
           runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    [[nodiscard]] ReleaseResult
    release_continuation(ContinuationHandle&& continuation) noexcept;
    [[nodiscard]] ReleaseResult
    release_shared_prefix(SharedPrefixHandle&& shared) noexcept;
    void fail_all_cleanup() noexcept;

    [[nodiscard]] bool
    isolated_request_feasible(const RequestBasePlan& base) const noexcept;
    [[nodiscard]] std::uint64_t resource_revision() const noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

public:
    explicit Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl> impl_;
};

class ContractAccess {
public:
    [[nodiscard]] static PendingBatch
    make_pending(const void* owner, std::uint64_t transaction,
                 std::span<const SequenceHandle> rows, std::span<const TokenId> tokens,
                 std::span<const std::int32_t> row_counts, std::uint32_t row_stride,
                 runtime::ExecutionTiming timing) {
        PendingBatch out;
        out.owner_       = owner;
        out.transaction_ = transaction;
        out.row_count_   = rows.size();
        for (std::size_t i = 0; i < rows.size(); ++i) { out.rows_[i] = rows[i]; }
        out.tokens_     = tokens;
        out.row_counts_ = row_counts;
        out.row_stride_ = row_stride;
        out.timing_     = timing;
        return out;
    }

    [[nodiscard]] static const void* owner(const PendingBatch& pending) noexcept {
        return pending.owner_;
    }

    [[nodiscard]] static std::uint64_t transaction(const PendingBatch& pending) noexcept {
        return pending.transaction_;
    }

    [[nodiscard]] static std::span<const SequenceHandle>
    rows(const PendingBatch& pending) noexcept {
        return {pending.rows_.data(), pending.row_count_};
    }

    static void consume(PendingBatch& pending) noexcept {
        pending.owner_       = nullptr;
        pending.transaction_ = 0;
        pending.row_count_   = 0;
        pending.tokens_      = {};
        pending.row_counts_  = {};
        pending.row_stride_  = 0;
        pending.timing_      = {};
    }

    [[nodiscard]] static CaptureOffer make_capture_offer(const void* owner,
                                                         runtime::LaneId lane,
                                                         std::uint64_t epoch,
                                                         std::uint64_t id) noexcept {
        CaptureOffer out;
        out.owner_ = owner;
        out.lane_  = lane;
        out.epoch_ = epoch;
        out.id_    = id;
        return out;
    }

    [[nodiscard]] static const void* owner(const CaptureOffer& offer) noexcept {
        return offer.owner_;
    }

    [[nodiscard]] static runtime::LaneId lane(const CaptureOffer& offer) noexcept {
        return offer.lane_;
    }

    [[nodiscard]] static std::uint64_t epoch(const CaptureOffer& offer) noexcept {
        return offer.epoch_;
    }

    [[nodiscard]] static std::uint64_t id(const CaptureOffer& offer) noexcept {
        return offer.id_;
    }

    static void consume(CaptureOffer& offer) noexcept {
        offer.owner_ = nullptr;
        offer.lane_  = {};
        offer.epoch_ = 0;
        offer.id_    = 0;
    }

    [[nodiscard]] static AdmissionCandidate
    take_admission(ResourcePlan& plan) noexcept {
        return std::move(plan.admission_);
    }

    [[nodiscard]] static const AdmissionCandidate&
    admission(const ResourcePlan& plan) noexcept {
        return plan.admission_;
    }
};

} // namespace ninfer::targets::qwen3_8_flash_next
