#pragma once

#include <ninfer/targets/qwen3_8_flash_next/runtime.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/sampling.h"
#include "targets/qwen3_8_flash_next/impl/load/materialized.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"
#include "targets/qwen3_8_flash_next/impl/runtime_state.h"
#include "targets/qwen3_8_flash_next/impl/text_executor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"
#include "targets/qwen3_8_flash_next/impl/vision_execute.h"

namespace ninfer::targets::qwen3_8_flash_next::detail {

class RequestBasePlanImpl {
public:
    runtime::RequestPlanSummary summary;
    qwen3_6::PreparedContextCache context_cache;
    ops::SamplingConfig sampling_config;
    std::uint32_t requested_output_tokens = 0;
    std::uint32_t effective_output_tokens = 0;
    bool allow_prefix_reuse               = false;
    ninfer::ThinkingControlOptions thinking;
    std::optional<qwen3_6::VisionControlPlan> vision_control_plan;
    std::optional<qwen3_6::VisionControl> vision_control;
    qwen3_6::detail::PrefixShortlistDigests prefix_digests;
    std::uint32_t prefix_identity_tag = 0;
};

class AdmissionCandidateImpl {
public:
    runtime::RequestPlanSummary summary;
    runtime::IdentityMaterializationAssessment assessment;
    std::unique_ptr<RequestBasePlanImpl> base_plan;
    std::uint32_t reusable_tokens = 0;
    std::uint32_t source_continuation_index = 0;
    std::uint64_t source_continuation_generation = 0;
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

enum class ContinuationSlotRole : std::uint8_t {
    Vacant,
    Catalogued,
};

struct ContinuationSlot {
    ContinuationSlotRole role = ContinuationSlotRole::Vacant;
    std::uint32_t cache_slot = 0;
    std::uint64_t generation = 1;
    std::vector<TokenId> committed_tokens;
    std::vector<std::uint32_t> physical_groups;
    std::int32_t committed_frontier = 0;
    std::uint64_t prefix_digest = 0;
    PleTokenHistory history{};
    std::uint64_t last_used_epoch = 0;
};

struct LaneState {
    bool active = false;
    std::uint64_t epoch = 0;
    LaneHandle lane_handle{};
    std::vector<TokenId> prompt_tokens;
    std::vector<std::int32_t> mrope_pos0;
    std::vector<std::int32_t> mrope_pos1;
    std::vector<std::int32_t> mrope_pos2;
    std::vector<std::shared_ptr<const qwen3_6::PreparedMediaPayload>> media_payloads;
    std::vector<qwen3_6::VisionItem> vision_items;
    std::optional<qwen3_6::VisionControl> vision_control;
    std::optional<std::size_t> encoded_item_index;
    std::uint32_t prompt_tokens_processed = 0;
    std::uint32_t reused_prompt_tokens    = 0;
    std::int32_t committed_frontier       = 0;
    std::int32_t last_token_id            = 0;
    std::int32_t last_token_pos           = 0;
    std::int32_t last_token_index         = 0;
    std::uint32_t total_generated_tokens  = 0;
    std::uint32_t requested_output_tokens = 0;
    std::uint32_t effective_output_tokens = 0;
    ops::SamplingConfig sampling_config{};
    bool publish_continuation = false;
    bool prefill_completed = false;
    bool finished          = false;
    qwen3_6::detail::PrefixShortlistDigests prefix_digests;
};

class ProgramImpl {
public:
    ProgramImpl(const LoadedModelData* model_data, FlashNextRuntimePlan plan_in, DeviceContext& dev,
                TextModelView text_override = {},
                std::optional<VisionModelView> vision_override = std::nullopt,
                PleIndexMetadata ple_override = kPleIndexMetadata);
    ~ProgramImpl() = default;

    ProgramImpl(const ProgramImpl&)            = delete;
    ProgramImpl& operator=(const ProgramImpl&) = delete;
    ProgramImpl(ProgramImpl&&)                 = delete;
    ProgramImpl& operator=(ProgramImpl&&)      = delete;

    void sample_tokens(const Tensor& logits,
                       std::span<const std::uint32_t> lane_indices,
                       std::span<std::int32_t> out_tokens);

    const LoadedModelData* model_data_ = nullptr;
    TextModelView text_override_{};
    std::optional<VisionModelView> vision_override_;
    FlashNextRuntimePlan plan_;
    DeviceContext& device_;
    FlashNextRuntimeAllocation allocation_;
    FlashNextTextExecutor executor_;
    std::optional<FlashNextVisionSession> vision_session_;

    std::uint64_t resource_revision_ = 1;

    std::vector<LaneState> lane_states_;
    std::vector<ContinuationSlot> continuation_slots_;
    std::uint64_t continuation_epoch_ = 0;

    bool has_context_transaction_ = false;
    std::optional<runtime::LaneId> transaction_lane_;
    std::optional<std::uint64_t> transaction_epoch_;
    bool transaction_has_source_ = false;
    runtime::ClaimDisposition transaction_source_disposition_ = runtime::ClaimDisposition::ConsumedToActive;

    PendingRound pending_round_;
    std::vector<TokenId> pending_batch_tokens_;
    std::vector<std::int32_t> pending_batch_row_counts_;

    WorkspaceArena sampling_workspace_;
    DeviceBuffer device_sampling_configs_;
    DeviceBuffer device_sampling_positions_;
    DeviceBuffer device_sampled_tokens_;
    std::vector<std::int32_t> host_sampled_tokens_;
    std::vector<ops::SamplingConfig> host_sampling_configs_;
    std::vector<std::int32_t> host_sampling_positions_;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
