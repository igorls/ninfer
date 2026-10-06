#pragma once
#include "models/qwen4_exp/program/internal.h"

#include "core/dtype.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "models/qwen4_exp/state/decoder_state.h"
#include "models/qwen4_exp/program/round_buffers.h"
#include "models/qwen4_exp/program/mtp_frame.h"
#include "models/qwen4_exp/state/state_image.h"
#include "models/load_options.h"
#include "models/qwen3_5/program/planning/startup.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace ninfer::models::qwen4_exp::detail {

using TensorLayout                              = TensorRegion;
inline constexpr std::uint32_t kCausalScoreTile = 1024;
// Log-probability rows one causal-score column reads: its target and the reference tokens.
inline constexpr std::uint32_t kCausalScoreReadoutRows = 1U + kMaximumCausalReferenceTokens;

// Floats one lane's device logprob readout needs for one ordinary round, and for the largest
// prompt-position readout; each column is (sampled, raw) followed by candidate pairs.
inline constexpr std::size_t kLogprobReadoutColumns = kMtpMaximumWidth;
inline constexpr std::size_t kLogprobReadoutFloats =
    (2U + 2U * kMaximumLogprobCandidates) * kLogprobReadoutColumns;
inline constexpr std::size_t kLogprobPromptReadoutFloats =
    (2U + 2U * kMaximumLogprobCandidates) * kMaximumPromptReadouts;

struct PersistentLayout {
    qwen4_exp::DecoderStateLayout decoder;
    qwen4_exp::StateImageDeviceLayout state_images;
    qwen4_exp::RoundStateLayout round;
    std::optional<MtpFrameLayout> mtp;
    std::optional<LayoutRegion> mtp_storage;
    // Mixed final hidden of a scoring chunk; generation reads the chunk tail from the round state.
    std::optional<TensorLayout> prefill_hidden;
    std::optional<TensorLayout> score_hidden;
    std::optional<TensorLayout> token_counts;
    std::optional<TensorLayout> sampling_config;
    // Structured-output token masks [words, draft_window + 1, lanes]: verification column j holds
    // the grammar state after drafts[0..j-1].
    std::optional<TensorLayout> constraint_masks;
    // Prompt-membership bitsets [words, lanes] for the repetition penalty.
    std::optional<TensorLayout> prompt_presence;
    // Token logprob device readout: candidate ids, per-round columns, and prompt positions.
    std::optional<TensorLayout> logprob_candidate_ids;
    std::optional<TensorLayout> logprob_readout;
    std::optional<TensorLayout> logprob_prompt_next_ids;
    std::optional<TensorLayout> logprob_prompt_readout;
    std::size_t bytes            = 0;
    std::size_t kv_payload_bytes = 0;
};

struct WorkspacePlan {
    std::size_t text_prefill     = 0;
    std::size_t ordinary_round   = 0;
    std::size_t causal_score     = 0;
    std::size_t mtp_round        = 0;
    std::size_t general_capacity = 0;
    std::optional<qwen3_5::detail::VisionWorkspacePlan> vision;
    std::size_t capacity         = 0;
};

struct SequencePlanningInputs {
    const execution::Parameters* parameters = nullptr;
    std::uint32_t capacity                  = 0;
    std::uint32_t max_concurrency           = 1;
    std::uint32_t prefill_chunk             = 0;
    std::uint32_t draft_window              = 0;
    SpeculativeBackend speculative_backend  = SpeculativeBackend::None;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    models::LoadOptions features;
    bool use_cuda_graph = true;
    bool causal_scoring = false;
    int device          = 0;
    ContextCacheOptions context_cache;
};

} // namespace ninfer::models::qwen4_exp::detail

namespace ninfer::models::qwen4_exp::detail {

struct SequencePlanImpl {
    const execution::Parameters* parameters = nullptr;
    std::uint32_t capacity                  = 0;
    std::uint32_t kv_capacity               = 0;
    std::uint32_t main_page_groups          = 0;
    std::uint32_t max_concurrency           = 1;
    std::uint32_t prefill_chunk             = 0;
    std::uint32_t draft_window              = 0;
    SpeculativeBackend speculative_backend  = SpeculativeBackend::None;
    KvCacheStorage kv_storage               = KvCacheStorage::BFloat16;
    models::LoadOptions features;
    bool use_cuda_graph = true;
    bool causal_scoring = false;
    int device          = 0;
    ContextCacheOptions context_cache;
    PersistentLayout persistent;
    WorkspacePlan workspace;
    std::size_t graph_allowance_bytes    = 0;
    std::size_t device_reservation_bytes = 0;
};

struct SequencePlannerImpl {
    SequencePlanningInputs inputs;
    runtime::SequenceCapacityCurve curve;
    std::unique_ptr<SequencePlanImpl> minimum;
};

} // namespace ninfer::models::qwen4_exp::detail

namespace ninfer::models::qwen4_exp::detail {


[[nodiscard]] std::unique_ptr<qwen4_exp::detail::SequencePlannerImpl>
make_sequence_planner_impl(const execution::Parameters& parameters, DeviceContext& device,
                           const EngineOptions& options);
[[nodiscard]] std::unique_ptr<SequencePlanImpl>
finalize_sequence_plan_impl(std::unique_ptr<qwen4_exp::detail::SequencePlannerImpl> planner,
                            std::uint32_t main_page_groups);

} // namespace ninfer::models::qwen4_exp::detail
