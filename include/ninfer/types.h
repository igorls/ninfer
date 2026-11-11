#pragma once

#include <array>
#include "ninfer/deployment.h"
#include "ninfer/observability/snapshot.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer {

using TokenId = std::int32_t;

inline constexpr std::uint32_t kMaximumConcurrency               = 8;
inline constexpr std::size_t kMaximumContextCacheSessionKeyBytes = 256;
inline constexpr std::size_t kMaximumExplicitPromptCacheMarkers  = 4;
// Aggregate encoded image/video payload retained by one prompt, independent of item count.
inline constexpr std::size_t kMaximumPromptMediaBytes = 256ULL << 20;
inline constexpr std::size_t kDefaultMediaCacheBytes  = 1ULL << 30;
inline constexpr std::size_t kDefaultMediaLiveBytes   = 2ULL << 30;


enum class EnginePurpose : std::uint8_t {
    Generation,
    CausalScoring,
};


// Device memory left for the shared desktop and other applications.
inline constexpr std::size_t kDefaultDesktopReserveBytes = 8ULL << 30;
inline constexpr std::size_t kDefaultMemoryHeadroomBytes = 1ULL << 30;

struct ContextCapacityPolicy {
    ContextCapacityMode mode   = ContextCapacityMode::Automatic;
    std::size_t explicit_bytes = 0;

    [[nodiscard]] static constexpr ContextCapacityPolicy
    explicit_capacity(std::size_t bytes) noexcept {
        return ContextCapacityPolicy{ContextCapacityMode::Explicit, bytes};
    }

    [[nodiscard]] static constexpr ContextCapacityPolicy automatic() noexcept { return {}; }
};

enum class ProposalHead : std::uint8_t {
    Full,
    Optimized,
};

struct SpeculativeOptions {
    SpeculativeBackend backend = SpeculativeBackend::None;
    // Startup-fixed K: MTP 1..5; DFlash and DFlash2 1..15 (query width K+1).
    std::uint32_t draft_tokens = 0;
    ProposalHead proposal_head = ProposalHead::Full;
};

enum class StartupPhase : std::uint8_t {
    EngineStartup,
    CudaInitialize,
    ArtifactInspect,
    TargetPlan,
    WeightsMaterialize,
    WeightsStagingPin,
    TargetFinalize,
    FrontendInitialize,
    ProgramInitialize,
    HostContextPin,
    CudaGraphPrepare,
    CudaGraphCalibration,
    CommunicationInitialize,
    CommunicationWarmup,
    RuntimeMemoryCheck,
    RuntimeRelease,
    EngineFinalize,
};

enum class StartupStatus : std::uint8_t {
    Begin,
    Progress,
    Complete,
    Failed,
};

enum class StartupProgressUnit : std::uint8_t {
    None,
    Bytes,
};

struct StartupOrigin {
    std::uint32_t rank = 0, node = 0;
    friend bool operator==(const StartupOrigin&, const StartupOrigin&) = default;
};

struct StartupEvent {
    StartupPhase phase                = StartupPhase::EngineStartup;
    StartupStatus status              = StartupStatus::Begin;
    StartupProgressUnit progress_unit = StartupProgressUnit::None;
    std::uint64_t current             = 0;
    std::uint64_t total               = 0;
    std::uint64_t elapsed_ns          = 0;
    std::optional<StartupOrigin> origin;
    std::uint32_t attempt    = 0;
    std::uint64_t invocation = 0;
};

struct StartupObserver {
    // Startup diagnostics never participate in Engine control flow. Callback exceptions are
    // ignored by the publishing boundary so a logging failure cannot invalidate model startup.
    std::function<void(const StartupEvent& event)> callback;
};

struct ContextCacheOptions {
    // Controls cross-request history reads and writes. Request pause/replay resources remain
    // available when history is disabled.
    bool enabled = true;
    // Shared Host quota for StateImages, KV, pause snapshots and in-flight destinations.
    // RTX 5090 defaults to 8 GiB plus eight model StateImages; DGX Spark resolves this to zero.
    // This does not bound total process RAM.
    // Engine::options() returns the resolved Host capacity after construction.
    std::optional<std::size_t> host_capacity_bytes;
    // Target spacing for optional intermediate prefill checkpoints.
    std::uint32_t checkpoint_interval = 8192;
};

struct ContextCostOptions {
    // Empty selects generic defaults plus any matching values compiled into the binary. A
    // nonempty runtime preset independently overrides its matching machine transfer and
    // artifact-prefill components; absent entries retain the preceding numerical layer.
    std::filesystem::path preset_path;
};

// Diagnostic record of the routed experts one executed forward unit selected. Columns are the
// unit's token columns in execution order; experts holds layer_count x top_k ids per column, in
// [column][layer][rank] order with rank 0 the largest router score. Published only while
// EngineOptions::expert_route_observer is set; observation never changes execution or output.
enum class ExpertRouteKind : std::uint8_t {
    Prefill     = 1,
    Decode      = 2,
    MtpProposal = 3,
    MtpVerify   = 4,
};

struct ExpertRouteColumn {
    std::uint64_t request_tag = 0; // GenerationObservationOptions::route_tag of the request
    std::uint32_t position    = 0; // absolute position of the column's token in its sequence
};

struct ExpertRouteRecord {
    ExpertRouteKind kind      = ExpertRouteKind::Decode;
    std::uint16_t first_layer = 0;
    std::uint16_t layer_count = 0;
    std::uint16_t top_k       = 0;
    std::span<const ExpertRouteColumn> columns;
    std::span<const std::uint16_t> experts;
};

// Routed-layer geometry, reported once before the first record. Text layers have indices
// [0, text_layers); an MTP layer, when present, has index mtp_layer.
struct ExpertRouteGeometry {
    std::uint16_t text_layers = 0;
    std::optional<std::uint16_t> mtp_layer;
    std::uint16_t experts = 0;
    std::uint16_t top_k   = 0;
};

class ExpertRouteObserver {
public:
    virtual ~ExpertRouteObserver()                                         = default;
    virtual void expert_route_started(const ExpertRouteGeometry& geometry) = 0;
    // Called on the Engine worker after each executed unit; it must return quickly.
    virtual void expert_route(const ExpertRouteRecord& record) noexcept = 0;
};

struct MoeOptions {
    // Descending cache priority: a permutation of all logical expert IDs per Main layer.
    // Dense layers have empty rows. nullopt uses the model's calibrated default when available,
    // otherwise ID order. Capacity determines each prefix.
    std::optional<std::vector<std::vector<std::uint32_t>>> expert_priority;
    // Zero selects physical cores while reserving a control-thread core when available.
    std::uint32_t cpu_workers = 0;
};

struct ObservabilityOptions {
    bool gpu_timing = false;
};

// Largest static YaRN RoPE factor the Engine accepts: 4 x the native 262,144 positions of the
// Qwen3.5-family checkpoints is the documented 1M-token extension, and the qualified range.
inline constexpr float kMaximumYarnFactor = 4.0F;

// Position capacity of a model whose native capacity is `native` under YaRN `factor`.
[[nodiscard]] inline std::uint32_t yarn_position_capacity(std::uint32_t native,
                                                          float factor) noexcept {
    return static_cast<std::uint32_t>(std::floor(static_cast<double>(native) * factor));
}

struct EngineOptions {
    ObservabilityOptions observability;
    std::optional<DeploymentOptions> deployment;
    std::filesystem::path artifact_path;
    std::filesystem::path chat_template_path;
    std::size_t grammar_cache_bytes = 256ULL * 1024 * 1024;
    EnginePurpose purpose           = EnginePurpose::Generation;
    int device                      = 0;
    std::uint32_t max_context       = 8192; // Logical ceiling of one request or score window.
    // Opt-in static YaRN RoPE scaling. Unset keeps the artifact's native RoPE and position
    // capacity. A factor in (1, kMaximumYarnFactor] applies the model's YaRN frequencies at every
    // position, short prompts included, and raises the max_context ceiling to
    // yarn_position_capacity(native, factor). Models without YaRN support reject it.
    std::optional<float> yarn_factor;
    ContextCapacityPolicy context_capacity;
    // Per-rank planning allowance that startup allocations may consume. Zero adds no allowance.
    std::size_t memory_headroom_bytes  = kDefaultMemoryHeadroomBytes;
    std::uint32_t max_concurrency      = 1;
    // Reduce startup-fixed lanes until their complete context working sets fit.
    bool clamp_concurrency_to_pool = false;
    // Admission uses device-wide free memory minus this desktop reserve.
    std::size_t desktop_reserve_bytes = kDefaultDesktopReserveBytes;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    // Zero uses 4096 for Qwen4Exp or CPU expert offload, otherwise 1024.
    std::uint32_t prefill_chunk = 0;
    MoeOptions moe;
    KvCacheStorage kv_cache = KvCacheStorage::BFloat16;
    SpeculativeOptions speculative;
    std::size_t media_cache_bytes = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes  = kDefaultMediaLiveBytes;
    // Zero selects a bounded worker count from the detected host concurrency.
    std::uint32_t media_preprocess_threads = 0;
    bool enable_vision                     = false;
    std::size_t vision_memory_budget_bytes = 1ULL << 30;
    bool use_cuda_graph                    = true;
    ContextCacheOptions context_cache;
    ContextCostOptions context_cost;
    StartupObserver startup_observer;
    // Called once from the worker after an Engine-wide failure has failed every request. A
    // supervised server exits here so its supervisor restarts it; unset, the Engine stays
    // unavailable (Engine::is_available() is false).
    std::function<void(std::exception_ptr, const std::string&)> on_fatal_error;
    // Routed-expert observation. Startup reserves the trace buffers only when it is set, and a
    // model without routed experts rejects it.
    std::shared_ptr<ExpertRouteObserver> expert_route_observer;
};

// Immutable model-owned values used when a request does not override a sampling field. Seed is
// deliberately excluded: it is an execution choice rather than a model recommendation.
struct SamplingPreset {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    // HF-style: prompt or generated tokens have positive logits divided and negative logits
    // multiplied by this value. 1 is neutral.
    float repetition_penalty = 1.0F;
};

struct ModelSamplingDefaults {
    SamplingPreset thinking;
    SamplingPreset non_thinking;
};

// Public request-side overrides. std::nullopt means "use the registered model/mode default";
// explicit zero remains a real override (including temperature=0 for exact argmax).
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<float> repetition_penalty;
    std::optional<std::uint64_t> seed;
};

// Complete parameters after Engine resolution. Target runtimes consume only this type.
struct ResolvedSamplingParameters {
    float temperature        = 0.0F;
    std::int32_t top_k       = 20;
    float top_p              = 1.0F;
    float min_p              = 0.0F;
    float presence_penalty   = 0.0F;
    float frequency_penalty  = 0.0F;
    float repetition_penalty = 1.0F;
    std::uint64_t seed       = 0;
};

enum class OutputChannel : std::uint8_t {
    Content,
    Reasoning,
};

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;
    bool include_in_output = false;
};

struct StopPolicy {
    std::vector<TokenId> token_ids;
    std::vector<StopString> strings;
    bool include_model_defaults = true;
    bool publish_stop_token     = false;
};

struct ThinkingControlOptions {
    // Positive maximum accepted model-origin tokens while the model thinking phase remains open.
    // Omitted means unlimited. Injected target-control tokens consume the total output budget but
    // not this model-origin budget.
    std::optional<std::uint32_t> budget;
};

// Opt-in per-token log probabilities. They are read from the target model's logits before any
// sampling adjustment (no penalty, temperature or truncation), so they are the model's own
// distribution at every position regardless of how the token was drawn.
struct TokenLogprobOptions {
    bool enabled = false;
    // Number of most likely alternatives reported per position, at most kMaximumTopLogprobs.
    std::uint32_t top = 0;
    // If nonempty, every position also reports the distribution renormalised over exactly these
    // tokens, in this order. Not bounded by `top`.
    std::vector<TokenId> candidates;
    // Prompt positions whose next-token distribution is read during prefill, as ascending
    // 0-based indices into the prepared prompt's tokens; the readout at position p is the
    // distribution over token p+1, and its `sampled` entry is the prompt's own token p+1. Needs
    // `top` 0. Every listed position is computed by this request: prefix reuse is limited to
    // frontiers at or below the first position.
    std::vector<std::uint32_t> prompt_positions;
};

inline constexpr std::uint32_t kMaximumTopLogprobs     = 20;
inline constexpr std::size_t kMaximumLogprobCandidates = 1024;
inline constexpr std::size_t kMaximumPromptReadouts    = 256;

// Optional distribution readout of causal scoring, used to compare an artifact's next-token
// distributions with a reference model's on the same tokens.
inline constexpr std::uint32_t kMaximumCausalReferenceTokens = 32;

struct CausalScoreReadout {
    // Reference tokens per scored target, at most kMaximumCausalReferenceTokens; 0 disables.
    std::uint32_t reference_count = 0;
    // reference_count token ids for every scored target, target-major.
    std::vector<TokenId> reference_tokens;
    // Training readout: return the final-normalized hidden row of every scored predictor
    // position, the row the output head reads, as BF16 words. Row i belongs to prompt position
    // first_target - 1 + i; the last prompt token is never a predictor, so a caller that needs
    // every position appends one token. Scores are unchanged by the capture.
    bool capture_hidden_rows = false;
};

struct CausalScores {
    // log p(tokens[i] | tokens[0..i)) for every scored target i, over the whole vocabulary.
    std::vector<float> target_logprobs;
    // Log probability of each requested reference token, target-major, reference_count per target.
    std::vector<float> reference_logprobs;
    // With a readout: the most likely token of every scored target (lowest id on a tie) and its
    // log probability.
    std::vector<TokenId> argmax_tokens;
    std::vector<float> argmax_logprobs;
    // With capture_hidden_rows: scored-target-major BF16 words, hidden_size per row; otherwise
    // empty and hidden_size is 0.
    std::vector<std::uint16_t> hidden_rows;
    std::uint32_t hidden_size = 0;
};

struct TokenLogprob {
    TokenId token = 0;
    // Log probability over the tokens the structured-output mask allows at this position; equal
    // to raw_logprob when the position is unconstrained, -inf for a forbidden token. Within
    // TokenLogprobs::candidates it is renormalised over the candidate list instead.
    float logprob = 0.0F;
    // Log probability over the whole vocabulary, ignoring any mask.
    float raw_logprob = 0.0F;
};

struct TokenLogprobs {
    // True for a token the engine inserted itself (thinking-budget control tokens). It was not
    // drawn from the model, so it carries logprob 0 and no alternatives.
    bool forced = false;
    TokenLogprob sampled;
    std::vector<TokenLogprob> top;
    std::vector<TokenLogprob> candidates;
};

struct ExecutionOptions {
    SamplingOverrides sampling;
    TokenLogprobOptions logprobs;
    // Training readout: the final-normalized hidden state at the last prompt position before the
    // new assistant turn's content (the thinking opener). Requires a text new-assistant prompt
    // and a positive output budget; disables prefix reuse and publication for the request.
    bool capture_reasoning_features       = false;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    // With allow_prefix_reuse, false makes the request read-only in the context cache: it may
    // start from a published prefix but captures no checkpoint and publishes no continuation,
    // so it cannot displace another conversation's cached state. For one-shot requests.
    bool allow_prefix_publication = true;
    ThinkingControlOptions thinking;
};

struct OutputOptions {
    bool raw                     = false;
    bool preserve_special_tokens = false;
    // Presentation constraint supplied by the protocol adapter. It bounds the model's emitted
    // function-name grammar; it does not require the name to match a currently declared tool.
    std::uint32_t tool_name_max_length = 128;
};

enum class OutputConstraintKind : std::uint8_t { Grammar, JsonObject, JsonSchema, Choice, Regex };

// Constrains generated content; Chat reasoning retains the model's framing. Source is owning
// GBNF, JSON Schema or regex text. Choice owns literal alternatives; JsonObject has no payload.
struct OutputConstraint {
    OutputConstraintKind kind = OutputConstraintKind::Grammar;
    std::string source;
    std::vector<std::string> choices;

    [[nodiscard]] static OutputConstraint grammar(std::string source) {
        return {OutputConstraintKind::Grammar, std::move(source), {}};
    }

    [[nodiscard]] static OutputConstraint json_object() {
        return {OutputConstraintKind::JsonObject, {}, {}};
    }

    [[nodiscard]] static OutputConstraint json_schema(std::string source) {
        return {OutputConstraintKind::JsonSchema, std::move(source), {}};
    }

    [[nodiscard]] static OutputConstraint choice(std::vector<std::string> values) {
        return {OutputConstraintKind::Choice, {}, std::move(values)};
    }

    [[nodiscard]] static OutputConstraint regex(std::string pattern) {
        return {OutputConstraintKind::Regex, std::move(pattern), {}};
    }

    bool operator==(const OutputConstraint&) const = default;
};

enum class ToolChoiceMode : std::uint8_t { Auto, None, Required };
// Basic protects every tool call. Automatic applies constraints only for strict tools or an
// explicit selection/count policy.
enum class ToolConstraintMode : std::uint8_t { Automatic, Basic };

// Declarations belong to PromptOptions. Selection and cardinality affect this generation only.
struct ToolChoice {
    ToolChoiceMode mode = ToolChoiceMode::Auto;
    std::optional<std::vector<std::string>> allowed_names;
    bool parallel                  = true;
    ToolConstraintMode constraints = ToolConstraintMode::Basic;
};

struct RequestOptions {
    std::optional<OutputConstraint> constraint;
    ToolChoice tool_choice;
    ExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

enum class MediaKind : std::uint8_t {
    Image,
    Video,
};

enum class ImageResizePolicy : std::uint8_t {
    Downsize,
    RejectOversized,
};

struct OwnedMedia {
    MediaKind kind = MediaKind::Image;
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
    ImageResizePolicy image_resize_policy = ImageResizePolicy::Downsize;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

enum class ToolCallStatus : std::uint8_t { InProgress, Complete, Incomplete };

// Model-origin structured output. Incomplete arguments are an unmodified JSON prefix, not an
// executable call. Protocol adapters own wire identifiers and their incomplete-result projection.
struct GeneratedToolCall {
    std::string name;
    std::string arguments_json;
    ToolCallStatus status = ToolCallStatus::Complete;

    bool operator==(const GeneratedToolCall&) const = default;
};

// Free generation can return an unrecognized tool header as text. Once a declared function has
// started, malformed output is a request error; previously published arguments are never retracted.
enum class ToolCallParseFallbackReason : std::uint8_t {
    None,
    MalformedStructure,
    InvalidToolName,
    UndeclaredTool,
};

[[nodiscard]] inline constexpr const char*
tool_call_parse_fallback_reason_name(ToolCallParseFallbackReason reason) noexcept {
    switch (reason) {
    case ToolCallParseFallbackReason::None:
        return "none";
    case ToolCallParseFallbackReason::MalformedStructure:
        return "malformed_structure";
    case ToolCallParseFallbackReason::InvalidToolName:
        return "invalid_tool_name";
    case ToolCallParseFallbackReason::UndeclaredTool:
        return "undeclared_tool";
    }
    return "malformed_structure";
}

struct ToolCallParseDiagnostics {
    bool marker_seen                            = false;
    std::uint32_t structured_call_count         = 0;
    std::uint32_t schema_mismatch_arguments     = 0;
    ToolCallParseFallbackReason fallback_reason = ToolCallParseFallbackReason::None;

    [[nodiscard]] friend constexpr bool
    operator==(const ToolCallParseDiagnostics&, const ToolCallParseDiagnostics&) noexcept = default;
};

// Wire-independent conversation authority. Protocol adapters preserve these roles and their
// ordering; a target frontend owns any model-specific role lowering.
enum class ChatRole : std::uint8_t {
    System,
    Developer,
    User,
    Assistant,
    Tool,
};

enum class MessagePartKind : std::uint8_t {
    Text,
    Media,
};

struct MessagePart {
    MessagePartKind kind = MessagePartKind::Text;
    std::string text;
    OwnedMedia media;
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<MessagePart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
    bool tool_result_is_error = false;
};

enum class ReasoningEffort : std::uint8_t {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
};

[[nodiscard]] constexpr std::string_view reasoning_effort_name(ReasoningEffort effort) noexcept {
    switch (effort) {
    case ReasoningEffort::None:
        return "none";
    case ReasoningEffort::Minimal:
        return "minimal";
    case ReasoningEffort::Low:
        return "low";
    case ReasoningEffort::Medium:
        return "medium";
    case ReasoningEffort::High:
        return "high";
    case ReasoningEffort::XHigh:
        return "xhigh";
    case ReasoningEffort::Max:
        return "max";
    }
    return {};
}

// What the loaded chat template does with a reasoning effort, derived once at load from the
// template's own rendering. None is never listed: it disables thinking on every template.
struct ReasoningEffortCapabilities {
    // Efforts the template accepts, in ReasoningEffort order. Empty when the template rejects
    // every effort or renders them all like an unspecified one.
    std::vector<ReasoningEffort> supported;
    // The supported effort whose rendering equals an unspecified effort's, when exactly one does.
    std::optional<ReasoningEffort> default_effort;

    [[nodiscard]] bool supports(ReasoningEffort effort) const noexcept {
        for (const ReasoningEffort value : supported) {
            if (value == effort) { return true; }
        }
        return false;
    }
};

struct PromptCapabilities {
    ReasoningEffortCapabilities reasoning_effort;
};

enum class PromptContinuationMode : std::uint8_t {
    NewAssistantTurn,
    ContinueFinalAssistant,
};

struct PromptDefaults {
    std::optional<bool> enable_thinking;
    std::optional<bool> preserve_thinking;
};

struct PromptOptions {
    PromptContinuationMode continuation = PromptContinuationMode::NewAssistantTurn;
    std::optional<bool> enable_thinking;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    // JSON object of template parameters. Unset typed fields leave template defaults intact.
    std::string chat_template_kwargs_json;
    bool add_vision_id = false;
    std::vector<std::string> tool_jsons;
    PromptDefaults defaults;
};

enum class PromptCacheMarkerKind : std::uint8_t {
    SharedStablePrefix,
    PrivateLongAnchor,
};

enum class SharedCandidateEvidence : std::uint8_t {
    None               = 0,
    ExplicitBoundary   = 1U << 0U,
    RequestedAutomatic = 1U << 1U,
    DefaultAutomatic   = 1U << 2U,
    EngineStructural   = 1U << 3U,
};

[[nodiscard]] constexpr SharedCandidateEvidence operator|(SharedCandidateEvidence left,
                                                          SharedCandidateEvidence right) noexcept {
    return static_cast<SharedCandidateEvidence>(static_cast<std::uint8_t>(left) |
                                                static_cast<std::uint8_t>(right));
}

constexpr SharedCandidateEvidence& operator|=(SharedCandidateEvidence& left,
                                              SharedCandidateEvidence right) noexcept {
    left = left | right;
    return left;
}

[[nodiscard]] constexpr bool has_shared_candidate_evidence(SharedCandidateEvidence value,
                                                           SharedCandidateEvidence evidence) {
    return (static_cast<std::uint8_t>(value) & static_cast<std::uint8_t>(evidence)) != 0;
}

enum class PromptCacheMarkerLocation : std::uint8_t {
    MessageBoundary,
    MessagePartBoundary,
    MessageContentBoundary,
    LeadingInstructionBoundary,
    ToolBoundary,
};

struct PromptCacheMarker {
    std::uint32_t after_message_count  = 0;
    PromptCacheMarkerKind kind         = PromptCacheMarkerKind::SharedStablePrefix;
    SharedCandidateEvidence evidence   = SharedCandidateEvidence::ExplicitBoundary;
    PromptCacheMarkerLocation location = PromptCacheMarkerLocation::MessageBoundary;
    // Byte count within the untrimmed leading System/Developer message.
    std::uint32_t leading_instruction_bytes = 0;
    std::uint32_t after_tool_count          = 0;
    // For MessagePartBoundary, after_message_count identifies the containing message using a
    // one-based count and this value identifies the number of serialized parts within it.
    std::uint32_t after_message_part_count = 0;
    // MessageContentBoundary addresses the complete semantic prefix of an assistant message.
    std::uint32_t after_tool_call_count = 0;
    std::uint32_t reasoning_bytes       = 0;

    [[nodiscard]] friend constexpr bool operator==(PromptCacheMarker,
                                                   PromptCacheMarker) noexcept = default;
};

struct ContextCacheHints {
    std::optional<std::string> session_key;
    std::vector<PromptCacheMarker> markers;
    // Protocols with their own automatic/explicit write policy disable the Engine's structural
    // candidates. Exact reads from already-published shared prefixes remain enabled.
    bool allow_engine_automatic_shared_prefixes = true;
    // Advance the named session lineage when session_key is present. This does not require an
    // anonymous content-matched source to be retained.
    bool update_session_index = true;
};

struct PromptInput {
    std::vector<ChatMessage> messages;
    PromptOptions options;
    ContextCacheHints context_cache;
};

enum class RequestErrorKind : std::uint8_t {
    ConflictingPromptOptions,
    UnsupportedPromptContinuation,
    UnsupportedOutputCombination,
    InvalidToolConstraint,
    InvalidToolOutput,
    InvalidGrammar,
    InvalidChoice,
    InvalidRegex,
    InvalidJsonSchema,
    UnsupportedJsonSchema,
    UnsatisfiableJsonSchema,
    ConstraintDeadEnd,
    ContextLengthExceeded,
    ThinkingBudgetCapacityInsufficient,
    MediaBudgetExceeded,
    InvalidMedia,
    Overloaded,
    QueueTimeout,
    Cancelled,
    Unavailable,
};

enum class RequestErrorSource : std::uint8_t { OutputConstraint, Tools };

class RequestError final : public std::invalid_argument {
public:
    RequestError(RequestErrorKind kind, std::string message, std::string pointer = {},
                 RequestErrorSource source = RequestErrorSource::OutputConstraint)
        : std::invalid_argument(std::move(message)), kind_(kind), pointer_(std::move(pointer)),
          source_(source) {}

    [[nodiscard]] RequestErrorKind kind() const noexcept { return kind_; }

    [[nodiscard]] RequestErrorSource source() const noexcept { return source_; }

    [[nodiscard]] const std::string& pointer() const noexcept { return pointer_; }

private:
    RequestErrorKind kind_;
    std::string pointer_;
    RequestErrorSource source_;
};

struct PromptSummary {
    bool starts_in_reasoning    = false;
    std::uint32_t prompt_tokens = 0;
    bool has_media              = false;
    // Token frontier right after the new assistant turn's header, before its content; absent for
    // raw-token prompts and assistant continuation.
    std::optional<std::uint32_t> reasoning_frontier;
    // First token of the supplied final assistant content, after any template-inserted prefix.
    // Absent outside assistant continuation or when that byte boundary splits a token.
    std::optional<std::uint32_t> continuation_frontier;
};

struct PromptPreparationStats {
    double seconds                       = 0.0;
    double media_preprocess_seconds      = 0.0;
    double media_preprocess_work_seconds = 0.0;
    double tokenize_seconds              = 0.0;
    std::size_t media_items              = 0;
    std::size_t media_bytes              = 0;
    std::uint64_t raw_patches            = 0;
    std::uint64_t vision_tokens          = 0;
    std::size_t patch_bytes              = 0;
    std::size_t media_cache_hits         = 0;
    std::size_t media_cache_misses       = 0;
    std::size_t media_singleflight_waits = 0;
    std::size_t built_patch_bytes        = 0;
    std::size_t reused_patch_bytes       = 0;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
};

struct OutputDelta {
    OutputChannel channel = OutputChannel::Content;
    std::string text;
};

struct ToolCallStart {
    std::uint32_t index = 0;
    std::string name;
};

struct ToolArgumentsDelta {
    std::uint32_t index = 0;
    std::string text;
    // A complete parameter value has been decoded. Protocols requesting buffered parameters can
    // flush here without parsing or interpreting the model's private framing themselves.
    bool parameter_complete = false;
};

struct ToolCallEnd {
    std::uint32_t index   = 0;
    ToolCallStatus status = ToolCallStatus::Complete;
};

using OutputEvent = std::variant<OutputDelta, ToolCallStart, ToolArgumentsDelta, ToolCallEnd>;

// Exact prompt accounting selected at admission. Streaming consumers receive this once before any
// OutputEvent, after the prefix choice and materialization reservation are committed and before
// transfer/prefill execution.
struct GenerationStart {
    PromptSummary prompt;
    std::uint32_t reused_prompt_tokens = 0;
};

// Cumulative prompt frontier published only after the corresponding Program work has completed.
// The Engine owns the request-relative clock and accounting; protocol adapters choose names and
// units for the wire representation.
struct PromptProgress {
    std::uint32_t total_prompt_tokens     = 0;
    std::uint32_t reused_prompt_tokens    = 0;
    std::uint32_t processed_prompt_tokens = 0;
    std::uint64_t elapsed_ns              = 0;
};

// Cumulative timing snapshot at one stable output-commit boundary. Generated tokens count model
// and Engine-injected tokens accepted into the sequence, independently of whether the Frontend has
// enough visible bytes to publish an OutputEvent for that boundary.
struct GenerationTimingObservation {
    std::uint32_t generated_tokens      = 0;
    std::uint64_t prompt_elapsed_ns     = 0;
    std::uint64_t generation_elapsed_ns = 0;
};

class OutputSink {
public:
    virtual ~OutputSink()                                   = default;
    virtual void start(GenerationStart start)               = 0;
    virtual void progress(PromptProgress progress)          = 0;
    virtual void timing(GenerationTimingObservation timing) = 0;
    virtual void publish(OutputEvent event)                 = 0;
};

enum class OutputConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

enum class GenerationSchedulingTransition : std::uint8_t {
    PauseStarted,
    Paused,
    RestoreStarted,
    Restored,
    ReplayComplete,
    RecoveryComplete,
    SnapshotRevoked,
    Terminal,
};

// Sparse lifecycle observations captured by the Engine worker and delivered by wait() on the
// consumer thread. Counter deltas between boundaries, less the request's own deltas, measure
// other requests' completed work during recovery. Decode includes forced control tokens.
struct GenerationSchedulingObservation {
    GenerationSchedulingTransition transition = GenerationSchedulingTransition::PauseStarted;
    GenerationRecoveryRoute route             = GenerationRecoveryRoute::None;
    RequestPhase phase                        = RequestPhase::Waiting;
    std::uint64_t engine_request_id           = 0;
    std::uint64_t steady_ns                   = 0;
    std::uint64_t elapsed_ns                  = 0;
    std::uint64_t preemption_index            = 0;
    std::uint64_t global_prefill_tokens       = 0;
    std::uint64_t global_decode_tokens        = 0;
    std::uint64_t global_replayed_tokens      = 0;
    std::uint64_t request_prefill_tokens      = 0;
    std::uint64_t request_decode_tokens       = 0;
    std::uint64_t request_replayed_tokens     = 0;
};

using GenerationSchedulingObserver = std::function<void(const GenerationSchedulingObservation&)>;

// Emitted once on the first accepted model/control token, delivered by wait() in either consumer
// mode. Submission elapsed includes queueing and binding; preparation is reported separately.
struct GenerationFirstTokenObservation {
    double prepare_seconds              = 0.0;
    double elapsed_since_submit_seconds = 0.0;
    double queue_wait_seconds           = 0.0;
};

using GenerationFirstTokenObserver = std::function<void(const GenerationFirstTokenObservation&)>;

// Observation affects only request publication. It never changes model execution, output
// semantics, scheduling, or cache selection. Live output timings and prompt progress require a
// Streaming consumer. Request wall/execution statistics are always collected.
struct GenerationObservationOptions {
    bool constraint_timings = false;
    bool live_timings       = false;
    bool prompt_progress    = false;
    // Optional for either consumer mode. No scheduling events are retained when unset.
    GenerationSchedulingObserver scheduling;
    GenerationFirstTokenObserver first_token;
    // Opaque tag copied into this request's ExpertRouteColumn::request_tag.
    std::uint64_t route_tag = 0;
};

class CancellationView {
public:
    CancellationView() = default;
    explicit CancellationView(std::function<bool()> requested);

    [[nodiscard]] bool requested() const;

private:
    std::function<bool()> requested_;
};

// Deadline and cancellation apply to all host-side prompt preparation work. Empty values mean
// unbounded preparation.
struct PreparationControl {
    std::chrono::steady_clock::time_point deadline;
    CancellationView cancellation;
};

struct RequestStats {
    // Unique within this Engine instance; diagnostic correlation only.
    std::uint64_t engine_request_id    = 0;
    std::uint32_t reused_prompt_tokens = 0;
    // Completed initial prefill, excluding cached tokens and separately counted replay work.
    std::uint32_t computed_prefill_tokens = 0;
    PrefixReusePath prefix_reuse_path     = PrefixReusePath::Root;
    GenerationTimings timings;
    RequestExecutionStatistics executions;
    // Entry B-1 records this request's full step exposure at actual decode batch B.
    // Summing a complete cohort duplicates each such step B times.
    std::vector<ExecutionExposure> decode_by_batch;
    GpuIntervalCounters max_rank_gpu{};
    GenerationEngineTiming engine_timing;
    GenerationSchedulingStats scheduling;
    GenerationAdmissionStats admission;
    std::optional<GenerationFirstOutputTiming> first_output_timing;
    SpeculativeStats speculative;
    ThinkingBudgetStats thinking;
    std::optional<ConstraintObservation> constraint;
};

struct GenerationResult {
    PromptSummary prompt;
    std::vector<TokenId> generated_token_ids;
    // Present only when the corresponding request readout was selected.
    std::vector<TokenLogprobs> token_logprobs;
    std::vector<TokenLogprobs> prompt_logprobs;
    // FP32 expansion of the BF16 reasoning-frontier hidden row.
    std::vector<float> reasoning_features;
    std::string content;
    std::string reasoning;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics tool_call_parse;
    std::uint32_t reasoning_tokens = 0;
    FinishReason finish_reason     = FinishReason::None;
    std::optional<std::string> matched_stop_string;
    RequestStats stats;
};

} // namespace ninfer
