#pragma once

// Product-side adapter from one protocol-neutral generation request to the public Engine. Wire
// adapters normalize before this layer and render IDs, usage, and response events after it.

#include "ninfer/engine.h"
#include "serve/expert_route_recorder.h"
#include "product/config/options.h"
#include "serve/request.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

struct RequestLifetime;
struct RequestCapacity;

struct TokenLogprobEntry {
    ninfer::TokenId token_id = 0;
    std::string bytes;
    float logprob     = 0.0F;
    float raw_logprob = 0.0F;
};

struct TokenLogprobPosition {
    bool forced = false;
    TokenLogprobEntry sampled;
    std::vector<TokenLogprobEntry> top;
    std::vector<TokenLogprobEntry> candidates;
};

struct PromptLogprobPosition {
    std::uint32_t position = 0;
    TokenLogprobPosition value; // `sampled` is the prompt's own next token
};

struct GenerationOutcome {
    std::string text;
    // One entry per generated token when the request enabled logprobs, otherwise empty.
    std::vector<TokenLogprobPosition> token_logprobs;
    // One entry per requested prompt position, in request order.
    std::vector<PromptLogprobPosition> prompt_logprobs;
    std::string reasoning;
    std::vector<ninfer::TokenId> generated_token_ids;
    std::vector<ninfer::GeneratedToolCall> tool_calls;
    ninfer::ToolCallParseDiagnostics tool_call_parse;
    int prompt_tokens                  = 0;
    int completion_tokens              = 0;
    int reasoning_tokens               = 0;
    ninfer::FinishReason finish_reason = ninfer::FinishReason::OutputLimit;
    std::optional<std::string> matched_stop_string;
    RequestStats stats;
    double service_prepare_seconds = 0.0;
};

// A transport may report an interrupted stream through a callback or is_cancelled().
// Generation still settles through Engine cancellation and returns its actual work statistics.
class ClientDisconnected final : public std::exception {
public:
    [[nodiscard]] const char* what() const noexcept override { return "client disconnected"; }
};

struct StreamSink {
    std::function<void(const ninfer::GenerationStart& start)> on_start;
    std::function<void(const ninfer::PromptProgress& progress)> on_progress;
    std::function<void(const ninfer::GenerationTimingObservation& timing)> on_timing;
    std::function<void(const std::string& delta_text)> on_content;
    std::function<void(const std::string& delta_text)> on_reasoning;
    std::function<void(const ninfer::ToolCallStart&)> on_tool_start;
    std::function<void(const ninfer::ToolArgumentsDelta&)> on_tool_arguments;
    std::function<void(const ninfer::ToolCallEnd&)> on_tool_end;
    std::function<bool()> is_cancelled;
};

enum class GenerationConsumerMode : std::uint8_t {
    Aggregate,
    Streaming,
};

// Translate Engine request failures into the shared protocol-neutral HTTP error contract.
ApiError
request_error_to_api_error(const ninfer::RequestError& exception,
                           std::string_view constraint_param = "structured_outputs.grammar",
                           std::span<const std::string> tool_schema_params = {});

// Preparation ends by synchronously submitting the owning prompt to the Engine FIFO. The returned
// request keeps its ingress/response lifetime reservation until the HTTP response is released and
// is consumed exactly once by run().
struct PreparedRequest {
    std::string constraint_param;
    std::vector<std::string> tool_schema_params;
    ninfer::GenerationHandle generation;
    ninfer::ResolvedSamplingParameters sampling;
    // Service input acquisition/bookkeeping; Engine timings own prompt and constraint preparation.
    double service_prepare_seconds = 0.0;
    double acquisition_seconds     = 0.0;
    PromptPreparationStats preparation;
    int prompt_tokens    = 0;
    bool enable_thinking = true;
    std::optional<std::uint32_t> thinking_budget;
    std::optional<ninfer::ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    // Prompt positions the request asked logprobs for, to label the readout in the outcome.
    std::vector<std::uint32_t> logprob_prompt_positions;
    std::shared_ptr<RequestLifetime> lifetime;
};

class GenerationService {
public:
    explicit GenerationService(product::ServeOptions options,
                               StartupObserver startup_observer = {});

    [[nodiscard]] const product::ServeOptions& options() const noexcept { return options_; }

    // Present when --expert-stats or --expert-trace is set.
    [[nodiscard]] ExpertRouteRecorder* expert_route_recorder() const noexcept {
        return expert_route_recorder_.get();
    }

    // Engine owns the once-normalized startup configuration. Serving diagnostics must use this
    // value instead of reinterpreting optional defaults from product::ServeOptions.
    [[nodiscard]] const ninfer::EngineOptions& engine_options() const { return engine_->options(); }

    // Raw artifact-tokenizer encoding, for validating single-token closed-set options.
    [[nodiscard]] std::vector<ninfer::TokenId> tokenize_text(std::string_view text) const {
        return engine_->tokenize_text(text);
    }

    [[nodiscard]] const EngineDescription& description() const { return engine_->description(); }

    [[nodiscard]] const StartupReport& startup_report() const { return engine_->startup_report(); }

    [[nodiscard]] EngineSnapshot snapshot(SnapshotOptions options = {}) const {
        return engine_->snapshot(options);
    }

    [[nodiscard]] std::optional<EngineSnapshot> try_snapshot(SnapshotOptions options = {}) const {
        return engine_->try_snapshot(options);
    }
    [[nodiscard]] bool is_available() const { return engine_->is_available(); }

    [[nodiscard]] ninfer::Engine::QuiescenceReport
    run_at_quiescence(std::function<void()> work) const {
        return engine_->run_at_quiescence(std::move(work));
    }

    [[nodiscard]] ninfer::ModelSamplingDefaults sampling_defaults() const {
        return engine_->sampling_defaults();
    }

    [[nodiscard]] const std::string& chat_template_source() const {
        return engine_->chat_template_source();
    }

    [[nodiscard]] PreparedRequest prepare(const GenerationRequest& req,
                                          GenerationConsumerMode consumer_mode,
                                          ninfer::GenerationObservationOptions observation = {},
                                          std::function<bool()> is_cancelled               = {},
                                          ContextCacheHints context_cache = {}) const;
    [[nodiscard]] int count_prompt_tokens(const GenerationRequest& req,
                                          std::function<bool()> is_cancelled = {}) const;
    // Frontend-only preparation: inspect template boundaries without submitting generation.
    [[nodiscard]] ninfer::PromptSummary
    describe_prompt(const GenerationRequest& req, std::function<bool()> is_cancelled = {}) const;

    // Establishes admission before an HTTP adapter commits its streaming response.
    void wait_for_admission(PreparedRequest& prepared,
                            std::function<bool()> is_cancelled = {}) const;

    // Consumes prepared.generation. A PreparedRequest is single-use.
    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink,
                          std::function<bool()> is_cancelled = {});

    void warmup();

private:
    [[nodiscard]] ninfer::PromptInput acquire_prompt_input(
        const GenerationRequest& req, std::chrono::steady_clock::time_point deadline,
        const std::function<bool()>& is_cancelled) const;

    enum class CacheParticipation : std::uint8_t {
        Disabled,
        ReadWrite,
    };

    enum class DeadlinePolicy : std::uint8_t {
        ClientPendingTimeout,
        UnboundedStartup,
    };

    [[nodiscard]] PreparedRequest
    prepare_impl(const GenerationRequest& req, GenerationConsumerMode consumer_mode,
                 ninfer::GenerationObservationOptions observation,
                 std::function<bool()> is_cancelled, ContextCacheHints context_cache,
                 CacheParticipation cache_participation, DeadlinePolicy deadline_policy) const;
    [[nodiscard]] std::shared_ptr<RequestLifetime>
    acquire_request_lifetime(DeadlinePolicy deadline_policy) const;

    product::ServeOptions options_;
    std::shared_ptr<ExpertRouteRecorder> expert_route_recorder_;
    std::unique_ptr<ninfer::Engine> engine_;
    std::shared_ptr<RequestCapacity> request_capacity_;
};

} // namespace ninfer::serve
