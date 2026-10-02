#include "ninfer/engine.h"

#include "core/device.h"
#include "core/nvtx.h"
#include "core/startup.h"
#include "runtime/contract/sampling.h"
#include "runtime/contract/request.h"
#include "runtime/engine/causal_score_core.h"
#include "runtime/engine/engine_core.h"
#include "runtime/engine/model_instance.h"

#include <algorithm>
#include <functional>
#include <future>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace ninfer {
namespace {

DeviceContext initialize_device(const EngineOptions& options) {
    StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaInitialize);
    DeviceContext device(options.device);
    phase.complete();
    return device;
}

runtime::ResolvedRequestOptions resolve_request_options(const ModelSamplingDefaults& defaults,
                                                        SamplingMode mode, RequestOptions options) {
    if (options.execution.thinking.budget && *options.execution.thinking.budget == 0) {
        throw std::invalid_argument("thinking budget must be positive");
    }
    runtime::ResolvedRequestOptions resolved;
    resolved.execution.sampling =
        runtime::resolve_sampling(defaults, mode, options.execution.sampling);
    resolved.execution.requested_output_tokens = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse      = options.execution.allow_prefix_reuse;
    resolved.execution.allow_prefix_publication = options.execution.allow_prefix_publication;
    resolved.execution.thinking                = options.execution.thinking;
    TokenLogprobOptions& logprobs               = options.execution.logprobs;
    if (!logprobs.enabled && (logprobs.top != 0 || !logprobs.candidates.empty())) {
        throw std::invalid_argument("token logprob alternatives require logprobs to be enabled");
    }
    if (logprobs.top > kMaximumTopLogprobs) {
        throw std::invalid_argument("top logprobs exceeds the supported maximum of 20");
    }
    if (logprobs.candidates.size() > kMaximumLogprobCandidates) {
        throw std::invalid_argument("too many logprob candidate tokens");
    }
    if (!logprobs.prompt_positions.empty()) {
        if (logprobs.top != 0) {
            throw std::invalid_argument("prompt position logprobs do not report top alternatives");
        }
        if (logprobs.prompt_positions.size() > kMaximumPromptReadouts) {
            throw std::invalid_argument("too many prompt positions for logprobs");
        }
        for (std::size_t i = 1; i < logprobs.prompt_positions.size(); ++i) {
            if (logprobs.prompt_positions[i] <= logprobs.prompt_positions[i - 1]) {
                throw std::invalid_argument("prompt positions for logprobs must be ascending");
            }
        }
    }
    resolved.execution.logprobs                = std::move(logprobs);
    resolved.stop                              = std::move(options.stop);
    resolved.output                            = options.output;
    return resolved;
}

std::string context_capacity_error(std::size_t prompt_tokens, std::uint32_t max_context) {
    return "prepared prompt has " + std::to_string(prompt_tokens) +
           " tokens, exceeding Engine max_context " + std::to_string(max_context);
}

} // namespace

class PreparedPrompt::Impl {
public:
    Impl(PromptSummary prompt_summary, PromptPreparationStats preparation, SamplingMode mode,
         models::qwen3_5::PreparedPrompt prepared)
        : summary(std::move(prompt_summary)), prepare(std::move(preparation)), sampling_mode(mode),
          value(std::move(prepared)) {}

    PromptSummary summary;
    PromptPreparationStats prepare;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    models::qwen3_5::PreparedPrompt value;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PromptPreparationStats& PreparedPrompt::preparation_stats() const noexcept {
    static const PromptPreparationStats empty;
    return impl_ != nullptr ? impl_->prepare : empty;
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

class GenerationHandle::Impl {
public:
    class Concept {
    public:
        virtual ~Concept() = default;
        virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
    };

    template <class Submission>
    class Model final : public Concept {
    public:
        Model(std::shared_ptr<void> keep_alive, Submission submission)
            : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)) {}

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }

    private:
        std::shared_ptr<void> keep_alive_;
        Submission submission_;
    };

    template <class Submission>
    Impl(std::shared_ptr<void> keep_alive, Submission submission,
         ResolvedSamplingParameters sampling)
        : state_(std::make_unique<Model<Submission>>(std::move(keep_alive), std::move(submission))),
          sampling_(sampling) {}

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return state_->wait(sink, cancellation);
    }

    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept {
        return sampling_;
    }

private:
    std::unique_ptr<Concept> state_;
    ResolvedSamplingParameters sampling_;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ != nullptr ? impl_->resolved_sampling() : empty;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

class Engine::Impl {
public:
    using GenerationCore = runtime::EngineCore<runtime::ModelInstance>;
    using ScoringCore    = runtime::CausalScoreCore<runtime::ModelInstance>;
    using Core =
        std::variant<std::monostate, std::unique_ptr<GenerationCore>, std::unique_ptr<ScoringCore>>;

    explicit Impl(EngineOptions engine_options)
        : options(runtime::normalize_engine_options(std::move(engine_options))),
          device(initialize_device(options)) {
        nvtx::ScopedRange load_range(nvtx::Name::EngineLoad, nvtx::Category::Runtime);
        auto constructed  = runtime::construct_model(options, device);
        active            = std::move(constructed.instance);
        load              = std::move(constructed.load);
        load.cuda_sync_mode = device.sync_mode();
        sampling_defaults = active->frontend.sampling_defaults();
        StartupPhaseScope finalize_phase(options.startup_observer, StartupPhase::EngineFinalize);
        if (options.purpose == EnginePurpose::CausalScoring) {
            core = std::make_unique<ScoringCore>(*active, device);
        } else {
            core = std::make_unique<GenerationCore>(*active, device, options,
                                                    std::move(constructed.context_cost));
        }
        finalize_phase.complete();
    }

    ~Impl() noexcept {
        device.bind_to_current_thread_noexcept();
        core.emplace<std::monostate>();
        try {
            device.synchronize();
        } catch (...) {}
    }

    EngineOptions options;
    DeviceContext device;
    std::unique_ptr<runtime::ModelInstance> active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    Core core;
};

Engine::Engine(EngineOptions options) {
    StartupObserver startup_observer = options.startup_observer;
    StartupPhaseScope startup_phase(startup_observer, StartupPhase::EngineStartup);
    impl_ = std::make_shared<Impl>(std::move(options));
    startup_phase.complete();
}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime);
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    auto prepared      = impl_->active->frontend.prepare(std::move(input), control);
    PromptSummary info = prepared.summary();
    const SamplingMode sampling_mode =
        info.starts_in_reasoning ? SamplingMode::Thinking : SamplingMode::NonThinking;
    if (info.prompt_tokens > impl_->active->capacity) {
        throw std::logic_error("target Frontend admitted a prompt beyond Engine capacity");
    }
    const PromptPreparationStats preparation = prepared.preparation_stats();
    return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(info, preparation, sampling_mode,
                                                                 std::move(prepared)));
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    nvtx::ScopedRange prepare_range(nvtx::Name::FrontendPrepare, nvtx::Category::Runtime,
                                    static_cast<std::uint64_t>(token_ids.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (token_ids.size() > impl_->active->capacity) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           context_capacity_error(token_ids.size(), impl_->active->capacity));
    }
    auto prepared =
        impl_->active->frontend.prepare_tokens(std::move(token_ids), allow_prefix_identity);
    PromptSummary info = prepared.summary();
    if (info.prompt_tokens > impl_->active->capacity) {
        throw std::logic_error("target Frontend admitted prompt tokens beyond capacity");
    }
    const PromptPreparationStats preparation = prepared.preparation_stats();
    return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
        info, preparation, SamplingMode::Thinking, std::move(prepared)));
}

std::vector<TokenId> Engine::tokenize_text(std::string_view text) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->active->frontend.tokenize_text(text);
}

std::string Engine::token_bytes(TokenId token) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->active->frontend.token_bytes(token);
}

std::vector<float> Engine::score_tokens(std::vector<TokenId> tokens, std::uint32_t first_target) {
    return score_tokens(std::move(tokens), first_target, CausalScoreReadout{}).target_logprobs;
}

CausalScores Engine::score_tokens(std::vector<TokenId> tokens, std::uint32_t first_target,
                                  const CausalScoreReadout& readout) {
    nvtx::ScopedRange score_range(nvtx::Name::Score, nvtx::Category::Scoring,
                                  static_cast<std::uint64_t>(tokens.size()));
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::CausalScoring) {
        throw std::logic_error("score_tokens requires a CausalScoring Engine");
    }
    if (tokens.size() < 2 || tokens.size() > impl_->options.max_context) {
        throw std::invalid_argument("score_tokens token count must be in [2,max_context]");
    }
    if (first_target == 0 || first_target >= tokens.size()) {
        throw std::invalid_argument("score_tokens first_target must be in [1,token_count-1]");
    }
    const std::size_t targets = tokens.size() - first_target;
    if (readout.reference_count > kMaximumCausalReferenceTokens ||
        readout.reference_tokens.size() != targets * readout.reference_count) {
        throw std::invalid_argument(
            "score_tokens readout needs reference_count<=kMaximumCausalReferenceTokens ids per "
            "scored target");
    }
    PreparedPrompt prompt = prepare_tokens(std::move(tokens), false);
    if (prompt.summary().prompt_tokens != targets + first_target) {
        throw std::logic_error("score_tokens preparation changed the token count");
    }
    CausalScores result = std::visit(
        [&](auto& core) -> CausalScores {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::unique_ptr<Impl::ScoringCore>>) {
                return core->score(std::move(prompt.impl_->value), first_target, readout);
            } else {
                throw std::logic_error("Engine scoring core is unavailable");
            }
        },
        impl_->core);
    const bool distributions = readout.reference_count != 0;
    if (result.target_logprobs.size() != targets ||
        result.reference_logprobs.size() != targets * readout.reference_count ||
        result.argmax_tokens.size() != (distributions ? targets : 0) ||
        result.argmax_logprobs.size() != result.argmax_tokens.size()) {
        throw std::logic_error("target Program returned an invalid causal score count");
    }
    if (readout.capture_hidden_rows
            ? (result.hidden_size == 0 ||
               result.hidden_rows.size() != targets * static_cast<std::size_t>(result.hidden_size))
            : (result.hidden_size != 0 || !result.hidden_rows.empty())) {
        throw std::logic_error("target Program returned an invalid hidden row readout");
    }
    return result;
}

std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->active->frontend.count_tokens(std::move(input), control);
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->sampling_defaults;
}

const PromptCapabilities& Engine::prompt_capabilities() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->active->frontend.prompt_capabilities();
}

const std::string& Engine::chat_template_source() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->active->frontend.chat_template_source();
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                OutputConsumerMode consumer_mode,
                                GenerationObservationOptions observation,
                                std::chrono::steady_clock::time_point pending_deadline) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (impl_->options.purpose != EnginePurpose::Generation) {
        throw std::logic_error("submit requires a Generation Engine");
    }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }
    if (observation.live_timings) { observation.phase_timings = true; }
    if (consumer_mode != OutputConsumerMode::Streaming &&
        (observation.live_timings || observation.prompt_progress)) {
        throw std::invalid_argument("live generation observations require a Streaming consumer");
    }

    const StructuredOutputOptions structured_output    = options.execution.structured_output;
    const std::vector<std::string> required_tool_names = options.execution.required_tool_names;
    if (!required_tool_names.empty() &&
        (!options.stop.strings.empty() || !options.stop.token_ids.empty() || options.output.raw)) {
        throw std::invalid_argument(
            "required tool calls cannot be combined with custom stops or raw output");
    }
    if (structured_output.kind != StructuredOutputKind::Text &&
        (!options.stop.strings.empty() || !options.stop.token_ids.empty() || options.output.raw ||
         options.output.preserve_special_tokens)) {
        throw std::invalid_argument(
            "structured output cannot be combined with custom stops or raw/special-token output");
    }
    const bool capture_features = options.execution.capture_reasoning_features;
    if (capture_features) {
        const PromptSummary& summary = prompt.impl_->summary;
        if (summary.has_media || !summary.reasoning_frontier || *summary.reasoning_frontier == 0 ||
            options.execution.requested_output_tokens == 0) {
            throw std::invalid_argument("reasoning features require a text new-assistant prompt "
                                        "and a positive output budget");
        }
        // The feature row must be computed by this request, and a training readout leaves the
        // context cache as it found it.
        options.execution.allow_prefix_reuse       = false;
        options.execution.allow_prefix_publication = false;
    }
    runtime::ResolvedRequestOptions resolved_options = resolve_request_options(
        impl_->sampling_defaults, prompt.impl_->sampling_mode, std::move(options));
    if (capture_features) {
        resolved_options.execution.reasoning_feature_position =
            *prompt.impl_->summary.reasoning_frontier - 1U;
    }
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;
    resolved_options.execution.output_constraint =
        impl_->active->frontend.compile_output_constraint(structured_output, required_tool_names);

    const PromptSummary prompt_summary = prompt.impl_->summary;
    // Validated here, before the request reaches a Program: inside admission an invalid
    // argument is an executor failure.
    if (!resolved_options.execution.logprobs.prompt_positions.empty() &&
        static_cast<std::uint64_t>(resolved_options.execution.logprobs.prompt_positions.back()) +
                1U >=
            prompt_summary.prompt_tokens) {
        throw std::invalid_argument("prompt position logprobs exceed the prompt");
    }
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            context_capacity_error(prompt_summary.prompt_tokens, impl_->options.max_context));
    }
    const double prepare_seconds = prompt.impl_->prepare.seconds;
    if (resolved_options.execution.requested_output_tokens == 0) {
        struct ImmediateSubmission {
            GenerationResult result;
            OutputConsumerMode consumer_mode = OutputConsumerMode::Aggregate;

            GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
                const bool streaming = consumer_mode == OutputConsumerMode::Streaming;
                if (streaming != (sink != nullptr)) {
                    throw std::invalid_argument(
                        "GenerationHandle wait sink does not match its submitted consumer mode");
                }
                if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
                return std::move(result);
            }
        } immediate{.consumer_mode = consumer_mode};

        immediate.result.prompt                     = prompt_summary;
        immediate.result.finish_reason              = FinishReason::OutputLimit;
        immediate.result.thinking.configured_budget = resolved_options.execution.thinking.budget;
        immediate.result.timings.prepare_seconds    = prepare_seconds;
        immediate.result.timings.total_seconds      = prepare_seconds;
        prompt.impl_.reset();
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(immediate), resolved_sampling));
    }

    return std::visit(
        [&](auto& core) -> GenerationHandle {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else if constexpr (std::is_same_v<CoreState, std::unique_ptr<Impl::ScoringCore>>) {
                throw std::logic_error("Engine generation core is unavailable");
            } else {
                auto submission = core->submit(std::move(prompt.impl_->value), prompt_summary,
                                               prepare_seconds, std::move(resolved_options),
                                               consumer_mode, observation, pending_deadline);
                return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
                    impl_, std::move(submission), resolved_sampling));
            }
        },
        impl_->core);
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    const OutputConsumerMode consumer_mode =
        sink != nullptr ? OutputConsumerMode::Streaming : OutputConsumerMode::Aggregate;
    return submit(std::move(prompt), std::move(options), consumer_mode, {})
        .wait(sink, cancellation);
}

const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

LoadSummary Engine::load_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->load;
}

MemorySummary Engine::memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& core) -> MemorySummary {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->memory_summary();
            }
        },
        impl_->core);
}

std::optional<MemorySummary> Engine::try_memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& core) -> std::optional<MemorySummary> {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else if constexpr (requires { core->try_memory_summary(); }) {
                return core->try_memory_summary();
            } else {
                return core->memory_summary();
            }
        },
        impl_->core);
}

MediaCacheSummary Engine::media_cache_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->active->frontend.media_cache_summary();
}

RuntimeStats Engine::runtime_stats() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& core) -> RuntimeStats {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else {
                return core->runtime_stats();
            }
        },
        impl_->core);
}

bool Engine::is_available() const {
    if (impl_ == nullptr) { return false; }
    return std::visit(
        [](const auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                return false;
            } else {
                return core != nullptr && core->is_available();
            }
        },
        impl_->core);
}

Engine::QuiescenceReport Engine::run_at_quiescence(std::function<void()> work) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (work == nullptr) { throw std::invalid_argument("run_at_quiescence needs work to run"); }
    auto pending = std::visit(
        [&work](auto& core) -> std::future<runtime::QuiescenceOutcome> {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (std::is_same_v<CoreState, std::monostate>) {
                throw std::logic_error("Engine core is unavailable");
            } else if constexpr (requires { core->run_at_quiescence(std::move(work)); }) {
                return core->run_at_quiescence(std::move(work));
            } else {
                // The causal-scoring cores run no worker loop and hold no lanes,
                // so there is no boundary to fence and nothing to drain. Saying
                // so beats inventing a no-op that would silently report success
                // for a residency change that never became safe.
                throw std::logic_error("this engine core does not schedule work to quiesce");
            }
        },
        impl_->core);
    // Blocking is the contract: the caller is changing residency and must not
    // proceed until the engine has actually stopped. get() rethrows whatever the
    // work threw, on this thread rather than the worker's.
    const runtime::QuiescenceOutcome outcome = pending.get();
    QuiescenceReport report;
    report.drain         = outcome.drain;
    report.work          = outcome.work;
    report.requests_held = outcome.requests_held;
    return report;
}

void Engine::reset_memory_peaks() noexcept {
    if (impl_ == nullptr) { return; }
    std::visit(
        [](auto& core) {
            using CoreState = std::remove_cvref_t<decltype(core)>;
            if constexpr (!std::is_same_v<CoreState, std::monostate>) {
                core->reset_memory_peaks();
            }
        },
        impl_->core);
}

} // namespace ninfer
