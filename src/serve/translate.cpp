#include "serve/translate.h"
#include "serve/request_json.h"
#include "serve/request_validation.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

std::uint64_t random_seed() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    return rng();
}

[[noreturn]] void invalid_sampling(std::string message, std::string param) {
    ApiError error;
    error.message = std::move(message);
    error.param   = std::move(param);
    throw ApiException(std::move(error));
}

[[noreturn]] void invalid_prompt_option(std::string message, std::string param, std::string code) {
    ApiError error;
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

ninfer::SamplingOverrides resolve_sampling_overrides(const SamplingParams& request,
                                                     const ServeOptions& server) {
    ninfer::SamplingOverrides sampling = server.sampling_overrides;
    if (request.temperature) { sampling.temperature = static_cast<float>(*request.temperature); }
    if (request.top_p) { sampling.top_p = static_cast<float>(*request.top_p); }
    if (request.min_p) { sampling.min_p = static_cast<float>(*request.min_p); }
    if (request.top_k) { sampling.top_k = static_cast<std::int32_t>(*request.top_k); }
    if (request.presence_penalty) {
        sampling.presence_penalty = static_cast<float>(*request.presence_penalty);
    }
    if (request.frequency_penalty) {
        sampling.frequency_penalty = static_cast<float>(*request.frequency_penalty);
    }
    if (request.repetition_penalty) {
        sampling.repetition_penalty = static_cast<float>(*request.repetition_penalty);
    }
    if (request.seed) {
        sampling.seed = *request.seed;
    } else if (server.sampling_overrides.seed) {
        sampling.seed = *server.sampling_overrides.seed;
    } else {
        sampling.seed = random_seed();
    }

    const auto finite = [](const std::optional<float>& value) {
        return !value || std::isfinite(*value);
    };
    if (!finite(sampling.temperature) || !finite(sampling.top_p) || !finite(sampling.min_p) ||
        !finite(sampling.presence_penalty) || !finite(sampling.frequency_penalty) ||
        !finite(sampling.repetition_penalty)) {
        invalid_sampling("sampling parameters must be finite", "sampling");
    }
    if (sampling.temperature && (*sampling.temperature < 0.0F || *sampling.temperature > 2.0F)) {
        invalid_sampling("temperature must be in [0,2]", "temperature");
    }
    if (sampling.top_p && (*sampling.top_p < 0.0F || *sampling.top_p > 1.0F)) {
        invalid_sampling("top_p must be in [0,1]", "top_p");
    }
    if (sampling.top_k && *sampling.top_k < 0) {
        invalid_sampling("top_k must not be negative", "top_k");
    }
    // Clamp rather than refuse. The sampler's candidate domain really is 20
    // (kSamplerFastCandidates: each 256-thread block reduces its tile to 20
    // candidates in shared memory, and widening that costs occupancy), but
    // llama.cpp and Ollama both default top_k to 40, so every client carrying
    // that default was getting a hard 400 on its first request against an
    // otherwise OpenAI-compatible server.
    //
    // Clamping is close to free in behaviour: with top_p or min_p also active
    // over a 248,320-token vocabulary the nucleus almost always closes well
    // inside 20 candidates, so 40 and 20 select the same token nearly always.
    // Trading an exact-parameter guarantee for "every client works" is the right
    // way round for an engine meant to sit behind whatever app the user already
    // has. The effective value is what the request log records, so a run is
    // still reproducible from its log rather than from what was asked for.
    if (sampling.top_k && *sampling.top_k > 20) { sampling.top_k = 20; }
    if (sampling.min_p && (*sampling.min_p < 0.0F || *sampling.min_p > 1.0F)) {
        invalid_sampling("min_p must be in [0,1]", "min_p");
    }
    if (sampling.presence_penalty &&
        (*sampling.presence_penalty < -2.0F || *sampling.presence_penalty > 2.0F)) {
        invalid_sampling("presence_penalty must be in [-2,2]", "presence_penalty");
    }
    if (sampling.frequency_penalty &&
        (*sampling.frequency_penalty < -2.0F || *sampling.frequency_penalty > 2.0F)) {
        invalid_sampling("frequency_penalty must be in [-2,2]", "frequency_penalty");
    }
    if (sampling.repetition_penalty && *sampling.repetition_penalty <= 0.0F) {
        invalid_sampling("repetition_penalty must be positive", "repetition_penalty");
    }
    if (server.greedy) { sampling.temperature = 0.0F; }
    return sampling;
}

std::vector<const ToolDefinition*> effective_tools(const GenerationRequest& request) {
    std::vector<const ToolDefinition*> tools;
    if (!request.uses_tools()) { return tools; }
    tools.reserve(request.tools.size());
    for (const ToolDefinition& tool : request.tools) { tools.push_back(&tool); }
    return tools;
}

std::string render_tool_definition(const ToolDefinition& tool) {
    using Json  = RequestJson;
    Json schema = Json::parse(tool.input_schema_json);
    Json function{{"name", tool.name}, {"parameters", std::move(schema)}, {"strict", false}};
    if (!tool.description.empty()) { function["description"] = tool.description; }
    if (tool.input_examples_json) {
        function["input_examples"] = Json::parse(*tool.input_examples_json);
    }
    return Json{{"type", "function"}, {"function", std::move(function)}}.dump();
}

ninfer::ReasoningEffort engine_reasoning_effort(RequestedReasoningEffort effort) {
    switch (effort) {
    case RequestedReasoningEffort::None:
        return ninfer::ReasoningEffort::None;
    case RequestedReasoningEffort::Minimal:
        return ninfer::ReasoningEffort::Minimal;
    case RequestedReasoningEffort::Low:
        return ninfer::ReasoningEffort::Low;
    case RequestedReasoningEffort::Medium:
        return ninfer::ReasoningEffort::Medium;
    case RequestedReasoningEffort::High:
        return ninfer::ReasoningEffort::High;
    case RequestedReasoningEffort::XHigh:
        return ninfer::ReasoningEffort::XHigh;
    case RequestedReasoningEffort::Max:
        return ninfer::ReasoningEffort::Max;
    }
    throw std::logic_error("invalid requested reasoning effort");
}

// "xhigh (default), medium, and low": the template default first, then descending effort.
std::string supported_reasoning_efforts(const ninfer::ReasoningEffortCapabilities& capabilities) {
    std::vector<std::string> names;
    if (capabilities.default_effort) {
        names.push_back(std::string(ninfer::reasoning_effort_name(*capabilities.default_effort)) +
                        " (default)");
    }
    for (auto it = capabilities.supported.rbegin(); it != capabilities.supported.rend(); ++it) {
        if (*it != capabilities.default_effort) {
            names.emplace_back(ninfer::reasoning_effort_name(*it));
        }
    }
    std::string text;
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (i > 0) { text += names.size() == 2 ? " " : ", "; }
        if (i > 0 && i + 1 == names.size()) { text += "and "; }
        text += names[i];
    }
    return text;
}

} // namespace

ResolvedPromptSemantics resolve_prompt_semantics(const GenerationRequest& request,
                                                 const ServeOptions& server,
                                                 const ninfer::PromptCapabilities& capabilities) {
    using Json  = RequestJson;
    Json kwargs = request.chat_template_kwargs_json.empty()
                      ? Json::object()
                      : Json::parse(request.chat_template_kwargs_json);
    if (!kwargs.is_object())
        invalid_prompt_option("chat_template_kwargs must be an object", "chat_template_kwargs",
                              "invalid_template_option");
    auto merge_boolean = [&](const char* key, std::optional<bool> typed) {
        if (!kwargs.contains(key) || kwargs[key].is_null()) {
            kwargs.erase(key);
            return typed;
        }
        if (!kwargs[key].is_boolean())
            invalid_prompt_option(std::string(key) + " must be a boolean", key,
                                  "invalid_template_option");
        const bool nested = kwargs[key].get<bool>();
        if (typed && *typed != nested)
            invalid_prompt_option(std::string("conflicting ") + key + " values", key,
                                  "conflicting_template_option");
        kwargs.erase(key);
        return std::optional<bool>(nested);
    };
    auto thinking = merge_boolean("enable_thinking", request.enable_thinking);
    auto preserve = merge_boolean("preserve_thinking", request.preserve_thinking);
    auto effort   = request.reasoning_effort;
    if (kwargs.contains("reasoning_effort") && !kwargs["reasoning_effort"].is_null()) {
        if (!kwargs["reasoning_effort"].is_string())
            invalid_prompt_option("reasoning_effort must be a string", "reasoning_effort",
                                  "invalid_template_option");
        const auto nested =
            parse_requested_reasoning_effort(kwargs["reasoning_effort"].get<std::string>());
        if (!nested)
            invalid_prompt_option("invalid reasoning_effort", "reasoning_effort",
                                  "invalid_template_option");
        if (effort && effort != nested)
            invalid_prompt_option("conflicting reasoning_effort values", "reasoning_effort",
                                  "conflicting_template_option");
        effort = nested;
    }
    kwargs.erase("reasoning_effort");
    ResolvedPromptSemantics result{
        .enable_thinking           = thinking ? thinking : server.enable_thinking,
        .preserve_thinking         = preserve.value_or(server.preserve_thinking),
        .chat_template_kwargs_json = kwargs.dump(),
    };
    if (effort) {
        const bool enables = *effort != RequestedReasoningEffort::None;
        if (thinking && *thinking != enables)
            invalid_prompt_option("reasoning effort conflicts with enable_thinking",
                                  "reasoning_effort", "conflicting_template_option");
        // A request effort overrides the server's thinking default.
        result.enable_thinking = enables;
        result.reasoning_effort = engine_reasoning_effort(*effort);
        // None disables thinking on every template. Any other effort must be one the loaded
        // template accepts, checked here so the template never rejects it during rendering.
        const ninfer::ReasoningEffortCapabilities& efforts = capabilities.reasoning_effort;
        if (enables && !efforts.supports(*result.reasoning_effort)) {
            if (efforts.supported.empty()) {
                invalid_prompt_option("the loaded chat template does not support reasoning effort",
                                      "reasoning_effort", "reasoning_effort_not_supported");
            }
            invalid_prompt_option("Unexpected reasoning effort " +
                                      std::string(requested_reasoning_effort_name(*effort)) +
                                      ". Supported types are " +
                                      supported_reasoning_efforts(efforts) + ".",
                                  "reasoning_effort", "reasoning_effort_not_supported");
        }
    }
    if (request.continuation == ninfer::PromptContinuationMode::ContinueFinalAssistant &&
        result.enable_thinking == true) {
        invalid_prompt_option("assistant prefill cannot be combined with enabled thinking",
                              "messages", "assistant_prefill_not_supported");
    }
    return result;
}

ninfer::PromptInput to_prompt_input(const GenerationRequest& request,
                                    const ResolvedPromptSemantics& semantics,
                                    const MediaAcquirer& acquire_media) {
    ninfer::PromptInput input;
    input.messages.reserve(request.messages.size());
    for (std::size_t turn_index = 0; turn_index < request.messages.size(); ++turn_index) {
        const ChatTurn& turn = request.messages[turn_index];
        ninfer::ChatMessage message;
        message.role              = turn.role;
        message.reasoning_content = turn.reasoning_content;
        message.tool_call_id      = turn.tool_call_id;
        message.tool_calls.reserve(turn.tool_calls.size());
        for (const ToolCall& call : turn.tool_calls) {
            message.tool_calls.push_back(ninfer::ToolCall{call.id, call.name, call.arguments_json});
        }

        if (turn.role == ChatRole::Tool && turn.tool_result_is_error) {
            ninfer::MessagePart error;
            error.text = "[tool_error]\n";
            message.parts.push_back(std::move(error));
        }

        std::uint64_t text_bytes = 0;
        for (std::size_t part_index = 0; part_index < turn.content.size(); ++part_index) {
            const ContentPart& part = turn.content[part_index];
            if (part.kind == ContentKind::Text) {
                ninfer::MessagePart text;
                text.text = part.text;
                message.parts.push_back(std::move(text));
                if (part.text.size() > std::numeric_limits<std::uint32_t>::max() - text_bytes) {
                    throw std::invalid_argument("cacheable instruction text exceeds uint32");
                }
                text_bytes += part.text.size();
            } else if (part.kind == ContentKind::Image || part.kind == ContentKind::Video) {
                if (!acquire_media) {
                    throw std::logic_error("media acquisition callback is not configured");
                }
                ninfer::MessagePart media;
                media.kind  = ninfer::MessagePartKind::Media;
                media.media = acquire_media(part);
                message.parts.push_back(std::move(media));
            } else {
                ApiError error;
                error.message = "content type '" + part.type_raw + "' is not supported";
                error.param   = "messages";
                error.code    = "modality_not_supported";
                throw ApiException(std::move(error));
            }
            if (part.cache_boundary_after) {
                if (turn_index >= std::numeric_limits<std::uint32_t>::max() ||
                    part_index >= std::numeric_limits<std::uint32_t>::max()) {
                    throw std::overflow_error("conversation cache boundary exceeds uint32");
                }
                const bool leading_instruction =
                    turn_index == 0 &&
                    (turn.role == ChatRole::System || turn.role == ChatRole::Developer) &&
                    part.kind == ContentKind::Text;
                if (leading_instruction) {
                    input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                        .kind     = part.cache_boundary_after->kind,
                        .evidence = part.cache_boundary_after->evidence,
                        .location = ninfer::PromptCacheMarkerLocation::LeadingInstructionBoundary,
                        .leading_instruction_bytes = static_cast<std::uint32_t>(text_bytes),
                    });
                } else {
                    input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                        .after_message_count = static_cast<std::uint32_t>(turn_index + 1U),
                        .kind                = part.cache_boundary_after->kind,
                        .evidence            = part.cache_boundary_after->evidence,
                        .location = ninfer::PromptCacheMarkerLocation::MessagePartBoundary,
                        .after_message_part_count =
                            static_cast<std::uint32_t>(message.parts.size()),
                    });
                }
            }
        }
        input.messages.push_back(std::move(message));
        if (turn.cache_boundary_after) {
            if (input.messages.size() > std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("conversation cache boundary exceeds uint32");
            }
            input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                .after_message_count = static_cast<std::uint32_t>(input.messages.size()),
                .kind                = turn.cache_boundary_after->kind,
                .evidence            = turn.cache_boundary_after->evidence,
                .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary,
            });
        }
    }

    input.options.continuation                     = request.continuation;
    input.options.enable_thinking                  = semantics.enable_thinking;
    input.options.reasoning_effort                 = semantics.reasoning_effort;
    input.options.preserve_thinking                = semantics.preserve_thinking;
    input.options.chat_template_kwargs_json        = semantics.chat_template_kwargs_json;
    input.options.add_vision_id                    = false;
    const std::vector<const ToolDefinition*> tools = effective_tools(request);
    input.options.tool_jsons.reserve(tools.size());
    for (std::size_t index = 0; index < tools.size(); ++index) {
        input.options.tool_jsons.push_back(render_tool_definition(*tools[index]));
        if (tools[index]->cache_boundary_after) {
            input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                .kind             = tools[index]->cache_boundary_after->kind,
                .evidence         = tools[index]->cache_boundary_after->evidence,
                .location         = ninfer::PromptCacheMarkerLocation::ToolBoundary,
                .after_tool_count = static_cast<std::uint32_t>(index + 1U),
            });
        }
    }
    input.context_cache.allow_engine_automatic_shared_prefixes =
        request.allow_engine_automatic_shared_prefixes;
    return input;
}

ninfer::RequestOptions to_request_options(const GenerationRequest& request,
                                          const ServeOptions& server,
                                          const ResolvedPromptSemantics& semantics,
                                          bool allow_prefix_reuse) {
    ninfer::RequestOptions options;
    options.execution.structured_output = request.structured_output;
    if (request.tool_choice.mode == ToolChoiceMode::Required) {
        if (request.tools.empty()) {
            bad_request("required tool choice needs at least one tool", "tool_choice");
        }
        if (!request.stop_strings.empty()) {
            bad_request("required tool choice conflicts with custom stop strings", "stop",
                        "tool_choice_conflict");
        }
        for (const auto& tool : request.tools) {
            options.execution.required_tool_names.push_back(tool.name);
        }
    }
    if (request.structured_output.kind != StructuredOutputKind::Text &&
        (request.uses_tools() || !request.stop_strings.empty())) {
        bad_request("structured output cannot be combined with active tools or custom stop strings",
                    "response_format", "structured_output_conflict");
    }
    options.execution.requested_output_tokens = static_cast<std::uint32_t>(request.max_tokens);
    options.execution.allow_prefix_reuse      = allow_prefix_reuse;
    options.execution.allow_prefix_publication = !request.prompt_cache_read_only;
    if (semantics.enable_thinking != false) {
        options.execution.thinking.budget =
            request.thinking_budget ? request.thinking_budget : server.default_thinking_budget;
    }
    options.execution.sampling             = resolve_sampling_overrides(request.sampling, server);
    options.output.raw                     = false;
    options.output.preserve_special_tokens =
        request.structured_output.kind == StructuredOutputKind::Text &&
        (request.uses_tools() || request.has_tool_history());
    options.output.tool_name_max_length = static_cast<std::uint32_t>(request.tool_name_max_length);
    options.stop.strings.reserve(request.stop_strings.size() *
                                 (request.stop_strings_apply_to_reasoning ? 2U : 1U));
    for (const std::string& stop : request.stop_strings) {
        if (!stop.empty()) {
            options.stop.strings.push_back(
                ninfer::StopString{.text              = stop,
                                   .channel           = ninfer::OutputChannel::Content,
                                   .include_in_output = false});
            if (request.stop_strings_apply_to_reasoning) {
                options.stop.strings.push_back(
                    ninfer::StopString{.text              = stop,
                                       .channel           = ninfer::OutputChannel::Reasoning,
                                       .include_in_output = false});
            }
        }
    }
    return options;
}

} // namespace ninfer::serve
