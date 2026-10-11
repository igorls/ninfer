#pragma once

// OpenAI wire objects shared by Chat Completions and Responses HTTP handlers.

#include "serve/request.h"
#include "serve/request_json.h"
#include "serve/serve_options.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// The names a request may give the served model: its public id, the configured aliases and
// kDefaultModelName. Responses report the public id.
struct ServedModelNames {
    std::string id;
    std::vector<std::string> aliases;

    [[nodiscard]] bool accepts(std::string_view name) const;
};

enum class OpenAIPromptCacheAutomatic : std::uint8_t {
    Default,
    Requested,
    Disabled,
};

struct OpenAIPromptCachePolicy {
    OpenAIPromptCacheAutomatic automatic = OpenAIPromptCacheAutomatic::Default;
};

[[nodiscard]] bool parse_openai_prompt_cache_breakpoint(const RequestJson& value,
                                                        std::string_view param);
[[nodiscard]] OpenAIPromptCachePolicy parse_openai_prompt_cache_policy(const RequestJson& body);
void apply_openai_prompt_cache_policy(GenerationRequest& request, OpenAIPromptCachePolicy policy);

// Lists the public id first, then each alias, as separate entries.
std::string make_models_list(const ServedModelNames& names, std::int64_t created,
                             std::uint32_t max_model_len);
std::string make_model_object(const std::string& model_id, std::int64_t created,
                              std::uint32_t max_model_len);
std::string make_error_body(const ApiError& error);
std::int64_t unix_time_now();

// Returns the public id that responses report for an accepted `requested` name; throws
// `model_not_found` otherwise.
[[nodiscard]] const std::string& resolve_openai_model(std::string_view requested,
                                                      const ServedModelNames& names);

std::string new_openai_chat_completion_id();
std::string new_openai_chat_tool_call_id();
std::string new_openai_request_id();
std::string new_openai_response_id();
std::string new_openai_response_item_id(std::string_view prefix);

} // namespace ninfer::serve
