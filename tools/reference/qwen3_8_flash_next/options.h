#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct ReferenceToolOptions {
    std::string model_path;
    std::string mode =
        "preflight"; // "preflight", "execute-token", "materialize-full", "materialize-vision", "chat-diagnostic", or "execute-vision"
    std::uint32_t max_context     = 4096;
    std::uint32_t max_concurrency = 1;
    std::uint32_t page_groups     = 0;
    std::uint32_t state_slots     = 0;
    std::int32_t token_id         = 0;
    bool do_commit                = true;
    bool json_output              = false;

    // Chat diagnostic options
    std::string prompt;
    std::string system_prompt;
    std::string dump_states;
    float temperature             = 1.0f;
    std::int32_t top_k            = 20;
    float top_p                   = 0.95f;
    unsigned long long seed       = 0ULL;
    std::uint32_t max_tokens      = 512;
    std::uint32_t thinking_budget = 0;
    std::string reasoning_effort  = "medium";
};

void print_reference_tool_usage(std::string_view prog);

[[nodiscard]] ReferenceToolOptions
parse_reference_tool_options(std::span<const std::string_view> args);

[[nodiscard]] ReferenceToolOptions parse_reference_tool_options(int argc, char** argv);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
