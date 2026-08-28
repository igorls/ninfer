#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct ReferenceToolOptions {
    std::string model_path;
    std::string mode =
        "preflight"; // "preflight", "execute-token", "materialize-full", or "materialize-vision"
    std::uint32_t max_context     = 4096;
    std::uint32_t max_concurrency = 1;
    std::uint32_t page_groups     = 0;
    std::uint32_t state_slots     = 0;
    std::int32_t token_id         = 0;
    bool do_commit                = true;
    bool json_output              = false;
};

void print_reference_tool_usage(std::string_view prog);

[[nodiscard]] ReferenceToolOptions
parse_reference_tool_options(std::span<const std::string_view> args);

[[nodiscard]] ReferenceToolOptions parse_reference_tool_options(int argc, char** argv);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
