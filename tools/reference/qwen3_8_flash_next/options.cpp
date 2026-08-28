#include "options.h"

#include <charconv>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

std::uint32_t parse_strict_u32(std::string_view str, std::string_view option_name) {
    if (str.empty()) { throw std::invalid_argument(std::string(option_name) + ": empty value"); }
    // Reject leading +/- signs for unsigned integers
    if (str.front() == '+' || str.front() == '-') {
        throw std::invalid_argument(std::string(option_name) + ": invalid unsigned integer '" +
                                    std::string(str) + "'");
    }
    std::uint32_t val    = 0;
    const auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), val);
    if (ec != std::errc{} || ptr != str.data() + str.size()) {
        throw std::invalid_argument(std::string(option_name) +
                                    ": invalid unsigned integer or trailing characters in '" +
                                    std::string(str) + "'");
    }
    return val;
}

std::int32_t parse_strict_i32(std::string_view str, std::string_view option_name) {
    if (str.empty()) { throw std::invalid_argument(std::string(option_name) + ": empty value"); }
    std::int32_t val     = 0;
    const auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), val);
    if (ec != std::errc{} || ptr != str.data() + str.size()) {
        throw std::invalid_argument(std::string(option_name) +
                                    ": invalid integer or trailing characters in '" +
                                    std::string(str) + "'");
    }
    return val;
}

} // namespace

void print_reference_tool_usage(std::string_view prog) {
    std::cout
        << "Usage: " << prog << " [options]\n\n"
        << "Target-private reference and inspection tool for Qwen3.8-Flash-Next native "
           "artifacts.\n\n"
        << "Options:\n"
        << "  --model, -m <path>         Path to .ninfer artifact (required)\n"
        << "  --mode <mode>              Execution mode: 'preflight' (default), "
           "'execute-token', or 'materialize-full'\n"
        << "  --preflight                Shortcut for --mode preflight\n"
        << "  --execute-token            Shortcut for --mode execute-token\n"
        << "  --materialize-full         Shortcut for --mode materialize-full\n"
        << "  --max-context <tokens>     Maximum context length in tokens (default: 4096, max: "
           "262144)\n"
        << "  --max-concurrency <B>      Maximum concurrent decode requests (default: 1, range: "
           "[1, "
           "8])\n"
        << "  --page-groups <N>          Exact physical page group count (default: 0 = max for "
           "context)\n"
        << "  --state-slots <N>          State slot capacity (default: 2 * concurrency, range: [2 "
           "* B, 64])\n"
        << "  --token-id <id>            Input token ID for execute-token mode in [0, 248320) "
           "(default: 0)\n"
        << "  --commit                   Commit the executed token round (default: true)\n"
        << "  --abort                    Abort the executed token round without committing\n"
        << "  --json                     Output structured JSON\n"
        << "  --help, -h                 Show this help message\n";
}

ReferenceToolOptions parse_reference_tool_options(std::span<const std::string_view> args) {
    ReferenceToolOptions opts;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        if (arg == "--help" || arg == "-h") {
            print_reference_tool_usage(args.empty() ? "ninfer_qwen3_8_flash_next_reference"
                                                    : args[0]);
            std::exit(0);
        } else if (arg == "--model" || arg == "-m") {
            if (++i >= args.size()) throw std::invalid_argument("Missing argument for --model");
            opts.model_path = std::string(args[i]);
        } else if (arg == "--mode") {
            if (++i >= args.size()) throw std::invalid_argument("Missing argument for --mode");
            opts.mode = std::string(args[i]);
        } else if (arg == "--preflight") {
            opts.mode = "preflight";
        } else if (arg == "--execute-token") {
            opts.mode = "execute-token";
        } else if (arg == "--materialize-full") {
            opts.mode = "materialize-full";
        } else if (arg == "--max-context") {
            if (++i >= args.size())
                throw std::invalid_argument("Missing argument for --max-context");
            opts.max_context = parse_strict_u32(args[i], "--max-context");
        } else if (arg == "--max-concurrency") {
            if (++i >= args.size())
                throw std::invalid_argument("Missing argument for --max-concurrency");
            opts.max_concurrency = parse_strict_u32(args[i], "--max-concurrency");
        } else if (arg == "--page-groups") {
            if (++i >= args.size())
                throw std::invalid_argument("Missing argument for --page-groups");
            opts.page_groups = parse_strict_u32(args[i], "--page-groups");
        } else if (arg == "--state-slots") {
            if (++i >= args.size())
                throw std::invalid_argument("Missing argument for --state-slots");
            opts.state_slots = parse_strict_u32(args[i], "--state-slots");
        } else if (arg == "--token-id") {
            if (++i >= args.size()) throw std::invalid_argument("Missing argument for --token-id");
            opts.token_id = parse_strict_i32(args[i], "--token-id");
        } else if (arg == "--commit") {
            opts.do_commit = true;
        } else if (arg == "--abort") {
            opts.do_commit = false;
        } else if (arg == "--json") {
            opts.json_output = true;
        } else {
            throw std::invalid_argument("Unknown argument: " + std::string(arg));
        }
    }

    if (opts.model_path.empty()) { throw std::invalid_argument("--model <path> is required"); }
    if (opts.mode != "preflight" && opts.mode != "execute-token" &&
        opts.mode != "materialize-full") {
        throw std::invalid_argument(
            "Invalid --mode: must be 'preflight', 'execute-token', or 'materialize-full'");
    }
    if (opts.max_concurrency < 1 || opts.max_concurrency > 8) {
        throw std::invalid_argument("--max-concurrency must be between 1 and 8");
    }
    if (opts.max_context < 1 || opts.max_context > 262'144) {
        throw std::invalid_argument("--max-context must be between 1 and 262144");
    }
    if (opts.state_slots == 0) {
        opts.state_slots = 2U * opts.max_concurrency;
    } else if (opts.state_slots < 2U * opts.max_concurrency || opts.state_slots > 64) {
        throw std::invalid_argument("--state-slots must be between 2 * max_concurrency and 64");
    }
    if (opts.token_id < 0 || opts.token_id >= 248'320) {
        throw std::invalid_argument("--token-id must be in range [0, 248320)");
    }

    return opts;
}

ReferenceToolOptions parse_reference_tool_options(int argc, char** argv) {
    std::vector<std::string_view> args;
    args.reserve(argc > 1 ? static_cast<std::size_t>(argc - 1) : 0);
    for (int i = 1; i < argc; ++i) { args.emplace_back(argv[i]); }
    return parse_reference_tool_options(args);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
