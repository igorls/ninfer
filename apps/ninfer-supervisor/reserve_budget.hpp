#pragma once
#include <nlohmann/json.hpp>
#include "product/config/options.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::supervisor {
inline std::vector<std::string> without_reserve_args(const std::vector<std::string>& args) {
    return product::remove_serve_argument(args, "engine.desktop_reserve");
}

// A Context capacity being saved together with the reserve. The reserve and the pool share one
// budget, so a reserve that only fits beside a smaller pool has to be judged against that pool
// rather than the one still running.
struct PlannedContext {
    bool automatic = false;
    double bytes   = 0;
};

// Preserve the loaded model and its capacity policy. A minimum runtime alone
// would silently shrink an explicitly configured Context backing; only auto may shrink. The
// startup budget already excludes foreign allocations and the chosen reserve;
// add that reserve back, then subtract the exact runtime and required slack.
inline nlohmann::json model_reserve_budget(const nlohmann::json& admin,
                                           std::optional<PlannedContext> planned = std::nullopt) {
    nlohmann::json out{{"ok", false}, {"reason", "Start the engine and wait for its memory plan before adjusting the reserve."}};
    try {
        const auto& plan = admin.at("plan");
        const auto& reserve = admin.at("desktop_reserve");
        const double runtime = plan.at("runtime_reservation_bytes").get<double>();
        const double weights = admin.at("resources").at("allocations").at("device_weights_bytes").get<double>();
        const double available = plan.at("available_after_weights_bytes").get<double>();
        const double configured = reserve.at("configured_bytes").get<double>();
        const double free = admin.at("device").at("free_bytes").get<double>();
        const double graphs = plan.at("cuda_graph_rebuild_estimate_bytes").get<double>();
        // Startup headroom applies to both explicit and automatic byte budgets.
        const bool automatic = planned ? planned->automatic : plan.at("context_capacity_automatic").get<bool>();
        const double slack = plan.at("headroom_bytes").get<double>();
        const double explicit_runtime = planned && !planned->automatic
            ? runtime - plan.at("context_capacity_bytes").get<double>() + planned->bytes
            : runtime;
        const double required_runtime = automatic ? plan.at("minimum_runtime_reservation_bytes").get<double>() : explicit_runtime;
        if (runtime <= 0 || weights <= 0 || available < 0 || configured < 0 || free < 0 || graphs < 0) return out;
        if (required_runtime <= 0 || (!planned && required_runtime > runtime)) return out;
        const double startup_limit = available - required_runtime - graphs - slack;
        const double live_limit = free + (runtime - required_runtime) - graphs - slack;
        const int maximum = static_cast<int>(std::floor(std::max(0.0, std::min({startup_limit, live_limit, 64.0 * 1073741824.0})) / 1073741824.0));
        return {{"ok", true}, {"max_gib", maximum}, {"weights_bytes", weights},
                {"runtime_bytes", required_runtime}, {"automatic_capacity", automatic}, {"model", admin.value("model_id", std::string{})},
                {"reason", "Keeps the loaded model and configured capacity policy, including startup slack and graph headroom. Other apps can reduce this limit."}};
    } catch (...) { return out; }
}

// What the Engine planned at its last start, from the request log's server_start record. This is
// the only source of its token estimate: /admin/vram reports bytes, and the Settings page needs
// what a byte of Context holds. Null when the line is empty or does not carry the plan.
inline nlohmann::json context_plan_from_server_start(const std::string& line) {
    if (line.empty()) return nullptr;
    try {
        const auto record  = nlohmann::json::parse(line);
        const auto& engine = record.at("engine");
        std::uint64_t capacity = 0, block = 0;
        for (const auto& pool : engine.at("context_pools")) {
            if (pool.at("domain").get<std::string>() != "device") continue;
            capacity += pool.at("capacity_bytes").get<std::uint64_t>();
            block = pool.at("block_bytes").get<std::uint64_t>();
        }
        return {{"token_estimate", engine.at("context_token_capacity_estimate").get<std::uint64_t>()},
                {"device_capacity_bytes", capacity},
                {"block_bytes", block},
                {"max_context", engine.at("max_context").get<std::uint64_t>()},
                {"max_concurrency", engine.at("max_concurrency").get<std::uint64_t>()},
                {"prefill_chunk", engine.value("prefill_chunk", std::uint64_t{0})},
                {"kv_cache", engine.value("kv_cache", std::string{})},
                {"speculative_backend", engine.value("speculative_backend", std::string{})},
                {"draft_tokens", engine.value("speculative_draft_window", std::uint64_t{0})},
                {"proposal_head", engine.value("proposal_head", std::string{})},
                {"vision", engine.value("vision", false)},
                {"prefix_reuse", engine.value("prefix_reuse", true)}};
    } catch (...) { return nullptr; }
}
} // namespace ninfer::supervisor
