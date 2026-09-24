#include "ninfer/engine.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

// Offline paired supervision through the product Engine. This program never trains or
// evaluates a Python copy of the backbone and never loads a production routing policy.
int main(int argc, char** argv) try {
    if (argc != 5) {
        std::cerr << "usage: ninfer-reasoning-collect model.ninfer requests.jsonl outcomes.jsonl artifact-sha256\n"
                     "Use reasoning_router.py collect to compute and validate the artifact digest.\n"
                     "Appends complete paired examples; matching existing IDs are resumed.\n";
        return 2;
    }
    using json = nlohmann::json;
    const std::string artifact_digest = argv[4];
    if (artifact_digest.size() != 64 || artifact_digest.find_first_not_of("0123456789abcdef") != std::string::npos) {
        throw std::invalid_argument("artifact SHA-256 must be 64 lowercase hex digits");
    }
    std::map<std::string, json> completed;
    std::ifstream previous(argv[3]);
    std::string line;
    while (std::getline(previous, line)) {
        if (line.empty()) { continue; }
        auto row = json::parse(line);
        if (!completed.emplace(row.at("input").at("id").get<std::string>(), row).second) {
            throw std::invalid_argument("duplicate completed ID");
        }
    }
    std::ifstream input(argv[2]);
    if (!input) { throw std::invalid_argument("cannot open requests"); }
    std::vector<json> pending;
    std::map<std::string, bool> ids;
    while (std::getline(input, line)) {
        if (line.empty()) { continue; }
        auto row = json::parse(line);
        const auto id = row.at("id").get<std::string>();
        if (!ids.emplace(id, true).second) { throw std::invalid_argument("duplicate input ID"); }
        if (const auto old = completed.find(id); old != completed.end()) {
            if (old->second.at("input") != row || old->second.at("artifact_sha256") != artifact_digest ||
                old->second.at("profile") != "qwen3.8-27b/nvfp4:fp8:chunk1024:cache-off:spec-none:medium:frontier-split") {
                throw std::invalid_argument("resume input or artifact differs for " + id);
            }
        } else { pending.push_back(std::move(row)); }
    }
    if (pending.empty()) { std::cout << "No pending examples\n"; return 0; }
    ninfer::EngineOptions options;
    options.artifact_path = argv[1];
    options.max_context = 8192;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(8192);
    options.prefill_chunk = 1024;
    options.kv_cache = ninfer::KvCacheStorage::Fp8E4M3Row256;
    options.context_cache.enabled = false;
    options.speculative.backend = ninfer::SpeculativeBackend::None;
    ninfer::Engine engine(options);
    const auto loaded = engine.load_summary();
    if (loaded.model_id != "qwen3.8-27b" || loaded.weights_id != "nvfp4") {
        throw std::invalid_argument("collector requires qwen3.8-27b/nvfp4");
    }
    std::ofstream output(argv[3], std::ios::app);
    if (!output) { throw std::invalid_argument("cannot open outcomes"); }
    for (const auto& row : pending) {
        json result{{"schema", 1}, {"input", row}, {"artifact", argv[1]},
                    {"artifact_sha256", artifact_digest},
                    {"profile", "qwen3.8-27b/nvfp4:fp8:chunk1024:cache-off:spec-none:medium:frontier-split"},
                    {"actions", json::array()}};
        std::vector<float> features;
        for (const unsigned budget : {0U, 1024U, 2048U}) {
            ninfer::PromptInput prompt;
            prompt.options.enable_thinking = budget != 0;
            // Medium injects no effort instructions, preserving one prefix for all actions.
            if (budget != 0) { prompt.options.reasoning_effort = ninfer::ReasoningEffort::Medium; }
            for (const auto& message : row.at("messages")) {
                const auto role = message.at("role").get<std::string>();
                if (role != "system" && role != "user") {
                    throw std::invalid_argument("collector accepts system/user messages only");
                }
                ninfer::ChatMessage msg;
                msg.role = role == "system" ? ninfer::ChatRole::System : ninfer::ChatRole::User;
                msg.parts.push_back({.kind = ninfer::MessagePartKind::Text,
                                     .text = message.at("content").get<std::string>()});
                prompt.messages.push_back(std::move(msg));
            }
            ninfer::RequestOptions request;
            request.execution.sampling.temperature = 0;
            request.execution.sampling.presence_penalty = 0;
            request.execution.sampling.frequency_penalty = 0;
            request.execution.sampling.repetition_penalty = 1;
            request.execution.requested_output_tokens = budget == 0 ? 64 : budget + 256;
            request.execution.capture_reasoning_features = true;
            request.execution.allow_prefix_reuse = false;
            request.execution.allow_prefix_publication = false;
            if (budget) { request.execution.thinking.budget = budget; }
            const auto response = engine.generate(engine.prepare(std::move(prompt)), request);
            // Exhausting the declared action budget is a measured failure. Cancellation and
            // context exhaustion are incomplete observations and must be retried.
            if (response.finish_reason != ninfer::FinishReason::StopToken &&
                response.finish_reason != ninfer::FinishReason::StopString &&
                response.finish_reason != ninfer::FinishReason::OutputLimit) {
                throw std::runtime_error("incomplete action: cancellation or context exhaustion");
            }
            if (response.reused_prompt_tokens != 0 || response.reasoning_features.size() != 5120 ||
                !std::all_of(response.reasoning_features.begin(), response.reasoning_features.end(),
                             [](float value) { return std::isfinite(value); })) {
                throw std::runtime_error("collector requires finite 5120-wide uncached Qwen features");
            }
            if (features.empty()) { features = response.reasoning_features; }
            else if (features != response.reasoning_features) {
                throw std::runtime_error("action suffix changed the common-prefix features");
            }
            result["feature_frontier"] = *response.prompt.reasoning_frontier;
            result["actions"].push_back({{"budget", budget}, {"content", response.content},
                {"reasoning", response.reasoning}, {"reasoning_tokens", response.reasoning_tokens},
                {"output_tokens", response.generated_token_ids.size()},
                {"finish_reason", static_cast<int>(response.finish_reason)},
                {"stopped", response.finish_reason == ninfer::FinishReason::StopToken ||
                            response.finish_reason == ninfer::FinishReason::StopString},
                {"seconds", response.timings.total_seconds}});
        }
        result["features"] = features;
        output << result.dump() << '\n';
        output.flush();
        if (!output) { throw std::runtime_error("writing outcomes failed"); }
        std::cout << row.at("id").get<std::string>() << " complete\n" << std::flush;
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
