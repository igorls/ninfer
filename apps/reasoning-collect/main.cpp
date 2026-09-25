#include "ninfer/engine.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Offline paired supervision through the product Engine. This program never trains or
// evaluates a Python copy of the backbone and never loads a production routing policy.
namespace {

using json = nlohmann::json;

// Every token that shapes the features or the labels is part of the profile; rows with
// different profiles never share a training set (reasoning_router.py load_rows).
constexpr const char* kBaseProfile =
    "qwen3.8-27b/nvfp4:fp8:chunk1024:cache-off:spec-none:medium:frontier-split";
constexpr std::uint32_t kMaxContext = 8192;

struct Settings {
    unsigned concurrency = 1;
    // Run the 2048-token action only when the 1024-token action's thinking cap fired.
    // Otherwise both actions decode the same greedy tokens under the same schedule, and
    // the 2048 action is recorded as derived from the 1024 observation.
    bool derive_2048 = false;
    // Only the direct action, re-observed for its answer-letter distribution; joined to a
    // paired collection of the same rows by reasoning_router.py load_confidence.
    bool direct_only = false;
};

std::string profile_for(const Settings& settings) {
    std::string profile = kBaseProfile;
    // Concurrent decode rounds change the batch composition each token sees, so labels
    // collected at C > 1 are a different, non-repeatable numerical profile.
    if (settings.concurrency > 1) { profile += ":c" + std::to_string(settings.concurrency); }
    if (settings.derive_2048) { profile += ":derive2048"; }
    if (settings.direct_only) { profile += ":direct-only"; }
    return profile;
}

// Removes an unterminated final line left by an interrupted run. Complete rows stay; the
// removed partial row is recollected.
void drop_partial_tail(const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size == 0) { return; }
    std::ifstream in(path, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    if (bytes.back() == '\n') { return; }
    const auto keep = bytes.rfind('\n') == std::string::npos ? 0 : bytes.rfind('\n') + 1;
    std::filesystem::resize_file(path, keep);
    std::cerr << "dropped an unterminated final outcome line (" << (bytes.size() - keep)
              << " bytes); that example will be recollected\n";
}

json run_action(ninfer::Engine& engine, const json& row, unsigned budget,
                std::vector<float>& features, json& result) {
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
    // The direct action's first answer position reports its distribution over the option
    // letters, the confidence a router can read after one decoded token.
    std::vector<std::string> letters;
    if (budget == 0) {
        request.execution.logprobs.enabled = true;
        request.execution.logprobs.top = 5;
        for (const auto& [letter, value] : row.at("mapping").items()) {
            const auto ids = engine.tokenize_text(letter);
            if (ids.size() != 1) { throw std::invalid_argument("option letter is not one token: " + letter); }
            letters.push_back(letter);
            request.execution.logprobs.candidates.push_back(ids.front());
        }
    }
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
    const auto& thinking = response.thinking;
    json action = {{"budget", budget}, {"content", response.content},
            {"reasoning", response.reasoning}, {"reasoning_tokens", response.reasoning_tokens},
            {"output_tokens", response.generated_token_ids.size()},
            {"finish_reason", static_cast<int>(response.finish_reason)},
            {"stopped", response.finish_reason == ninfer::FinishReason::StopToken ||
                        response.finish_reason == ninfer::FinishReason::StopString},
            {"thinking", {{"model_tokens", thinking.model_thinking_tokens},
                          {"injected_tokens", thinking.injected_tokens},
                          {"cap_applied", thinking.applied}}},
            {"seconds", response.timings.total_seconds}};
    if (budget == 0) {
        const auto first = std::find_if(response.token_logprobs.begin(), response.token_logprobs.end(),
                                         [](const ninfer::TokenLogprobs& entry) { return !entry.forced; });
        if (first == response.token_logprobs.end() || first->candidates.size() != letters.size()) {
            throw std::runtime_error("direct action lacks answer-letter logprobs");
        }
        json answer = json::object(), raw = json::object(), top = json::array();
        for (std::size_t i = 0; i < letters.size(); ++i) {
            answer[letters[i]] = first->candidates[i].logprob;
            raw[letters[i]] = first->candidates[i].raw_logprob;
        }
        for (const auto& entry : first->top) { top.push_back({engine.token_bytes(entry.token), entry.raw_logprob}); }
        action["answer_logprobs"] = std::move(answer);
        action["answer_raw_logprobs"] = std::move(raw);
        action["first_top"] = std::move(top);
    }
    return action;
}

json collect_row(ninfer::Engine& engine, const json& row, const Settings& settings,
                 const std::string& profile, const std::string& artifact,
                 const std::string& artifact_digest) {
    json result{{"schema", 1}, {"input", row}, {"artifact", artifact},
                {"artifact_sha256", artifact_digest}, {"profile", profile},
                {"actions", json::array()}};
    std::vector<float> features;
    result["actions"].push_back(run_action(engine, row, 0, features, result));
    if (settings.direct_only) {
        result["features"] = features;
        return result;
    }
    const json reasoned = run_action(engine, row, 1024, features, result);
    result["actions"].push_back(reasoned);
    // A 1024 run that stopped on its own without the cap firing never reached 1024 thinking
    // tokens and never hit its output limit, so the 2048 run would repeat it token for token.
    if (settings.derive_2048 && reasoned.at("stopped").get<bool>() &&
        !reasoned.at("thinking").at("cap_applied").get<bool>()) {
        json derived = reasoned;
        derived["budget"] = 2048;
        derived["derived_from_budget"] = 1024;
        result["actions"].push_back(std::move(derived));
    } else {
        result["actions"].push_back(run_action(engine, row, 2048, features, result));
    }
    result["features"] = features;
    return result;
}

} // namespace

int main(int argc, char** argv) try {
    Settings settings;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--concurrency" && i + 1 < argc) {
            settings.concurrency = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if (arg == "--derive-2048") {
            settings.derive_2048 = true;
        } else if (arg == "--direct-only") {
            settings.direct_only = true;
        } else {
            positional.push_back(arg);
        }
    }
    if (positional.size() != 4 || settings.concurrency < 1 || settings.concurrency > 8 ||
        (settings.direct_only && settings.derive_2048)) {
        std::cerr << "usage: ninfer-reasoning-collect model.ninfer requests.jsonl outcomes.jsonl "
                     "artifact-sha256 [--concurrency 1..8] [--derive-2048 | --direct-only]\n"
                     "Use reasoning_router.py collect to compute and validate the artifact digest.\n"
                     "Appends complete paired examples; matching existing IDs are resumed.\n";
        return 2;
    }
    const std::string artifact_path = positional[0];
    const std::filesystem::path requests_path = positional[1];
    const std::filesystem::path outcomes_path = positional[2];
    const std::string artifact_digest = positional[3];
    if (artifact_digest.size() != 64 || artifact_digest.find_first_not_of("0123456789abcdef") != std::string::npos) {
        throw std::invalid_argument("artifact SHA-256 must be 64 lowercase hex digits");
    }
    const std::string profile = profile_for(settings);
    drop_partial_tail(outcomes_path);
    std::map<std::string, json> completed;
    std::ifstream previous(outcomes_path);
    std::string line;
    while (std::getline(previous, line)) {
        if (line.empty()) { continue; }
        auto row = json::parse(line);
        if (!completed.emplace(row.at("input").at("id").get<std::string>(), row).second) {
            throw std::invalid_argument("duplicate completed ID");
        }
    }
    previous.close();
    std::ifstream input(requests_path);
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
                old->second.at("profile") != profile) {
                throw std::invalid_argument("resume input, artifact or profile differs for " + id);
            }
        } else { pending.push_back(std::move(row)); }
    }
    if (pending.empty()) { std::cout << "No pending examples\n"; return 0; }
    ninfer::EngineOptions options;
    options.artifact_path = artifact_path;
    options.max_context = kMaxContext;
    // Each worker keeps exactly one request in flight, so the pool holds every lane's full
    // context and admission never waits on KV or hits the pending deadline.
    options.max_concurrency = settings.concurrency;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(kMaxContext * settings.concurrency);
    options.pending_timeout_ms = 3'600'000;
    options.prefill_chunk = 1024;
    options.kv_cache = ninfer::KvCacheStorage::Fp8E4M3Row256;
    options.context_cache.enabled = false;
    options.speculative.backend = ninfer::SpeculativeBackend::None;
    ninfer::Engine engine(options);
    const auto loaded = engine.load_summary();
    if (loaded.model_id != "qwen3.8-27b" || loaded.weights_id != "nvfp4") {
        throw std::invalid_argument("collector requires qwen3.8-27b/nvfp4");
    }
    std::ofstream output(outcomes_path, std::ios::app | std::ios::binary);
    if (!output) { throw std::invalid_argument("cannot open outcomes"); }

    // Workers take whole rows; a row's actions run in sequence on its worker, and only a
    // complete, validated row reaches the single writer. The file order is completion order.
    std::atomic<std::size_t> next{0};
    std::atomic<bool> stop{false};
    std::mutex writer;
    std::exception_ptr failure;
    std::size_t written = 0;
    const auto worker = [&] {
        while (!stop.load()) {
            const std::size_t index = next.fetch_add(1);
            if (index >= pending.size()) { return; }
            try {
                const json result = collect_row(engine, pending[index], settings, profile,
                                                artifact_path, artifact_digest);
                const std::string serialized = result.dump(-1, ' ', false, json::error_handler_t::replace);
                std::lock_guard lock(writer);
                output << serialized << '\n';
                output.flush();
                if (!output) { throw std::runtime_error("writing outcomes failed"); }
                ++written;
                std::cout << pending[index].at("id").get<std::string>() << " complete (" << written
                          << "/" << pending.size() << ")\n" << std::flush;
            } catch (...) {
                std::lock_guard lock(writer);
                if (!failure) { failure = std::current_exception(); }
                stop.store(true);
                return;
            }
        }
    };
    std::vector<std::thread> workers;
    workers.reserve(settings.concurrency);
    for (unsigned i = 0; i < settings.concurrency; ++i) { workers.emplace_back(worker); }
    for (auto& thread : workers) { thread.join(); }
    if (failure) { std::rethrow_exception(failure); }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
