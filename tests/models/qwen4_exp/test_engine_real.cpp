#include "ninfer/engine.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Qwen4Exp (Qwen3.8-Flash-Next) Text execution through the public Engine on the real artifact:
// option admission, CUDA Graph replay against eager execution, batched decode against
// one-request decode, prefix reuse including a Host round trip of the complete continuation
// state, catalog turnover, token logprobs, structured output and causal scoring.
namespace {

const char* g_artifact = nullptr;

ninfer::EngineOptions base_options() {
    ninfer::EngineOptions options;
    options.artifact_path        = g_artifact;
    options.max_context          = 4096;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk        = 1024;
    options.max_concurrency      = 1;
    options.max_pending_requests = 8;
    return options;
}

ninfer::RequestOptions greedy(std::uint32_t outputs, bool reuse = true) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = outputs;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = reuse;
    request.stop.include_model_defaults       = false;
    return request;
}

ninfer::ChatMessage message(ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage out;
    out.role = role;
    out.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    return out;
}

ninfer::PromptInput user_prompt(std::string text, std::string session = {}) {
    ninfer::PromptInput input;
    input.messages.push_back(message(ninfer::ChatRole::User, std::move(text)));
    input.options.enable_thinking = false;
    if (!session.empty()) {
        input.context_cache.session_key = std::move(session);
        input.context_cache.retention   = ninfer::CacheRetentionHint::LiveSession;
    }
    return input;
}

ninfer::PromptInput continued(ninfer::PromptInput input, const ninfer::GenerationResult& reply,
                              std::string followup) {
    ninfer::ChatMessage assistant = message(ninfer::ChatRole::Assistant, reply.content);
    assistant.reasoning_content   = reply.reasoning;
    input.messages.push_back(std::move(assistant));
    input.messages.push_back(message(ninfer::ChatRole::User, std::move(followup)));
    return input;
}

const std::vector<std::string>& prompts() {
    static const std::vector<std::string> values{
        "Explain in two sentences why the sky is blue.",
        "Write a short Python function that returns the n-th Fibonacci number.",
        "List three prime numbers greater than one hundred and explain how you checked them.",
        "Translate to Portuguese: The library opens at nine and closes at five.",
        "What is the capital of Australia, and why is it not Sydney?",
        "Summarize the plot of Romeo and Juliet in one paragraph.",
        "Give two arguments for and against daylight saving time.",
        "Describe how a hash table resolves collisions.",
    };
    return values;
}

bool fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return false;
}

// Engines are constructed one at a time: each holds the complete Device weights.
template <class Body>
bool with_engine(ninfer::EngineOptions options, Body&& body) {
    ninfer::Engine engine(std::move(options));
    return body(engine);
}

bool rejects_backends_before_loading() {
    const auto rejected = [](ninfer::EngineOptions options, const char* expected) {
        const auto started = std::chrono::steady_clock::now();
        try {
            ninfer::Engine engine(std::move(options));
        } catch (const std::invalid_argument& error) {
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            const std::string text = error.what();
            if (text.find(expected) == std::string::npos) {
                return fail("unexpected rejection: " + text);
            }
            // The rejection precedes the 70 GiB weight upload.
            if (seconds > 30.0) { return fail("rejection came after the weight load"); }
            return true;
        }
        return fail(std::string("an Engine was constructed with ") + expected);
    };
    ninfer::EngineOptions vision = base_options();
    vision.enable_vision         = true;
    ninfer::EngineOptions mtp    = base_options();
    mtp.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
    mtp.speculative.draft_tokens = 3;
    return rejected(vision, "Vision") && rejected(mtp, "speculative");
}

// Greedy generation replays CUDA Graphs; the eager schedule is the same Op sequence, so the
// tokens are identical.
bool graph_matches_eager(std::vector<std::vector<ninfer::TokenId>>& graph_tokens) {
    const auto run = [&](bool graphs, std::vector<std::vector<ninfer::TokenId>>& out) {
        ninfer::EngineOptions options = base_options();
        options.use_cuda_graph        = graphs;
        options.context_cache.enabled = false;
        return with_engine(options, [&](ninfer::Engine& engine) {
            for (const std::string& text : prompts()) {
                const auto result = engine.generate(engine.prepare(user_prompt(text)), greedy(48));
                if (result.generated_token_ids.size() != 48) {
                    return fail("greedy request stopped early");
                }
                out.push_back(result.generated_token_ids);
            }
            std::cout << "sample: " << prompts().front() << " -> "
                      << engine.generate(engine.prepare(user_prompt(prompts().front())),
                                         greedy(48))
                             .content
                      << '\n';
            return true;
        });
    };
    std::vector<std::vector<ninfer::TokenId>> eager;
    if (!run(true, graph_tokens) || !run(false, eager)) { return false; }
    for (std::size_t i = 0; i < eager.size(); ++i) {
        if (eager[i] != graph_tokens[i]) {
            return fail("CUDA Graph decode differs from eager decode on prompt " +
                        std::to_string(i));
        }
    }
    return true;
}

// B rows decode through batched Op routes whose reductions differ from B=1, so a row may leave
// the B=1 trajectory only where B=1 itself was undecided: the first differing position must be a
// near tie of the B=1 distribution.
bool batched_matches_single(const std::vector<std::vector<ninfer::TokenId>>& single) {
    constexpr float kNearTie = 0.05F;
    std::vector<std::vector<ninfer::TokenLogprobs>> single_logprobs;
    {
        ninfer::EngineOptions options = base_options();
        options.context_cache.enabled = false;
        const bool ok                 = with_engine(options, [&](ninfer::Engine& engine) {
            for (const std::string& text : prompts()) {
                ninfer::RequestOptions request     = greedy(48);
                request.execution.logprobs.enabled = true;
                request.execution.logprobs.top     = 2;
                single_logprobs.push_back(
                    engine.generate(engine.prepare(user_prompt(text)), request).token_logprobs);
            }
            return true;
        });
        if (!ok) { return false; }
    }
    for (const std::uint32_t concurrency : {2U, 4U, 8U}) {
        ninfer::EngineOptions options = base_options();
        options.max_concurrency       = concurrency;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(4096U * concurrency);
        options.context_cache.enabled = false;
        const bool ok = with_engine(options, [&](ninfer::Engine& engine) {
            std::vector<ninfer::GenerationHandle> handles;
            for (std::uint32_t i = 0; i < concurrency; ++i) {
                handles.push_back(engine.submit(
                    engine.prepare(user_prompt(prompts()[i % prompts().size()])), greedy(48)));
            }
            for (std::uint32_t i = 0; i < concurrency; ++i) {
                const auto result   = handles[i].wait();
                const auto& tokens  = result.generated_token_ids;
                const auto& oracle  = single[i % prompts().size()];
                const auto& weights = single_logprobs[i % prompts().size()];
                const auto first    = std::mismatch(tokens.begin(), tokens.end(), oracle.begin());
                if (first.first == tokens.end()) { continue; }
                const std::size_t position = static_cast<std::size_t>(first.first - tokens.begin());
                const auto& top            = weights.at(position).top;
                const float gap = top.size() >= 2 ? top[0].raw_logprob - top[1].raw_logprob : 1e9F;
                std::cout << "B=" << concurrency << " row " << i << " leaves B=1 at " << position
                          << " (B=1 top-2 gap " << gap << ")\n";
                if (gap > kNearTie) {
                    return fail("batched decode diverged from B=1 away from a near tie");
                }
            }
            return true;
        });
        if (!ok) { return false; }
    }
    return true;
}

// The complete continuation state (GDN, QSA forming blocks, PLE history, KV with indexer block
// keys) survives a Host demotion and restore byte for byte: a resume from Host produces exactly
// the tokens of the same resume from a Device-resident checkpoint.
bool host_round_trip_is_exact() {
    std::string long_text;
    for (int i = 0; i < 300; ++i) { long_text += "alpha beta gamma "; }
    const auto scenario = [&](ninfer::EngineOptions options, bool expect_host,
                              std::vector<ninfer::TokenId>& tokens) {
        return with_engine(options, [&](ninfer::Engine& engine) {
            const ninfer::PromptInput first = user_prompt(long_text, "round-trip");
            const auto retained = engine.generate(engine.prepare(first), greedy(12));
            const ninfer::PromptInput followup = continued(first, retained, "Continue briefly.");
            const ninfer::RuntimeStats before  = engine.runtime_stats();
            const auto pressure = engine.generate(engine.prepare(followup), greedy(4, false));
            const ninfer::RuntimeStats demoted = engine.runtime_stats();
            const auto restored = engine.generate(engine.prepare(followup), greedy(24));
            const ninfer::RuntimeStats after   = engine.runtime_stats();
            if (pressure.generated_token_ids.size() != 4 || restored.reused_prompt_tokens == 0) {
                return fail("the follow-up did not reuse its retained prefix");
            }
            const bool through_host = demoted.state_d2h_count > before.state_d2h_count &&
                                      after.state_h2d_count > demoted.state_h2d_count &&
                                      after.main_kv_h2d_pages > demoted.main_kv_h2d_pages;
            if (through_host != expect_host) {
                return fail(std::string("the resume ") + (expect_host ? "did not go" : "went") +
                            " through Host");
            }
            tokens = restored.generated_token_ids;
            return true;
        });
    };
    ninfer::EngineOptions host                              = base_options();
    host.context_cache.device_state_slots                   = 1;
    host.context_cache.host_state_slots                     = 4;
    host.context_cache.host_kv_capacity_bytes               = 1ULL << 30U;
    host.context_cache.max_private_continuations            = 2;
    host.context_cache.max_shared_prefixes                  = 0;
    host.context_cache.max_long_anchors_per_continuation    = 0;
    ninfer::EngineOptions device                            = host;
    device.context_cache.device_state_slots                 = 4;
    device.context_cache.host_state_slots                   = 0;
    device.context_cache.host_kv_capacity_bytes             = 0;
    std::vector<ninfer::TokenId> through_host;
    std::vector<ninfer::TokenId> resident;
    if (!scenario(host, true, through_host) || !scenario(device, false, resident)) { return false; }
    if (through_host != resident) {
        return fail("a resume from Host differs from the same resume from Device");
    }
    return true;
}

// Endpoint, exact-hit and turn-closure reuse, and reuse that keeps working after the bounded
// private catalog has turned over many times.
bool reuse_survives_catalog_turnover() {
    ninfer::EngineOptions options                           = base_options();
    options.context_cache.device_state_slots                = 2;
    options.context_cache.host_state_slots                  = 0;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 0;
    options.context_cache.max_long_anchors_per_continuation = 0;
    return with_engine(options, [&](ninfer::Engine& engine) {
        // Exact-hit: the whole prompt is resident, so the first token comes from the stored
        // continuation hidden.
        const std::vector<ninfer::TokenId> prompt =
            engine.tokenize_text("The quick brown fox jumps over the lazy dog.");
        const auto base = engine.generate(engine.prepare_tokens(prompt), greedy(8));
        std::vector<ninfer::TokenId> frontier = prompt;
        frontier.insert(frontier.end(), base.generated_token_ids.begin(),
                        base.generated_token_ids.end() - 1);
        const auto exact = engine.generate(engine.prepare_tokens(frontier), greedy(2));
        if (exact.reused_prompt_tokens != frontier.size() ||
            exact.generated_token_ids.front() != base.generated_token_ids.back()) {
            return fail("exact-hit reuse did not continue the retained trajectory");
        }
        for (int round = 0; round < 6; ++round) {
            const std::string session = "turnover-" + std::to_string(round);
            const ninfer::PromptInput first =
                user_prompt(prompts()[static_cast<std::size_t>(round) % prompts().size()], session);
            const auto reply  = engine.generate(engine.prepare(first), greedy(16));
            const auto second = engine.generate(
                engine.prepare(continued(first, reply, "And in one word?")), greedy(4));
            if (second.reused_prompt_tokens == 0) {
                return fail("round " + std::to_string(round) +
                            " lost prefix reuse after the catalog turned over");
            }
        }
        return true;
    });
}

bool logprobs_and_structured_output() {
    ninfer::EngineOptions options = base_options();
    return with_engine(options, [&](ninfer::Engine& engine) {
        ninfer::RequestOptions request     = greedy(16);
        request.execution.logprobs.enabled = true;
        request.execution.logprobs.top     = 5;
        const auto result = engine.generate(engine.prepare(user_prompt(prompts()[4])), request);
        if (result.token_logprobs.size() != result.generated_token_ids.size()) {
            return fail("token logprobs do not cover every generated token");
        }
        for (std::size_t i = 0; i < result.token_logprobs.size(); ++i) {
            const auto& entry = result.token_logprobs[i];
            if (entry.top.empty() || entry.top.front().token != result.generated_token_ids[i] ||
                !std::isfinite(entry.sampled.raw_logprob) || entry.sampled.raw_logprob > 0.0F) {
                return fail("a greedy token is not the most likely token of its readout");
            }
        }
        ninfer::RequestOptions device_readout     = greedy(16);
        device_readout.execution.logprobs.enabled = true;
        const auto device = engine.generate(engine.prepare(user_prompt(prompts()[4])),
                                            device_readout);
        for (std::size_t i = 0; i < device.token_logprobs.size(); ++i) {
            if (std::abs(device.token_logprobs[i].sampled.raw_logprob -
                         result.token_logprobs[i].sampled.raw_logprob) > 1e-3F) {
                return fail("device and host logprob readouts disagree");
            }
        }

        ninfer::RequestOptions structured = greedy(96);
        structured.stop.include_model_defaults           = true;
        structured.execution.structured_output.kind      = ninfer::StructuredOutputKind::JsonSchema;
        structured.execution.structured_output.schema    = R"({"type":"object",
            "properties":{"city":{"type":"string"},"population":{"type":"integer"}},
            "required":["city","population"],"additionalProperties":false})";
        const auto json = engine.generate(
            engine.prepare(user_prompt("Name a large city and its population as JSON.")),
            structured);
        try {
            const auto parsed = nlohmann::json::parse(json.content);
            if (!parsed.contains("city") || !parsed.at("population").is_number_integer()) {
                return fail("structured output does not follow its schema: " + json.content);
            }
        } catch (const std::exception&) {
            return fail("structured output is not JSON: " + json.content);
        }
        return true;
    });
}

// The scoring Program evaluates the same prefill schedule as generation's prompt readout; their
// target log probabilities agree within the mixer's decode/prefill route difference.
bool causal_score_matches_prompt_readout() {
    std::vector<ninfer::TokenId> tokens;
    std::vector<float> readout;
    {
        ninfer::EngineOptions options = base_options();
        const bool ok                 = with_engine(options, [&](ninfer::Engine& engine) {
            tokens = engine.tokenize_text(
                "In 1969 the Apollo 11 mission landed two astronauts on the Moon. They collected "
                "samples, deployed experiments and returned safely to Earth four days later.");
            ninfer::RequestOptions request     = greedy(1, false);
            request.execution.logprobs.enabled = true;
            for (std::uint32_t p = 0; p + 1 < tokens.size(); ++p) {
                request.execution.logprobs.prompt_positions.push_back(p);
            }
            const auto result = engine.generate(engine.prepare_tokens(tokens), request);
            for (const auto& entry : result.prompt_logprobs) {
                readout.push_back(entry.sampled.raw_logprob);
            }
            return readout.size() + 1 == tokens.size();
        });
        if (!ok) { return fail("prompt readout did not cover the prompt"); }
    }
    ninfer::EngineOptions options = base_options();
    options.purpose               = ninfer::EnginePurpose::CausalScoring;
    return with_engine(options, [&](ninfer::Engine& engine) {
        const std::vector<float> scores = engine.score_tokens(tokens, 1);
        if (scores.size() != readout.size()) { return fail("causal scores have the wrong count"); }
        double worst = 0.0;
        for (std::size_t i = 0; i < scores.size(); ++i) {
            worst = std::max(worst, std::abs(static_cast<double>(scores[i]) - readout[i]));
        }
        std::cout << "causal score vs prompt readout: max |dlogprob| = " << worst << '\n';
        return worst <= 0.05 ? true : fail("causal scores differ from the prompt readout");
    });
}

} // namespace

int main() {
    g_artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (g_artifact == nullptr || *g_artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        std::vector<std::vector<ninfer::TokenId>> single;
        const std::vector<std::pair<const char*, std::function<bool()>>> cases{
            {"rejects unavailable backends", rejects_backends_before_loading},
            {"graph matches eager", [&] { return graph_matches_eager(single); }},
            {"batched matches single", [&] { return batched_matches_single(single); }},
            {"host round trip", host_round_trip_is_exact},
            {"reuse survives catalog turnover", reuse_survives_catalog_turnover},
            {"logprobs and structured output", logprobs_and_structured_output},
            {"causal score", causal_score_matches_prompt_readout},
        };
        for (const auto& [name, body] : cases) {
            std::cout << "== " << name << std::endl;
            if (!body()) { return 1; }
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
