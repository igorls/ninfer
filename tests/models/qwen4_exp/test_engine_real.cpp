#include "ninfer/engine.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Qwen4Exp (Qwen3.8-Flash-Next) execution through the public Engine on the real artifact:
// option admission, CUDA Graph replay against eager execution, batched decode against
// one-request decode, prefix reuse including a Host round trip of the complete continuation
// state, catalog turnover, token logprobs, structured output, causal scoring and prefill-chunk
// invariance.
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
    if (!session.empty()) { input.context_cache.session_key = std::move(session); }
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
    ninfer::EngineOptions mtp    = base_options();
    mtp.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
    mtp.speculative.draft_tokens = 99;
    return rejected(mtp, "MTP");
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
                      << engine.generate(engine.prepare(user_prompt(prompts().front())), greedy(48))
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
bool batched_matches_single(std::vector<std::vector<ninfer::TokenId>> single) {
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
                const auto result = engine.generate(engine.prepare(user_prompt(text)), request);
                single_logprobs.push_back(result.token_logprobs);
                if (single.size() < prompts().size()) {
                    single.push_back(result.generated_token_ids);
                }
            }
            return true;
        });
        if (!ok) { return false; }
    }
    bool passed = true;
    for (const std::uint32_t concurrency : {2U, 4U, 8U}) {
        ninfer::EngineOptions options = base_options();
        options.max_concurrency       = concurrency;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(4096U * concurrency);
        options.context_cache.enabled = false;
        const bool ok                 = with_engine(options, [&](ninfer::Engine& engine) {
            std::vector<ninfer::GenerationHandle> handles;
            for (std::uint32_t i = 0; i < concurrency; ++i) {
                handles.push_back(engine.submit(
                    engine.prepare(user_prompt(prompts()[i % prompts().size()])), greedy(48)));
            }
            for (std::uint32_t i = 0; i < concurrency; ++i) {
                const auto result  = handles[i].wait();
                const auto& tokens = result.generated_token_ids;
                if (tokens.size() != 48) {
                    return fail("batched request did not meet its token budget");
                }
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
                    passed = fail("batched decode diverged from B=1 away from a near tie");
                }
            }
            return true;
        });
        if (!ok) { return false; }
    }
    return passed;
}

// The complete continuation state (GDN, QSA forming blocks, PLE history, KV with indexer block
// keys) survives a Host demotion and restore byte for byte: a resume from Host produces exactly
// the tokens of the same resume from a Device-resident checkpoint.
bool host_round_trip_is_exact(bool mtp = false) {
    std::string long_text;
    for (int i = 0; i < 300; ++i) { long_text += "alpha beta gamma "; }
    const auto scenario = [&](ninfer::EngineOptions options, bool expect_host,
                              std::vector<ninfer::TokenId>& tokens) {
        return with_engine(options, [&](ninfer::Engine& engine) {
            const ninfer::RuntimeStats before  = engine.runtime_stats();
            const ninfer::PromptInput first    = user_prompt(long_text, "round-trip");
            const auto retained                = engine.generate(engine.prepare(first), greedy(12));
            const ninfer::PromptInput followup = continued(first, retained, "Continue briefly.");
            auto pressure_ids                  = engine.tokenize_text("delta epsilon theta ");
            const auto pressure_seed           = pressure_ids;
            while (pressure_ids.size() < 3300) {
                pressure_ids.insert(pressure_ids.end(), pressure_seed.begin(), pressure_seed.end());
            }
            pressure_ids.resize(3300);
            const auto pressure =
                engine.generate(engine.prepare_tokens(pressure_ids), greedy(4, false));
            const ninfer::RuntimeStats demoted = engine.runtime_stats();
            const auto restored = engine.generate(engine.prepare(followup), greedy(24));
            const ninfer::RuntimeStats after = engine.runtime_stats();
            if (pressure.generated_token_ids.size() != 4 || restored.reused_prompt_tokens == 0) {
                return fail("the follow-up did not reuse its retained prefix");
            }
            const bool through_host =
                demoted.state_d2h_count > before.state_d2h_count &&
                after.state_h2d_count > demoted.state_h2d_count &&
                after.main_kv_h2d_pages > demoted.main_kv_h2d_pages &&
                (!mtp || after.backend_kv_h2d_pages > demoted.backend_kv_h2d_pages);
            std::cout << "restore expected_host=" << expect_host
                      << " d2h=" << demoted.state_d2h_count - before.state_d2h_count
                      << " h2d=" << after.state_h2d_count - demoted.state_h2d_count
                      << " kv_h2d=" << after.main_kv_h2d_pages - demoted.main_kv_h2d_pages
                      << std::endl;
            if (through_host != expect_host) {
                return fail(std::string("the resume ") + (expect_host ? "did not go" : "went") +
                            " through Host");
            }
            tokens = restored.generated_token_ids;
            return true;
        });
    };
    ninfer::EngineOptions host             = base_options();
    host.context_cache.device_state_slots  = 0;
    host.context_cache.host_capacity_bytes = 1ULL << 30U;
    if (mtp) {
        host.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        host.speculative.draft_tokens = 3;
    }
    ninfer::EngineOptions device             = host;
    device.context_cache.device_state_slots  = 4;
    device.max_context                       = 8192;
    device.kv_capacity                       = ninfer::KvCapacityPolicy::explicit_capacity(8192);
    device.context_cache.host_capacity_bytes = 0;
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
    ninfer::EngineOptions options             = base_options();
    options.context_cache.device_state_slots  = 2;
    options.context_cache.host_capacity_bytes = 0;
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
            auto chat_request                        = greedy(32);
            chat_request.stop.include_model_defaults = true;
            const auto reply  = engine.generate(engine.prepare(first), chat_request);
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

bool logprobs_and_structured_output(bool mtp = false) {
    ninfer::EngineOptions options = base_options();
    if (mtp) {
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 5;
    }
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
        const auto device =
            engine.generate(engine.prepare(user_prompt(prompts()[4])), device_readout);
        for (std::size_t i = 0; i < device.token_logprobs.size(); ++i) {
            if (std::abs(device.token_logprobs[i].sampled.raw_logprob -
                         result.token_logprobs[i].sampled.raw_logprob) > 1e-3F) {
                return fail("device and host logprob readouts disagree");
            }
        }

        ninfer::RequestOptions structured      = greedy(96);
        structured.stop.include_model_defaults = true;
        structured.constraint = ninfer::OutputConstraint::json_schema(R"({"type":"object",
            "properties":{"city":{"type":"string"},"population":{"type":"integer"}},
            "required":["city","population"],"additionalProperties":false})");
        const auto json       = engine.generate(
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

// Changing only future tokens must not alter a fixed predictor prefix. Keep the total
// extent fixed so this checks causality, including QSA's sparse boundary, without changing
// the permitted activation-quantization route at a GEMM shape boundary.
bool causal_prefix_independence() {
    auto options          = base_options();
    options.purpose       = ninfer::EnginePurpose::CausalScoring;
    options.prefill_chunk = 4096;
    return with_engine(options, [&](ninfer::Engine& engine) {
        const auto vocabulary = engine.tokenize_text(
            "The telescope records distant galaxies. A library stores books and historical maps. ");
        double worst = 0.0;
        for (const auto [extent, prefix] :
             {std::pair{64U, 31U}, std::pair{1024U, 257U}, std::pair{2200U, 2053U}}) {
            std::vector<ninfer::TokenId> first(extent), second(extent);
            for (std::uint32_t i = 0; i < extent; ++i) {
                first[i]  = vocabulary[i % vocabulary.size()];
                second[i] = i < prefix ? first[i] : vocabulary[(i + 7) % vocabulary.size()];
            }
            const auto a = engine.score_tokens(first, 1);
            const auto b = engine.score_tokens(second, 1);
            if (a.size() != extent - 1 || b.size() != a.size()) {
                return fail("causal prefix score count is invalid");
            }
            for (std::uint32_t i = 0; i + 1 < prefix; ++i) {
                worst = std::max(worst, std::abs(static_cast<double>(a[i]) - b[i]));
            }
        }
        std::cout << "causal prefix independence: max |dlogprob| = " << worst << '\n';
        return worst <= 0.05 ? true : fail("future tokens alter the predictor prefix");
    });
}

// The prefill chunk is a scheduling choice: the same tokens must score bit-identically whether
// they are prefilled in one chunk or several, including past QSA's sparse boundary. Every chunk
// here, the final one included, stays above the small-chunk routes (the A16 MoE below 256
// tokens and the small-T projection tiles through 288), which a short final chunk may still take.
bool prefill_chunk_invariance() {
    static const std::vector<std::string> sentences{
        "The telescope records distant galaxies across the southern sky.",
        "A library stores books, letters and historical maps of the harbor.",
        "Engineers measured the bridge after the winter storms had passed.",
        "The recipe calls for flour, two eggs and a pinch of salt.",
        "Migrating birds follow the coastline before crossing the strait."};
    std::vector<ninfer::TokenId> tokens;
    std::vector<float> reference;
    bool passed = true;
    for (const std::uint32_t chunk : {4096U, 1024U, 1536U}) {
        auto options          = base_options();
        options.purpose       = ninfer::EnginePurpose::CausalScoring;
        options.prefill_chunk = chunk;
        passed =
            with_engine(
                options,
                [&](ninfer::Engine& engine) {
                    if (tokens.empty()) {
                        std::string text;
                        for (int i = 0; tokens.size() < 3600; ++i) {
                            text += "Entry " + std::to_string((i * 7919) % 10007) + ": " +
                                    sentences[static_cast<std::size_t>(i) % sentences.size()] + ' ';
                            if (i % 16 == 15) { tokens = engine.tokenize_text(text); }
                        }
                        tokens.resize(3600);
                    }
                    const auto scores = engine.score_tokens(tokens, 1);
                    if (scores.size() != tokens.size() - 1) {
                        return fail("chunked score count is invalid");
                    }
                    if (reference.empty()) {
                        reference.assign(scores.begin(), scores.end());
                        return true;
                    }
                    std::size_t differing = 0;
                    double worst          = 0.0;
                    for (std::size_t i = 0; i < scores.size(); ++i) {
                        if (scores[i] != reference[i]) {
                            ++differing;
                            worst = std::max(
                                worst, std::abs(static_cast<double>(scores[i]) - reference[i]));
                        }
                    }
                    std::cout << "prefill chunk " << chunk << " vs 4096: " << differing << " of "
                              << scores.size() << " scores differ, max |dlogprob| = " << worst
                              << '\n';
                    return differing == 0 ? true : fail("prefill chunking changes the scores");
                }) &&
            passed;
    }
    return passed;
}

bool causal_hidden_readout() {
    ninfer::EngineOptions options = base_options();
    options.purpose               = ninfer::EnginePurpose::CausalScoring;
    options.prefill_chunk         = 256;
    return with_engine(options, [&](ninfer::Engine& engine) {
        std::string text;
        for (int i = 0; i < 300; ++i) { text += "The sky is blue. "; }
        const auto tokens = engine.tokenize_text(text);
        ninfer::CausalScoreReadout capture;
        capture.capture_hidden_rows = true;
        const auto full             = engine.score_tokens(tokens, 1, capture);
        const auto plain            = engine.score_tokens(tokens, 1, {});
        if (full.hidden_size != 2560 ||
            full.hidden_rows.size() != (tokens.size() - 1) * full.hidden_size ||
            plain.hidden_size != 0 || !plain.hidden_rows.empty() ||
            full.target_logprobs != plain.target_logprobs) {
            return fail("hidden capture changes scoring or returns the wrong row geometry");
        }
        // Offset crosses both the score tile and a prefill chunk; captured rows must still
        // correspond to predictors of the requested targets, including the final partial tile.
        constexpr std::uint32_t first_target = 1030;
        if (tokens.size() <= first_target) { return fail("hidden-readout prompt is too short"); }
        const auto suffix = engine.score_tokens(tokens, first_target, capture);
        if (suffix.hidden_size != full.hidden_size ||
            suffix.hidden_rows.size() != (tokens.size() - first_target) * full.hidden_size ||
            !std::equal(suffix.hidden_rows.begin(), suffix.hidden_rows.end(),
                        full.hidden_rows.begin() + (first_target - 1) * full.hidden_size)) {
            return fail("hidden readout rows do not follow the scored-target offset");
        }
        return true;
    });
}

ninfer::MessagePart image_part(bool blue) {
    constexpr int side       = 112;
    const std::string header = "P6\n112 112\n255\n";
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = blue ? "blue.ppm" : "red.ppm";
    image.media.bytes.assign(header.begin(), header.end());
    for (int i = 0; i < side * side; ++i) {
        image.media.bytes.push_back(blue ? 0 : 255);
        image.media.bytes.push_back(0);
        image.media.bytes.push_back(blue ? 255 : 0);
    }
    return image;
}

std::string color_readout(std::string content) {
    std::transform(content.begin(), content.end(), content.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return content;
}

bool represents_color(const std::string& content, bool blue) {
    const auto normalized = color_readout(content);
    // A color name and its exact RGB hex code represent the same fixture pixel.
    return normalized.find(blue ? "blue" : "red") != std::string::npos ||
           normalized.find(blue ? "#0000ff" : "#ff0000") != std::string::npos;
}

bool video_color_sequence(ninfer::Engine& engine) {
    const auto path = std::filesystem::path(__FILE__).parent_path() / "fixtures/red-blue.mp4";
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { return fail("generated video fixture is unavailable"); }
    ninfer::MessagePart video;
    video.kind              = ninfer::MessagePartKind::Media;
    video.media.kind        = ninfer::MediaKind::Video;
    video.media.media_type  = "video/mp4";
    video.media.source_name = "red-blue.mp4";
    video.media.bytes.assign(std::istreambuf_iterator<char>(stream), {});
    auto input = user_prompt("Describe the two colors in this video in their order of appearance.");
    input.messages.front().parts.insert(input.messages.front().parts.begin(), std::move(video));
    auto request                        = greedy(64, false);
    request.stop.include_model_defaults = true;
    const auto result                   = engine.generate(engine.prepare(input), request);
    std::cout << "Vision video=" << result.content << std::endl;
    const auto content = color_readout(result.content);
    const auto red     = content.find("red");
    const auto blue    = content.find("blue");
    if (!result.prompt.has_media || red == std::string::npos || blue == std::string::npos ||
        red > blue) {
        return fail("video did not preserve the red-to-blue temporal sequence");
    }
    const auto memory = engine.memory_summary();
    return memory.vision_workspace && memory.vision_workspace->handoff_active_bytes == 0 &&
           memory.workspace_logical_peak_bytes <= memory.workspace.capacity_bytes;
}

bool vision_generation_and_reuse() {
    auto options          = base_options();
    options.enable_vision = true;
    options.prefill_chunk = 128;
    return with_engine(options, [&](ninfer::Engine& engine) {
        engine.reset_memory_peaks();
        const auto before = engine.memory_summary();
        if (!before.vision_workspace || before.vision_workspace->handoff_active_bytes != 0 ||
            before.vision_workspace->handoff_peak_bytes != 0) {
            return fail("Vision startup does not expose its bounded handoff");
        }
        auto input = user_prompt("Name the dominant color in this image.", "flash-vision");
        input.messages.front().parts.insert(input.messages.front().parts.begin(),
                                            image_part(false));
        auto request                        = greedy(24);
        request.stop.include_model_defaults = true;
        const auto first                    = engine.generate(engine.prepare(input), request);
        if (!first.prompt.has_media || first.generated_token_ids.empty() || first.content.empty()) {
            return fail("Vision request did not complete through Engine");
        }
        auto followup = continued(input, first, "Name the color in the second image.");
        followup.messages.back().parts.insert(followup.messages.back().parts.begin(),
                                              image_part(true));
        const auto reused                    = engine.generate(engine.prepare(followup), request);
        request.execution.allow_prefix_reuse = false;
        const auto cold                      = engine.generate(engine.prepare(followup), request);
        std::cout << "Vision red=" << first.content << " blue=" << reused.content
                  << " reused=" << reused.reused_prompt_tokens << '\n';
        if (!represents_color(first.content, false) ||
            !represents_color(reused.content, true) ||
            !represents_color(cold.content, true)) {
            return fail("Vision image colors differ from their represented input");
        }
        if (reused.reused_prompt_tokens == 0 || !reused.prompt.has_media ||
            reused.generated_token_ids.empty() || cold.generated_token_ids.empty() ||
            reused.generated_token_ids.front() != cold.generated_token_ids.front()) {
            return fail("Vision append/reuse differs from cold execution at the first token");
        }
        const auto after = engine.memory_summary();
        if (!after.vision_workspace || after.vision_workspace->handoff_active_bytes != 0 ||
            after.vision_workspace->handoff_peak_bytes == 0 ||
            after.vision_workspace->handoff_peak_bytes >
                after.vision_workspace->handoff_capacity_bytes ||
            after.workspace_logical_peak_bytes > after.workspace.capacity_bytes ||
            after.workspace.capacity_bytes != before.workspace.capacity_bytes) {
            return fail("Vision execution escaped its startup workspace or retained a handoff");
        }
        return video_color_sequence(engine);
    });
}

bool mtp_generation_and_reuse() {
    auto options                     = base_options();
    options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 3;
    options.use_cuda_graph           = false;
    options.enable_vision            = true;
    options.prefill_chunk            = 128;
    return with_engine(options, [&](ninfer::Engine& engine) {
        auto request                        = greedy(48);
        request.stop.include_model_defaults = true;
        auto input =
            user_prompt("Count from one to ten, separating the numbers with commas.", "flash-mtp");
        const auto first = engine.generate(engine.prepare(input), request);
        std::cout << "MTP result=" << first.content << " rounds=" << first.speculative.rounds
                  << " drafted=" << first.speculative.drafted_tokens
                  << " accepted=" << first.speculative.accepted_tokens << std::endl;
        if (first.generated_token_ids.empty() || !first.speculative.enabled ||
            first.speculative.rounds == 0 || first.speculative.accepted_tokens == 0) {
            return fail("MTP did not license any drafts through the Engine");
        }
        auto followup     = continued(input, first, "Now count backwards from ten to one.");
        const auto reused = engine.generate(engine.prepare(followup), request);
        request.execution.allow_prefix_reuse = false;
        const auto cold                      = engine.generate(engine.prepare(followup), request);
        if (reused.reused_prompt_tokens == 0 || reused.generated_token_ids.empty() ||
            cold.generated_token_ids.empty() ||
            reused.generated_token_ids.front() != cold.generated_token_ids.front()) {
            return fail("MTP continuation differs from cold prefill at the first token");
        }
        auto visual = user_prompt("Name the dominant color in this image.");
        visual.messages.front().parts.insert(visual.messages.front().parts.begin(),
                                             image_part(false));
        const auto image = engine.generate(engine.prepare(visual), request);
        std::cout << "MTP Vision result=" << image.content << std::endl;
        if (!represents_color(image.content, false) ||
            !image.prompt.has_media) {
            return fail("MTP Vision execution failed its represented color input");
        }
        const auto memory = engine.memory_summary();
        return memory.workspace_logical_peak_bytes <= memory.workspace.capacity_bytes &&
               video_color_sequence(engine);
    });
}

bool mtp_batch_target_parity() {
    auto options                  = base_options();
    options.max_context           = 512;
    options.max_concurrency       = 8;
    options.prefill_chunk         = 128;
    options.context_cache.enabled = false;
    std::vector<ninfer::GenerationResult> reference;
    // The oracle here is the ordinary target route, not MTP with zero accepted drafts.
    // Keep only one resident model; each Engine is destroyed before the next is loaded.
    if (!with_engine(options, [&](ninfer::Engine& engine) {
            for (const auto& text : prompts()) {
                auto request                       = greedy(96, false);
                request.execution.logprobs.enabled = true;
                request.execution.logprobs.top     = 2;
                reference.push_back(engine.generate(engine.prepare(user_prompt(text)), request));
            }
            // Diagnostic control: ordinary B=8 isolates batching from speculation.
            std::vector<ninfer::GenerationHandle> controls;
            for (const auto& text : prompts()) {
                controls.push_back(engine.submit(engine.prepare(user_prompt(text)),
                                                  greedy(96, false)));
            }
            for (std::size_t row = 0; row < controls.size(); ++row) {
                const auto result = controls[row].wait();
                for (std::size_t i = 0; i < result.generated_token_ids.size(); ++i) {
                    if (result.generated_token_ids[i] == reference[row].generated_token_ids[i])
                        continue;
                    const auto& top = reference[row].token_logprobs[i].top;
                    std::cout << "ordinary B=8 control row=" << row << " first divergence=" << i
                              << " reference gap=" << top[0].raw_logprob - top[1].raw_logprob
                              << " batched=" << result.generated_token_ids[i]
                              << " single=" << reference[row].generated_token_ids[i] << std::endl;
                    break;
                }
            }
            return true;
        }))
        return false;
    bool ok = true;
    for (const auto drafts : {1U, 3U, 5U}) {
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = drafts;
        ok                               = with_engine(
                 options,
                 [&](ninfer::Engine& engine) {
                     std::vector<ninfer::PreparedPrompt> prepared;
                     for (const auto& text : prompts())
                         prepared.push_back(engine.prepare(user_prompt(text)));
                     std::vector<ninfer::GenerationHandle> handles;
                     for (auto& prompt : prepared)
                         handles.push_back(engine.submit(std::move(prompt), greedy(96, false)));
                     bool matched = true;
                     for (std::size_t row = 0; row < handles.size(); ++row) {
                         const auto result = handles[row].wait();
                         if (result.speculative.rounds == 0 ||
                             result.generated_token_ids.size() != 96) {
                             matched =
                                 fail("MTP batch target comparison did not execute speculation") &&
                                 matched;
                             continue;
                         }
                         for (std::size_t i = 0; i < result.generated_token_ids.size(); ++i) {
                             if (result.generated_token_ids[i] ==
                                 reference[row].generated_token_ids[i])
                                 continue;
                             const auto& top = reference[row].token_logprobs[i].top;
                             const auto gap  = top[0].raw_logprob - top[1].raw_logprob;
                             std::cout << "MTP K=" << drafts << " target row=" << row
                                       << " first divergence=" << i << " reference gap=" << gap
                                       << '\n';
                             if (gap > 0.05F)
                                 matched = fail("MTP verification changes a target decision away "
                                                                                                            "from a tie") &&
                                           matched;
                             break;
                         }
                     }
                     return matched;
                 }) &&
             ok;
    }
    return ok;
}

bool mtp_graphs_and_ragged_batches() {
    for (const auto drafts : {1U, 3U, 5U}) {
        std::vector<ninfer::TokenId> eager;
        for (const bool graphs : {false, true}) {
            auto options            = base_options();
            options.max_context     = 512;
            options.max_concurrency = graphs ? 8 : 1;
            options.kv_capacity =
                ninfer::KvCapacityPolicy::explicit_capacity(512 * options.max_concurrency);
            options.prefill_chunk            = 128;
            options.context_cache.enabled    = false;
            options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
            options.speculative.draft_tokens = drafts;
            options.use_cuda_graph           = graphs;
            const bool ok                    = with_engine(options, [&](ninfer::Engine& engine) {
                const auto input =
                    user_prompt("Count from one to twenty, separating the numbers with commas.");
                const auto result = engine.generate(engine.prepare(input), greedy(40, false));
                std::cout << "MTP K=" << drafts << " graphs=" << graphs
                          << " accepted=" << result.speculative.accepted_tokens << std::endl;
                if (!graphs) {
                    eager = result.generated_token_ids;
                    return true;
                }
                if (result.generated_token_ids != eager) {
                    return fail("MTP CUDA Graph replay differs from eager execution");
                }
                for (const auto batch : {2U, 4U, 8U}) {
                    constexpr std::uint32_t budgets[]{1, 2, 5, 7, 16, 25, 33, 41};
                    std::vector<ninfer::GenerationHandle> handles;
                    for (std::uint32_t row = 0; row < batch; ++row) {
                        handles.push_back(
                            engine.submit(engine.prepare(input), greedy(budgets[row], false)));
                    }
                    for (std::uint32_t row = 0; row < batch; ++row) {
                        const auto r = handles[row].wait();
                        if (r.generated_token_ids.size() != budgets[row] ||
                            r.generated_token_ids.front() != eager.front()) {
                            return fail(
                                "MTP ragged batch failed its token budget or initial distribution");
                        }
                    }
                }
                // Keep eight distinct rows active long enough to exercise the complete B=8
                // verification window before ragged completion drains it.
                std::vector<ninfer::TokenId> first_tokens;
                for (const auto& text : prompts()) {
                    first_tokens.push_back(
                        engine.generate(engine.prepare(user_prompt(text)), greedy(1, false))
                            .generated_token_ids.front());
                }
                std::vector<ninfer::PreparedPrompt> prepared;
                for (const auto& text : prompts()) {
                    prepared.push_back(engine.prepare(user_prompt(text)));
                }
                std::vector<ninfer::GenerationHandle> concurrent;
                for (std::uint32_t row = 0; row < 8; ++row) {
                    concurrent.push_back(
                        engine.submit(std::move(prepared[row]), greedy(96 + row, false)));
                }
                for (std::uint32_t row = 0; row < 8; ++row) {
                    const auto r = concurrent[row].wait();
                    if (r.generated_token_ids.size() != 96 + row ||
                        r.generated_token_ids.front() != first_tokens[row] ||
                        r.speculative.rounds == 0) {
                        return fail(
                            "heterogeneous MTP batch lost a row or its target distribution");
                    }
                }
                // Cross the context boundary with a physical W wider than the remaining budget.
                auto tokens     = engine.tokenize_text("alpha beta gamma delta ");
                const auto seed = tokens;
                while (tokens.size() < 509) {
                    tokens.insert(tokens.end(), seed.begin(), seed.end());
                }
                tokens.resize(509);
                const auto edge = engine.generate(engine.prepare_tokens(tokens), greedy(3, false));
                if (edge.generated_token_ids.size() != 3) {
                    return fail("MTP context-tail verification lost tokens");
                }
                return true;
            });
            if (!ok) { return false; }
        }
    }
    return true;
}

// Long prompts at concurrency with the context cache on, at serving scale: two code requests with
// long outputs and two 20K-token documents are submitted together twice. The second round's
// admissions are planned in one burst while earlier admissions of the burst may still hold
// unsettled StateImage Forks: every plan must seal or wait, and a deferred document must be
// admitted beside the running code requests rather than after them.
bool concurrent_long_prompts_with_reuse(bool mtp) {
    static const std::vector<std::string> sentences{
        "The harbor master logged every ship that crossed the breakwater before dawn.",
        "A survey team mapped the river delta after the spring floods receded.",
        "The orchestra rehearsed the second movement until the tempo settled.",
        "Archivists sorted the letters by sender, date and the port they came from.",
        "The observatory recorded a faint comet low on the western horizon."};
    auto options            = base_options();
    options.max_context     = 32768;
    options.max_concurrency = 4;
    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(131072);
    options.prefill_chunk   = 8192;
    if (mtp) {
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
    }
    return with_engine(options, [&](ninfer::Engine& engine) {
        const auto document = [&](int seed) {
            std::string text = "Read the text and summarize it in five sentences.\n\n";
            for (int i = 0; i < 1200; ++i) {
                text += "Entry " + std::to_string((i * 7919 + seed * 104729) % 10007) + ": " +
                        sentences[static_cast<std::size_t>(i + seed) % sentences.size()] + ' ';
            }
            return text;
        };
        ninfer::PromptInput game;
        game.messages.push_back(message(
            ninfer::ChatRole::System,
            "You are a senior front-end engineer. Every reply is the complete, updated single-file "
            "HTML document, followed by at most three sentences on what changed."));
        game.messages.push_back(message(
            ninfer::ChatRole::User,
            "Write a single-file HTML5 canvas game where a pelican rides a bicycle along a road."));
        game.options.enable_thinking = false;
        const std::vector<std::pair<ninfer::PromptInput, std::uint32_t>> jobs{
            {game, 2000},
            {user_prompt("Write a complete Python implementation of a red-black tree with insert, "
                         "delete and an in-order iterator, with docstrings and tests."),
             2000},
            {user_prompt(document(1)), 400},
            {user_prompt(document(2)), 400},
        };
        std::vector<std::vector<ninfer::TokenId>> first_round;
        for (int round = 0; round < 2; ++round) {
            // Prepare first so the four requests arrive together, as concurrent clients do.
            std::vector<ninfer::PreparedPrompt> prepared;
            for (const auto& job : jobs) { prepared.push_back(engine.prepare(job.first)); }
            std::vector<ninfer::GenerationHandle> handles;
            for (std::size_t row = 0; row < jobs.size(); ++row) {
                // Model stops apply as for a served chat request; the documents end at EOS.
                auto request                        = greedy(jobs[row].second);
                request.stop.include_model_defaults = true;
                handles.push_back(engine.submit(std::move(prepared[row]), request));
            }
            std::uint32_t reused = 0;
            std::vector<ninfer::GenerationResult> results;
            for (std::size_t row = 0; row < handles.size(); ++row) {
                results.push_back(handles[row].wait());
                const auto& result        = results.back();
                const std::size_t outputs = result.generated_token_ids.size();
                if (outputs == 0 || outputs > jobs[row].second) {
                    return fail("a long concurrent request missed its token budget");
                }
                std::cout << "  row " << row << ": reused " << result.reused_prompt_tokens << " of "
                          << result.prompt.prompt_tokens << " via path "
                          << static_cast<int>(result.prefix_reuse_path) << ", " << outputs
                          << " outputs" << std::endl;
                reused += result.reused_prompt_tokens != 0 ? 1U : 0U;
                if (round == 0) {
                    first_round.push_back(result.generated_token_ids);
                } else if (result.generated_token_ids.front() != first_round[row].front()) {
                    return fail("a repeated long request changed its first token");
                }
            }
            // The documents are admitted beside the running code requests, not after them.
            const double code_seconds =
                std::min(results[0].timings.total_seconds, results[1].timings.total_seconds);
            const double document_wait = std::max(results[2].engine_timing.queue_wait_seconds,
                                                  results[3].engine_timing.queue_wait_seconds);
            std::cout << (mtp ? "MTP " : "") << "long concurrent round " << round << ": " << reused
                      << " of " << handles.size() << " reused a prefix; documents "
                      << "waited " << document_wait << " s beside " << code_seconds
                      << " s code requests" << std::endl;
            if (round == 1 && reused == 0) {
                return fail("no repeated long request reused its retained prefix");
            }
            if (document_wait > 0.5 * code_seconds) {
                return fail("long documents waited for the running code requests to finish");
            }
        }
        return true;
    });
}

// MTP rounds of admitted lanes run while a later request's materialization is staged; the staged
// prompt ledger must reach the new sequence intact. Without a system prompt, no request captures
// a shared prefix.
bool mtp_staged_materialization_ledger() {
    auto options                     = base_options();
    options.max_context              = 8192;
    options.max_concurrency          = 4;
    options.kv_capacity              = ninfer::KvCapacityPolicy::explicit_capacity(16384);
    options.prefill_chunk            = 2048;
    options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 3;
    static const std::vector<std::string> sentences{
        "The harbor master logged every ship that crossed the breakwater before dawn.",
        "A survey team mapped the river delta after the spring floods receded.",
        "The orchestra rehearsed the second movement until the tempo settled.",
        "Archivists sorted the letters by sender, date and the port they came from.",
        "The observatory recorded a faint comet low on the western horizon."};
    return with_engine(options, [&](ninfer::Engine& engine) {
        const auto document = [&](int seed) {
            std::string text = "Read the text and summarize it in three sentences.\n\n";
            for (int i = 0; i < 360; ++i) {
                text += "Entry " + std::to_string((i * 7919 + seed * 104729) % 10007) + ": " +
                        sentences[static_cast<std::size_t>(i + seed) % sentences.size()] + ' ';
            }
            return text;
        };
        const std::vector<std::pair<std::string, std::uint32_t>> jobs{
            {document(1), 96},
            {document(2), 96},
            {"Write a Python module implementing a binary heap with push, pop and heapify, with "
             "docstrings and tests.",
             640},
            {"Write a single-file HTML page with a canvas animation of a bouncing ball and a speed "
             "slider.",
             640},
        };
        for (int round = 0; round < 2; ++round) {
            std::vector<ninfer::GenerationHandle> handles;
            for (const auto& [text, outputs] : jobs) {
                handles.push_back(
                    engine.submit(engine.prepare(user_prompt(text)), greedy(outputs)));
            }
            for (std::size_t row = 0; row < handles.size(); ++row) {
                if (handles[row].wait().generated_token_ids.size() != jobs[row].second) {
                    return fail("an MTP request beside a staged materialization missed its budget");
                }
            }
        }
        return true;
    });
}

// A conversation under a system prompt, then a new request repeating its first turn: the new
// request resumes from the system-prompt boundary and promotes it to a shared prefix before its
// first prefill chunk. The promotion rebinds the request's KV, and the MTP KV must stay mapped
// for the rest of the prompt.
bool mtp_shared_prefix_promotion() {
    auto options                     = base_options();
    options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 5;
    return with_engine(options, [&](ninfer::Engine& engine) {
        ninfer::PromptInput first;
        first.messages.push_back(message(
            ninfer::ChatRole::System,
            "You are a senior front-end engineer. Every reply is the complete, updated single-file "
            "HTML document, followed by at most three sentences on what changed."));
        first.messages.push_back(message(
            ninfer::ChatRole::User,
            "Write a single-file HTML5 canvas game where a pelican rides a bicycle along a road."));
        first.options.enable_thinking = false;
        const auto reply              = engine.generate(engine.prepare(first), greedy(64));
        const auto second             = engine.generate(
            engine.prepare(continued(first, reply, "Add obstacles and a score.")), greedy(64));
        const auto repeated = engine.generate(engine.prepare(first), greedy(64));
        std::cout << "MTP shared promotion: second reused " << second.reused_prompt_tokens
                  << ", repeated first turn reused " << repeated.reused_prompt_tokens << " of "
                  << repeated.prompt.prompt_tokens << std::endl;
        if (second.generated_token_ids.size() != 64 || repeated.generated_token_ids.size() != 64) {
            return fail("a request after the shared-prefix promotion missed its token budget");
        }
        if (repeated.reused_prompt_tokens == 0) {
            return fail("the repeated first turn did not resume from the system prompt");
        }
        return repeated.generated_token_ids.front() == reply.generated_token_ids.front()
                   ? true
                   : fail("the repeated first turn changed its first token");
    });
}

bool mtp_long_fp8() {
    std::vector<ninfer::TokenId> eager;
    for (const bool graphs : {false, true}) {
        auto options                     = base_options();
        options.kv_cache                 = ninfer::KvCacheStorage::Fp8E4M3Row256;
        options.context_cache.enabled    = false;
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 5;
        options.use_cuda_graph           = graphs;
        const bool ok                    = with_engine(options, [&](ninfer::Engine& engine) {
            auto tokens     = engine.tokenize_text("alpha beta gamma delta epsilon ");
            const auto seed = tokens;
            while (tokens.size() < 2039) { tokens.insert(tokens.end(), seed.begin(), seed.end()); }
            tokens.resize(2039);
            // Cross QSA's identity/select topology boundary within a verification window.
            const auto result = engine.generate(engine.prepare_tokens(tokens), greedy(48, false));
            std::cout << "MTP FP8 long graphs=" << graphs
                      << " accepted=" << result.speculative.accepted_tokens << std::endl;
            if (!graphs) {
                eager = result.generated_token_ids;
            } else if (eager != result.generated_token_ids) {
                return fail("long FP8 MTP graph replay differs from eager");
            }
            auto request                                  = greedy(48, false);
            request.execution.sampling.temperature        = 0.7F;
            request.execution.sampling.seed               = 7193;
            request.execution.sampling.presence_penalty   = 0.4F;
            request.execution.sampling.repetition_penalty = 1.1F;
            request.execution.logprobs.enabled            = true;
            request.execution.logprobs.top                = 0;
            const auto prompt   = user_prompt("Write a short tale about a fox and a clock.");
            const auto sampled  = engine.generate(engine.prepare(prompt), request);
            const auto repeated = engine.generate(engine.prepare(prompt), request);
            if (sampled.generated_token_ids != repeated.generated_token_ids ||
                sampled.token_logprobs.size() != 48 || sampled.speculative.rounds == 0) {
                return fail(
                    "seeded MTP sampling with penalties did not reproduce its tokens/readout");
            }
            for (const auto& token : sampled.token_logprobs) {
                if (!std::isfinite(token.sampled.raw_logprob) || token.sampled.raw_logprob > 0) {
                    return fail("MTP sampled logprob is invalid");
                }
            }
            return true;
        });
        if (!ok) { return false; }
    }
    return true;
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
            {"host round trip", [] { return host_round_trip_is_exact(); }},
            {"mtp host round trip", [] { return host_round_trip_is_exact(true); }},
            {"reuse survives catalog turnover", reuse_survives_catalog_turnover},
            {"logprobs and structured output", [] { return logprobs_and_structured_output(); }},
            {"mtp logprobs and structured output",
             [] { return logprobs_and_structured_output(true); }},
            {"causal score", causal_score_matches_prompt_readout},
            {"causal prefix independence", causal_prefix_independence},
            {"prefill chunk invariance", prefill_chunk_invariance},
            {"causal hidden readout", causal_hidden_readout},
            {"vision generation and reuse", vision_generation_and_reuse},
            {"mtp generation and reuse", mtp_generation_and_reuse},
            {"mtp graphs and ragged batches", mtp_graphs_and_ragged_batches},
            {"mtp batch target parity", mtp_batch_target_parity},
            {"mtp long fp8", mtp_long_fp8},
            {"long concurrent reuse", [] { return concurrent_long_prompts_with_reuse(false); }},
            {"mtp long concurrent reuse", [] { return concurrent_long_prompts_with_reuse(true); }},
            {"mtp shared prefix promotion", mtp_shared_prefix_promotion},
            {"mtp staged materialization ledger", mtp_staged_materialization_ledger},
        };
        bool passed         = true;
        const char* filter  = std::getenv("NINFER_TEST_CASE");
        std::size_t matched = 0;
        for (const auto& [name, body] : cases) {
            if (filter && std::string(name).find(filter) == std::string::npos) { continue; }
            ++matched;
            std::cout << "== " << name << std::endl;
            try {
                const bool result = body();
                std::cout << (result ? "PASS: " : "FAIL: ") << name << std::endl;
                passed = result && passed;
            } catch (const std::exception& error) {
                std::cerr << "FAIL: " << name << ": " << error.what() << std::endl;
                passed = false;
            }
        }
        if (matched == 0) {
            fail("no test case matched NINFER_TEST_CASE");
            return 1;
        }
        if (!passed) { return 1; }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
