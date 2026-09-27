#include "ninfer/engine.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

// Reasoning feature readout on a real artifact: the row at the reasoning frontier does not depend
// on the thinking suffix rendered after it, repeats exactly, differs across prompts, spans a
// multi-chunk prefill, is an exact BF16-to-FP32 expansion, is empty when not requested, and is
// refused for a raw-token prompt that has no reasoning frontier.
int main() try {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    ninfer::EngineOptions options;
    options.artifact_path         = artifact;
    options.max_context           = 4096;
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk         = 1024;
    options.kv_cache              = ninfer::KvCacheStorage::Fp8E4M3Row256;
    options.context_cache.enabled = false;
    ninfer::Engine engine(options);
    const auto run = [&](std::string text, bool thinking, bool capture) {
        ninfer::PromptInput input;
        input.options.enable_thinking = thinking;
        if (thinking) { input.options.reasoning_effort = ninfer::ReasoningEffort::Medium; }
        input.messages.push_back(
            {.role  = ninfer::ChatRole::User,
             .parts = {{.kind = ninfer::MessagePartKind::Text, .text = std::move(text)}}});
        ninfer::RequestOptions request;
        request.execution.capture_reasoning_features = capture;
        request.execution.requested_output_tokens    = 1;
        request.execution.sampling.temperature       = 0.0F;
        return engine.generate(engine.prepare(std::move(input)), request);
    };
    std::string long_text;
    while (engine.tokenize_text(long_text).size() < 1300) {
        long_text += "Count each active account once and reject duplicates. ";
    }
    const auto direct      = run(long_text, false, true);
    const auto different   = run("What is three plus five?", false, true);
    const auto reasoned    = run(long_text, true, true);
    const auto repeated    = run(long_text, false, true);
    const auto unrequested = run("Hello", false, false);
    if (direct.reasoning_features.empty() || !direct.prompt.reasoning_frontier ||
        *direct.prompt.reasoning_frontier <= 1024 ||
        direct.prompt.reasoning_frontier != reasoned.prompt.reasoning_frontier ||
        direct.reasoning_features != reasoned.reasoning_features ||
        direct.reasoning_features != repeated.reasoning_features ||
        direct.reasoning_features.size() != different.reasoning_features.size() ||
        direct.reasoning_features == different.reasoning_features ||
        !unrequested.reasoning_features.empty() || direct.reused_prompt_tokens != 0) {
        throw std::runtime_error("feature boundary, repeatability or lane reset failed");
    }
    for (const float value : direct.reasoning_features) {
        if (!std::isfinite(value) || (std::bit_cast<std::uint32_t>(value) & 0xffffU) != 0) {
            throw std::runtime_error("feature is not an exact finite BF16-to-FP32 expansion");
        }
    }
    ninfer::RequestOptions invalid;
    invalid.execution.requested_output_tokens    = 1;
    invalid.execution.capture_reasoning_features = true;
    bool rejected                                = false;
    try {
        (void)engine.generate(engine.prepare_tokens(engine.tokenize_text("Raw tokens")), invalid);
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) { throw std::runtime_error("raw prompt without reasoning frontier was accepted"); }
    std::cout << "OK reasoning features: " << direct.reasoning_features.size()
              << " values at frontier " << *direct.prompt.reasoning_frontier
              << "; suffix-independent, repeatable, exact expansion, lane reset, raw rejection\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
