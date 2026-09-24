#include "ninfer/engine.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main() try {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_NVFP4_WEIGHTS");
    if (!artifact || !*artifact) { return 77; }
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 4096;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk = 1024;
    options.kv_cache = ninfer::KvCacheStorage::Fp8E4M3Row256;
    options.context_cache.enabled = false;
    ninfer::Engine engine(options);
    const auto run = [&](std::string text, bool thinking, bool capture) {
        ninfer::PromptInput input;
        input.options.enable_thinking = thinking;
        if (thinking) { input.options.reasoning_effort = ninfer::ReasoningEffort::Medium; }
        input.messages.push_back({.role = ninfer::ChatRole::User,
                                  .parts = {{.kind = ninfer::MessagePartKind::Text, .text = text}}});
        ninfer::RequestOptions request;
        request.execution.capture_reasoning_features = capture;
        request.execution.requested_output_tokens = 1;
        request.execution.sampling.temperature = 0;
        return engine.generate(engine.prepare(std::move(input)), request);
    };
    std::string long_text;
    while (engine.tokenize_text(long_text).size() < 1300) {
        long_text += "Count each active account once and reject duplicates. ";
    }
    auto direct = run(long_text, false, true);
    auto different = run("What is three plus five?", false, true);
    auto reasoned = run(long_text, true, true);
    auto repeated = run(long_text, false, true);
    auto unrequested = run("Hello", false, false);
    if (direct.reasoning_features.size() != 5120 || direct.prompt.reasoning_frontier <= 1024 ||
        direct.reasoning_features != reasoned.reasoning_features ||
        direct.reasoning_features != repeated.reasoning_features ||
        direct.reasoning_features == different.reasoning_features ||
        !unrequested.reasoning_features.empty() || direct.reused_prompt_tokens != 0) {
        throw std::runtime_error("feature boundary, repeatability or lane reset failed");
    }
    for (float value : direct.reasoning_features) {
        if (!std::isfinite(value) || (std::bit_cast<std::uint32_t>(value) & 0xffffU) != 0) {
            throw std::runtime_error("feature is not an exact finite BF16-to-FP32 expansion");
        }
    }
    ninfer::RequestOptions invalid;
    invalid.execution.requested_output_tokens = 1;
    invalid.execution.capture_reasoning_features = true;
    bool rejected = false;
    try { engine.generate(engine.prepare_tokens(engine.tokenize_text("Raw tokens")), invalid); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) { throw std::runtime_error("raw prompt without reasoning boundary was accepted"); }
    std::cout << "OK reasoning features: common prefix, multichunk, exact expansion, lane reset, raw rejection\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
