// Paired comparison of two NInfer builds through the public Engine; see README.md.
// NINFER_CURRENT selects the Engine API: 1 for 99d7f90f, 0 for 2b62e8e5.
#include "ninfer/engine.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#if NINFER_CURRENT
#    include "models/qwen3_5/context_budget.h"
#    include "models/qwen4_exp/context_budget.h"
#endif

using namespace ninfer;
using Json  = nlohmann::json;
using Clock = std::chrono::steady_clock;

Json wave(Engine& engine, const std::vector<TokenId>& corpus, int prompt, int concurrency,
          int outputs) {
    std::vector<PreparedPrompt> prepared;
    std::vector<GenerationHandle> handles;
    for (int row = 0; row < concurrency; ++row)
        prepared.push_back(engine.prepare_tokens({corpus.begin(), corpus.begin() + prompt}, false));
    RequestOptions request;
    request.execution.requested_output_tokens = outputs;
    request.execution.allow_prefix_reuse      = false;
    request.execution.sampling.temperature    = 0;
    request.stop.include_model_defaults       = false;
    request.output.raw                        = true;
    request.output.preserve_special_tokens    = true;
#if NINFER_CURRENT
    (void)engine.begin_measurement();
#endif
    const auto start = Clock::now();
    for (auto& p : prepared) {
#if NINFER_CURRENT
        handles.push_back(engine.submit(std::move(p), request, OutputConsumerMode::Aggregate));
#else
        handles.push_back(engine.submit(std::move(p), request, OutputConsumerMode::Aggregate,
                                        {.phase_timings = true}));
#endif
    }
    std::vector<GenerationResult> results;
    for (auto& handle : handles) results.push_back(handle.wait());
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
#if NINFER_CURRENT
    (void)engine.end_measurement();
#endif
    Json rows = Json::array();
    for (const auto& result : results) {
        if (result.generated_token_ids.size() != static_cast<std::size_t>(outputs) ||
            result.finish_reason != FinishReason::OutputLimit)
            throw std::runtime_error("request did not complete its fixed output budget");
#if NINFER_CURRENT
        const auto& s = result.stats.speculative;
        const auto& t = result.stats.timings;
#else
        const auto& s = result.speculative;
        const auto& t = result.timings;
#endif
        rows.push_back({{"ids", result.generated_token_ids},
                        {"rounds", s.rounds},
                        {"drafted_tokens", s.drafted_tokens},
                        {"accepted_tokens", s.accepted_tokens},
                        {"fallback_steps", s.fallback_steps},
                        {"accepted_per_position", s.accepted_per_position},
                        {"prefill_seconds", t.prefill_seconds},
                        {"decode_seconds", t.decode_seconds}});
    }
    return {
        {"prompt_tokens", prompt}, {"outputs", outputs}, {"wave_seconds", seconds}, {"rows", rows}};
}

int main(int argc, char** argv) {
    try {
        if (argc != 9)
            throw std::invalid_argument("model artifact corpus concurrency K capacity mode output");
        const std::string model = argv[1], mode = argv[7];
        const int concurrency = std::stoi(argv[4]), drafts = std::stoi(argv[5]);
        const auto capacity = std::stoull(argv[6]);
        EngineOptions options;
        options.artifact_path                     = argv[2];
        options.max_context                       = 4096;
        options.prefill_chunk                     = 1024;
        options.max_concurrency                   = concurrency;
        options.kv_cache                          = KvCacheStorage::BFloat16;
        options.use_cuda_graph                    = true;
        options.context_cache.enabled             = false;
        options.context_cache.host_capacity_bytes = 0;
        if (const char* reserve = std::getenv("NINFER_COMPARE_DESKTOP_RESERVE_MIB"))
            options.desktop_reserve_bytes = std::size_t(std::stoull(reserve)) << 20;
        if (drafts) {
            options.speculative.backend      = SpeculativeBackend::Mtp;
            options.speculative.draft_tokens = drafts;
            options.speculative.proposal_head =
                model == "dense" ? ProposalHead::Optimized : ProposalHead::Full;
        }
#if NINFER_CURRENT
        if (mode == "capacity") {
            options.clamp_concurrency_to_pool = true;
            const auto plan = model == "dense" ? test::qwen3_5::context_capacity_plan(options)
                                               : test::qwen4_exp::context_capacity_plan(options);
            Json records    = Json::array();
            for (const auto& r : plan.records)
                records.push_back({{"bytes", r.bytes}, {"alignment", r.alignment}});
            std::ofstream(argv[8]) << Json{{"device_bytes", plan.minimum_capacity().device_bytes},
                                           {"records", records},
                                           {"complete_contexts", concurrency},
                                           {"max_context", 4096}}
                                          .dump(2);
            return 0;
        }
        options.context_capacity = ContextCapacityPolicy::explicit_capacity(capacity);
#else
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(4096 * concurrency);
        options.context_cache.device_state_slots = 0;
#endif
        std::ifstream corpus_file(argv[3]);
        std::vector<TokenId> corpus;
        TokenId token;
        while (corpus_file >> token) corpus.push_back(token);
        if (corpus.size() != 65536) throw std::runtime_error("unexpected corpus length");
        const auto start = Clock::now();
        Engine engine(options);
        const double load = std::chrono::duration<double>(Clock::now() - start).count();
        Json report{{"model", model},
                    {"concurrency", concurrency},
                    {"draft_tokens", drafts},
                    {"capacity_argument_bytes", capacity},
                    {"engine_constructor_seconds", load},
                    {"source_current", bool(NINFER_CURRENT)},
                    {"tests", Json::array()}};
#if NINFER_CURRENT
        const auto memory              = engine.snapshot().ranks.at(0).memory;
        const auto host_experts        = memory.experts ? memory.experts->host_weights_bytes : 0;
        report["device_weights_bytes"] = memory.allocations.device_weights_bytes;
#else
        const auto memory = engine.memory_summary();
        // 2b62e8e5 reports no Host expert bytes, so this arm cannot check placement there.
        const auto host_experts = [](const auto& summary) -> std::size_t {
            if constexpr (requires { summary.moe; })
                return summary.moe ? summary.moe->host_weights_bytes : 0;
            else
                return 0;
        }(memory);
        report["device_weights_bytes"] = memory.weights.capacity_bytes;
#endif
        if (host_experts) throw std::runtime_error("comparison requires resident GPU experts");
        report["resident_gpu_experts"] = true;
        report["prime"] = wave(engine, corpus, 1, concurrency, drafts ? 2 * (drafts + 1) + 1 : 3);
        for (const int prompt : {512, 2048}) {
            Json test{{"prompt_tokens", prompt},
                      {"warmup", wave(engine, corpus, prompt, concurrency, 65)},
                      {"repetitions", Json::array()}};
            for (int repetition = 0; repetition < 3; ++repetition)
                test["repetitions"].push_back(wave(engine, corpus, prompt, concurrency, 65));
            report["tests"].push_back(std::move(test));
        }
        std::ofstream output(argv[8]);
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output << report.dump(2) << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
