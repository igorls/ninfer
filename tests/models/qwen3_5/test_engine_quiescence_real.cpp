#include "ninfer/engine.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
using namespace ninfer;
using namespace std::chrono_literals;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

RequestOptions request(std::uint32_t tokens) {
    RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature = 0;
    options.execution.allow_prefix_reuse = false;
    options.stop.include_model_defaults = false;
    options.output.raw = true;
    return options;
}

void completed(const GenerationResult& result, std::size_t tokens) {
    require(result.finish_reason == FinishReason::OutputLimit &&
                result.generated_token_ids.size() == tokens,
            "quiescence lost or truncated a request");
}

struct Hold {
    std::promise<void> entered;
    std::promise<void> release;
    std::shared_future<void> released = release.get_future().share();

    void run() {
        entered.set_value();
        require(released.wait_for(10s) == std::future_status::ready,
                "test did not release the quiescence callback");
    }
    void finish() noexcept {
        try { release.set_value(); } catch (...) {}
    }
    ~Hold() { finish(); }
};

void exercise(const std::filesystem::path& artifact) {
    EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 512;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk = 128;
    options.max_concurrency = 1;
    options.max_pending_requests = 4;
    options.pending_timeout_ms = 200;
    options.context_cache.enabled = false;
    options.use_cuda_graph = std::getenv("NINFER_TEST_EAGER") == nullptr;
    const char* backend = std::getenv("NINFER_TEST_BACKEND");
    if (backend && std::string(backend) == "mtp") {
        options.speculative.backend = SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
        const char* head = std::getenv("NINFER_TEST_PROPOSAL_HEAD");
        options.speculative.proposal_head = head && std::string(head) == "full"
                                               ? ProposalHead::Full
                                               : ProposalHead::Optimized;
    }
    Engine engine(options);

    // An idle worker must wake for the fence without a new generation request.
    unsigned calls = 0;
    auto idle = std::async(std::launch::async, [&] {
        return engine.run_at_quiescence([&] { ++calls; });
    });
    require(idle.wait_for(5s) == std::future_status::ready, "idle fence did not wake worker");
    require(idle.get().requests_held == 0 && calls == 1, "idle callback ran incorrectly");
    try {
        engine.run_at_quiescence([] { throw std::runtime_error("refused test transition"); });
        throw std::logic_error("callback exception was swallowed");
    } catch (const std::runtime_error& error) {
        require(std::string(error.what()) == "refused test transition",
                "callback exception changed");
    }
    completed(engine.generate(engine.prepare_tokens({198, 1000, 198}), request(4)), 4);

    // Queue a second fence while the first is executing. Requests arriving during
    // the hold must retain their admission time even after their original deadline.
    Hold first, second;
    auto first_entered = first.entered.get_future();
    auto second_entered = second.entered.get_future();
    auto hold_one = std::async(std::launch::async, [&] {
        return engine.run_at_quiescence([&] { first.run(); });
    });
    require(first_entered.wait_for(5s) == std::future_status::ready, "first hold did not run");
    auto hold_two = std::async(std::launch::async, [&] {
        return engine.run_at_quiescence([&] { second.run(); });
    });
    auto queued = engine.submit(engine.prepare_tokens({198, 1001, 198}), request(8));
    auto result = std::async(std::launch::async, [&] { return queued.wait(); });
    std::this_thread::sleep_for(300ms);
    require(result.wait_for(0ms) != std::future_status::ready,
            "request ran or expired inside the physical-capacity hold");
    first.finish();
    require(second_entered.wait_for(5s) == std::future_status::ready, "second hold did not run");
    std::this_thread::sleep_for(300ms);
    require(result.wait_for(0ms) != std::future_status::ready,
            "queued fence admitted or expired a held request");
    second.finish();
    require(hold_one.get().requests_held == 1 && hold_two.get().requests_held == 1,
            "queued callbacks did not account for the held request");
    completed(result.get(), 8);
    require(engine.is_available(), "callback refusal made engine unavailable");

    // Begin a fence while generation owns a lane. It must drain that owner before
    // running the callback and remain usable after both execution and retirement.
    auto active = engine.submit(engine.prepare_tokens({198, 1002, 198}), request(32));
    auto active_result = std::async(std::launch::async, [&] { return active.wait(); });
    const auto admitted_by = std::chrono::steady_clock::now() + 5s;
    while (engine.snapshot().runtime.running_requests == 0 &&
           active_result.wait_for(0ms) != std::future_status::ready) {
        require(std::chrono::steady_clock::now() < admitted_by, "prior request was not admitted");
        std::this_thread::yield();
    }
    engine.run_at_quiescence([&] {
        require(active_result.wait_for(5s) == std::future_status::ready,
                "fence ran before the prior request completed");
    });
    completed(active_result.get(), 32);
    completed(engine.generate(engine.prepare_tokens({198, 1003, 198}), request(4)), 4);
    std::cout << "real Engine quiescence: PASS\n";
}
} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        exercise(artifact);
    } catch (const std::exception& error) {
        std::cerr << "real Engine quiescence failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
