// Residency of the served model across idle unloads: the state machine the HTTP server relies
// on to keep answering while unloaded, reload on the first request, and share one reload among
// concurrent requests. A fake service stands in for the Engine so every case runs without a GPU.

#include "serve/model_residency.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using ninfer::serve::LoadCause;
using ninfer::serve::ModelUnavailable;
using ninfer::serve::ResidencyState;
using namespace std::chrono_literals;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct FakeService {
    explicit FakeService(std::atomic<int>& live) : live_(live) { ++live_; }
    ~FakeService() { --live_; }
    FakeService(const FakeService&)            = delete;
    FakeService& operator=(const FakeService&) = delete;
    std::atomic<int>& live_;
};

using Residency = ninfer::serve::ModelResidency<FakeService>;

// A loader whose request loads can be held open and made to fail, counting what it was asked.
struct Loader {
    std::atomic<int> live{0};
    std::atomic<int> startup_loads{0};
    std::atomic<int> request_loads{0};
    std::atomic<int> attaches{0};
    std::atomic<int> detaches{0};
    std::atomic<int> unload_events{0};
    std::mutex mutex;
    std::condition_variable changed;
    bool hold    = false; // request loads wait until released
    bool entered = false;
    int failures = 0; // request loads left to fail

    Residency::Loader loader() {
        return [this](LoadCause cause) {
            if (cause == LoadCause::Startup) {
                ++startup_loads;
                return std::make_unique<FakeService>(live);
            }
            ++request_loads;
            std::unique_lock lock(mutex);
            entered = true;
            changed.notify_all();
            changed.wait(lock, [this] { return !hold; });
            if (failures > 0) {
                --failures;
                throw std::runtime_error("device out of memory");
            }
            return std::make_unique<FakeService>(live);
        };
    }

    Residency::Hooks hooks() {
        Residency::Hooks result;
        result.attach   = [this](FakeService&, LoadCause) { ++attaches; };
        result.detach   = [this](FakeService&) { ++detaches; };
        result.unloaded = [this](double, double) { ++unload_events; };
        return result;
    }

    void hold_loads() {
        std::lock_guard lock(mutex);
        hold    = true;
        entered = false;
    }
    void wait_entered() {
        std::unique_lock lock(mutex);
        require(changed.wait_for(lock, 5s, [this] { return entered; }), "reload never started");
    }
    void release_loads() {
        {
            std::lock_guard lock(mutex);
            hold = false;
        }
        changed.notify_all();
    }
};

bool wait_for_state(const Residency& residency, ResidencyState state,
                    std::chrono::milliseconds limit = 5s) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (std::chrono::steady_clock::now() < deadline) {
        if (residency.status().state == state) return true;
        std::this_thread::sleep_for(1ms);
    }
    return residency.status().state == state;
}

// Every thread enters acquire() before any of them may return; waits for the threads to block.
void wait_until_waiting(const std::atomic<int>& started, int expected) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (started.load() < expected) {
        require(std::chrono::steady_clock::now() < deadline, "request threads did not start");
        std::this_thread::sleep_for(1ms);
    }
    // acquire() blocks on the residency's condition variable; give the scheduler time to park.
    std::this_thread::sleep_for(50ms);
}

void test_idle_unload_then_request_reloads() {
    Loader loader;
    Residency residency(loader.loader(), loader.hooks(), 30);
    residency.load_at_startup();
    require(residency.status().state == ResidencyState::Loaded, "startup load did not attach");
    require(loader.live == 1 && loader.startup_loads == 1, "startup load count is wrong");
    { auto lease = residency.acquire(); }

    require(wait_for_state(residency, ResidencyState::Unloaded), "idle model was not unloaded");
    require(loader.live == 0, "the unloaded service was not destroyed");
    require(loader.detaches == 1 && loader.unload_events == 1, "unload was not reported once");
    require(!residency.peek(), "a status lease must not load an unloaded model");
    require(loader.request_loads == 0, "a status lease started a load");

    auto lease = residency.acquire();
    require(static_cast<bool>(lease), "the request after an unload got no service");
    const auto status = residency.status();
    require(status.state == ResidencyState::Loaded && status.loads == 2 && status.unloads == 1,
            "reload did not restore a loaded model");
    require(loader.request_loads == 1 && loader.attaches == 2 && loader.live == 1,
            "reload did not run the loader exactly once");
}

void test_concurrent_requests_share_one_reload() {
    Loader loader;
    Residency residency(loader.loader(), loader.hooks(), 20);
    residency.load_at_startup();
    require(wait_for_state(residency, ResidencyState::Unloaded), "idle model was not unloaded");

    loader.hold_loads();
    constexpr int kRequests = 8;
    std::atomic<int> started{0};
    std::atomic<int> served{0};
    std::vector<FakeService*> services(kRequests, nullptr);
    std::vector<std::thread> threads;
    for (int i = 0; i < kRequests; ++i) {
        threads.emplace_back([&, i] {
            ++started;
            auto lease  = residency.acquire();
            services[i] = &*lease;
            ++served;
        });
    }
    loader.wait_entered();
    wait_until_waiting(started, kRequests);
    require(residency.status().state == ResidencyState::Loading, "reload is not in progress");
    require(served == 0, "a request was served before the reload finished");
    loader.release_loads();
    for (auto& thread : threads) thread.join();

    require(served == kRequests, "not every waiting request was served");
    require(loader.request_loads == 1, "concurrent requests started more than one reload");
    for (FakeService* service : services)
        require(service == services.front(), "requests were served by different loads");
    require(loader.live == 1, "more than one service is resident");
}

void test_failed_reload_fails_its_waiters_and_next_request_retries() {
    Loader loader;
    Residency residency(loader.loader(), loader.hooks(), 20);
    residency.load_at_startup();
    require(wait_for_state(residency, ResidencyState::Unloaded), "idle model was not unloaded");

    {
        std::lock_guard lock(loader.mutex);
        loader.failures = 1;
    }
    loader.hold_loads();
    constexpr int kRequests = 4;
    std::atomic<int> started{0};
    std::atomic<int> refused{0};
    std::vector<std::string> messages(kRequests);
    std::vector<std::thread> threads;
    for (int i = 0; i < kRequests; ++i) {
        threads.emplace_back([&, i] {
            ++started;
            try {
                auto lease = residency.acquire();
            } catch (const ModelUnavailable& exception) {
                messages[i] = exception.what();
                ++refused;
            }
        });
    }
    loader.wait_entered();
    wait_until_waiting(started, kRequests);
    loader.release_loads();
    for (auto& thread : threads) thread.join();

    require(refused == kRequests, "requests that waited for a failed reload were served");
    for (const std::string& message : messages)
        require(message.find("device out of memory") != std::string::npos,
                "the refusal does not carry the load error: " + message);
    const auto failed = residency.status();
    require(failed.state == ResidencyState::Unloaded && failed.last_load_error == "device out of memory",
            "a failed reload did not leave the model unloaded with its error");
    require(loader.request_loads == 1 && loader.live == 0, "a failed reload left a service");

    auto lease = residency.acquire();
    require(static_cast<bool>(lease) && loader.request_loads == 2, "the next request did not retry");
    require(residency.status().last_load_error.empty(), "a successful load kept the old error");
}

void test_requests_hold_the_model_and_restart_the_idle_clock() {
    Loader loader;
    Residency residency(loader.loader(), loader.hooks(), 150);
    residency.load_at_startup();
    {
        auto lease = residency.acquire();
        std::this_thread::sleep_for(400ms); // longer than the idle limit
        require(residency.status().state == ResidencyState::Loaded,
                "the model was unloaded under a held lease");
    }
    // A request every 50 ms keeps a 150 ms limit from expiring, because each release restarts it.
    for (int i = 0; i < 8; ++i) {
        std::this_thread::sleep_for(50ms);
        auto lease = residency.acquire();
    }
    require(residency.status().state == ResidencyState::Loaded && loader.unload_events == 0,
            "the idle clock did not restart after each request");
    const auto released = std::chrono::steady_clock::now();
    require(wait_for_state(residency, ResidencyState::Unloaded), "idle model was not unloaded");
    require(std::chrono::steady_clock::now() - released >= 150ms,
            "the model unloaded before the idle limit after the last request");
}

void test_status_leases_do_not_keep_the_model_loaded() {
    Loader loader;
    Residency residency(loader.loader(), loader.hooks(), 100);
    residency.load_at_startup();
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (residency.status().state == ResidencyState::Loaded &&
           std::chrono::steady_clock::now() < deadline) {
        auto lease = residency.peek(); // health and metrics polling
        std::this_thread::sleep_for(10ms);
    }
    require(residency.status().state != ResidencyState::Loaded,
            "status polling kept an idle model loaded");
}

void test_unload_waits_for_status_reads_in_progress() {
    Loader loader;
    Residency residency(loader.loader(), loader.hooks(), 20);
    residency.load_at_startup();
    auto reading = residency.peek();
    require(wait_for_state(residency, ResidencyState::Unloading),
            "an idle model with a status read in progress did not start unloading");
    require(!residency.peek(), "a status lease was granted while unloading");
    require(loader.live == 1, "the service was destroyed under a status read");
    reading = Residency::Lease{};
    require(wait_for_state(residency, ResidencyState::Unloaded), "the unload did not finish");
    require(loader.live == 0, "the unloaded service was not destroyed");
}

void test_zero_disables_and_limit_changes_live() {
    Loader loader;
    Residency residency(loader.loader(), loader.hooks(), 0);
    residency.load_at_startup();
    { auto lease = residency.acquire(); }
    std::this_thread::sleep_for(200ms);
    require(residency.status().state == ResidencyState::Loaded, "0 ms did not disable idle unload");
    residency.set_idle_unload_ms(20);
    require(residency.status().idle_unload_ms == 20, "the idle limit change was not applied");
    require(wait_for_state(residency, ResidencyState::Unloaded),
            "a live idle limit change did not unload the idle model");
}

void test_stop_refuses_waiting_and_new_requests() {
    Loader loader;
    Residency residency(loader.loader(), loader.hooks(), 0);
    residency.load_at_startup();
    int woken = 0;
    residency.stop([&](FakeService&) { ++woken; });
    require(woken == 1, "stop did not reach the resident service");
    bool refused = false;
    try {
        auto lease = residency.acquire();
    } catch (const ModelUnavailable&) {
        refused = true;
    }
    require(refused, "a request was admitted after stop");
    require(static_cast<bool>(residency.peek()), "status reads must still work while stopping");
}

void test_startup_failure_propagates() {
    Loader loader;
    Residency residency([](LoadCause) -> std::unique_ptr<FakeService> {
        throw std::runtime_error("artifact not found");
    }, {}, 0);
    bool failed = false;
    try {
        residency.load_at_startup();
    } catch (const std::runtime_error& exception) {
        failed = std::string(exception.what()) == "artifact not found";
    }
    require(failed, "a failed startup load did not propagate its error");
    require(residency.status().state == ResidencyState::Unloaded, "failed startup left a state");
}

} // namespace

int main() {
    const std::vector<std::pair<const char*, std::function<void()>>> tests{
        {"idle unload then request reloads", test_idle_unload_then_request_reloads},
        {"concurrent requests share one reload", test_concurrent_requests_share_one_reload},
        {"failed reload fails waiters, next retries",
         test_failed_reload_fails_its_waiters_and_next_request_retries},
        {"requests hold the model and restart the idle clock",
         test_requests_hold_the_model_and_restart_the_idle_clock},
        {"status leases do not keep the model loaded", test_status_leases_do_not_keep_the_model_loaded},
        {"unload waits for status reads in progress",
         test_unload_waits_for_status_reads_in_progress},
        {"zero disables, limit changes live", test_zero_disables_and_limit_changes_live},
        {"stop refuses requests", test_stop_refuses_waiting_and_new_requests},
        {"startup failure propagates", test_startup_failure_propagates},
    };
    int failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS " << name << '\n';
        } catch (const std::exception& exception) {
            ++failures;
            std::cout << "FAIL " << name << ": " << exception.what() << '\n';
        }
    }
    return failures == 0 ? 0 : 1;
}
