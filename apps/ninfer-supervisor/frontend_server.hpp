#pragma once

#include "config.hpp"
#include "frontends.hpp"

#include <nlohmann/json.hpp>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ninfer::supervisor {

// The engine a frontend forwards to, read for every request so a key file or port changed
// through the dashboard applies without restarting the supervisor.
using EngineSpecProvider = std::function<EngineSpec()>;

enum class FrontendState : std::uint8_t {
    Stopped,
    Serving,
    AssetsMissing, // directory missing or without index.html
    PortInUse,
};

// One frontend: static files, a history-API fallback to index.html, and a streaming proxy for the
// engine's API, all on one loopback port. The header stays free of httplib; the server lives
// behind a pointer, as in DashboardServer.
class FrontendServer {
public:
    FrontendServer(FrontendSpec spec, EngineSpecProvider engine);
    ~FrontendServer();
    FrontendServer(const FrontendServer&)            = delete;
    FrontendServer& operator=(const FrontendServer&) = delete;

    // Checks the assets and binds the port; port 0 binds an ephemeral port. False leaves the
    // reason in status().
    bool bind();
    // Serves until stop(). Only after a successful bind().
    void run();
    void stop();

    [[nodiscard]] int bound_port() const noexcept { return bound_port_.load(); }
    [[nodiscard]] FrontendState state() const noexcept { return state_.load(); }
    [[nodiscard]] const FrontendSpec& spec() const noexcept { return spec_; }
    // {name, url, port, state, reason}
    [[nodiscard]] nlohmann::json status_json() const;

private:
    FrontendSpec spec_;
    EngineSpecProvider engine_;
    std::atomic<FrontendState> state_{FrontendState::Stopped};
    // Set by stop(). httplib joins its request threads before listen returns and ends chunked
    // responses itself, but a request still waiting for the engine's headers (a long answer
    // without streaming) only sees this; without it, quitting would wait for the engine.
    std::atomic<bool> stopping_{false};
    std::atomic<int> bound_port_{0};
    mutable std::mutex reason_mutex_;
    std::string reason_;
    void* server_ = nullptr; // httplib::Server*
};

// What the tray needs about one frontend, copied out so a later apply() cannot pull it away.
struct FrontendLink {
    std::string name;
    std::string url;
    FrontendState state = FrontendState::Stopped;
    std::string reason;
};

// Every configured frontend, each on its own thread. apply() reconfigures the set while the
// Supervisor and its engine keep running.
class FrontendHost {
public:
    explicit FrontendHost(EngineSpecProvider engine);
    ~FrontendHost();
    FrontendHost(const FrontendHost&)            = delete;
    FrontendHost& operator=(const FrontendHost&) = delete;

    // Makes the running set match `specs`, in that order. A frontend whose name, folder and port
    // are unchanged and that is serving keeps running untouched; every other one is stopped
    // (ending its open requests) and started anew, so saving an unchanged entry retries a
    // missing folder or a taken port. Removed and changed frontends stop before new ones bind,
    // so two frontends can swap ports.
    void apply(const std::vector<FrontendSpec>& specs);
    void stop();

    [[nodiscard]] nlohmann::json status_json() const;
    [[nodiscard]] std::vector<FrontendLink> links() const;

private:
    struct Entry {
        std::unique_ptr<FrontendServer> server;
        std::thread thread;
    };
    static void shut_down(std::vector<Entry>& entries);

    EngineSpecProvider engine_;
    mutable std::mutex mutex_;
    std::mutex apply_mutex_; // one apply() at a time; never held by status readers
    std::vector<Entry> entries_;
};

} // namespace ninfer::supervisor
