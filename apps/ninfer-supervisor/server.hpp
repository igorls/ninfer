#pragma once

#include "collector.hpp"
#include "dashboard.hpp"
#include "engine_child.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <thread>

namespace ninfer::supervisor {

class DashboardServer {
public:
    DashboardServer(SupervisorConfig cfg, EngineChild& child, Collector& collector);
    ~DashboardServer();

    void run();
    void stop();

    // True once listen() has come back having failed to bind. A second
    // supervisor on the same port otherwise runs with no dashboard at all and
    // says nothing about it; the tray shows this as a disabled menu line.
    [[nodiscard]] bool listen_failed() const noexcept { return listen_failed_.load(); }

private:
    nlohmann::json state_json();
    bool control_allowed(const std::string& remote) const;

    SupervisorConfig cfg_;
    EngineChild& child_;
    Collector& collector_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> listen_failed_{false};
    void* server_ = nullptr; // httplib::Server*
};

} // namespace ninfer::supervisor
