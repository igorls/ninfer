#pragma once

#include "config.hpp"
#include "reserve_budget.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace ninfer::supervisor {

enum class EngineState : std::uint8_t {
    Stopped,
    Starting,
    Running,
    Stopping,
    BackingOff,
    Halted,
};

struct EngineStatus {
    EngineState state          = EngineState::Stopped;
    std::uint64_t pid          = 0;
    std::int64_t started_unix_ms = 0;
    int restart_count          = 0;
    int last_exit_code         = 0;
    bool crash_loop_halted     = false;
    std::string last_event;
    std::string health; // ok / unhealthy / unreachable
    int health_fails           = 0;

    // Why, not just what. The tray's whole failure mode so far has been an amber
    // dot with no explanation; these come from scanning the engine's own stderr,
    // which is read for engine.log anyway.
    std::string last_error_line;
    std::int64_t last_error_unix_ms    = 0;
    bool ready                         = false; // engine printed `server status=ready`
    int recent_exits                   = 0;
    int crash_window_s                 = 0;
    int desktop_reserve_gib            = 0; // effective at the last spawn; <0 = config default
    // KV plan actually launched. effective is 0 unless the supervisor shrank the configured
    // value to fit the runtime reservation the engine could get; note says why.
    // The argument vector the running process was actually started with, after every
    // supervisor adjustment (reserve, request log, key file, KV capacity). This, not the
    // config file, is what the engine is running.
    std::vector<std::string> launch_args;
};

class EngineChild {
public:
    explicit EngineChild(SupervisorConfig cfg);
    ~EngineChild();

    EngineChild(const EngineChild&)            = delete;
    EngineChild& operator=(const EngineChild&) = delete;

    void start();
    void stop();
    void restart();
    void request_quit();
    void observe_health(int http_status);

    // The reserve the tray last chose, applied to the argument vector at the
    // NEXT spawn. This lives here rather than in the tray because EngineChild
    // owns a *copy* of the config: mutating main's copy from the tray, which is
    // what the first version did, could never reach the command line.
    // kReserveUnset leaves the config's own arguments untouched.
    void set_desktop_reserve_gib(int gib);
    // The tray's idle-unload choice, passed to the NEXT spawn as --idle-unload. The engine
    // itself unloads and reloads the model; a running engine is updated through
    // /admin/residency. Negative leaves the configured arguments untouched.
    void set_idle_unload_minutes(int minutes);
    void note_config_migration(const std::vector<std::string>& changes, const std::string& error);
    bool save_desktop_reserve_gib(int gib, std::optional<PlannedContext> planned = std::nullopt);
    void set_reserve_budget_provider(std::function<nlohmann::json(std::optional<PlannedContext>)> provider) { reserve_budget_provider_ = std::move(provider); }
    nlohmann::json reserve_budget(std::optional<PlannedContext> planned = std::nullopt) const { return reserve_budget_provider_ ? reserve_budget_provider_(planned) : nlohmann::json{{"ok", false}}; }
    bool reserve_plan_matches_config() const;

    // Replaces the configuration the NEXT spawn will use. The dashboard's config
    // editor writes the file and calls this; without it the file would change and
    // a restart would still launch the previous parameters, because this object
    // owns a copy. Takes effect on the next start, never on the running process.
    void update_config(SupervisorConfig cfg);

    [[nodiscard]] EngineStatus status() const;
    [[nodiscard]] std::string log_tail(std::size_t max_bytes) const;
    // By value and under the lock: update_config can swap this out while the tray
    // is drawing a menu from it, and a reference into a member that another thread
    // is assigning is a race whatever the members happen to be.
    [[nodiscard]] SupervisorConfig config() const;
    [[nodiscard]] std::string logs_dir() const;

    void run_loop();

private:
    void spawn();
    void capture_wait();
    void append_log(const char* data, std::size_t n);
    // Records a restart's cause where it outlives st_.last_event, which is
    // overwritten within seconds of the restart it describes. Caller holds mu_.
    void note_restart_reason(const std::string& reason);
    void rotate_logs_if_needed();
    void note_engine_output(const char* data, std::size_t n);

    // The output reader owns nothing: it locks mu_ and mutates st_/line_buf_, so
    // it must never outlive them. Joining is the whole guarantee -- the thread
    // ends on its own when the child's last write handle to the pipe closes, so
    // this only makes the ordering explicit. Never call it while holding mu_.
    void join_reader();

    SupervisorConfig cfg_;
    RestartGate gate_;
    mutable std::mutex mu_;
    EngineStatus st_;
    std::atomic<bool> stop_child_{false};
    std::atomic<bool> quit_{false};
    std::atomic<bool> auto_restart_{true};
    int desktop_reserve_gib_ = kReserveUnset; // under mu_
    int idle_unload_minutes_ = -1;            // under mu_; negative = not chosen by the tray
    EngineSpec launched_spec_;
    std::function<nlohmann::json(std::optional<PlannedContext>)> reserve_budget_provider_;
    std::string line_buf_;                    // under mu_; partial stderr line
    // Under mu_. The last reservation failure the engine reported, consumed by the next
    // spawn; and the shrunken KV plan that spawn chose, kept across automatic restarts and
    // retired when the configuration changes or the engine is started by hand.
    void* process_handle_ = nullptr;          // HANDLE
    void* job_handle_     = nullptr;
    std::string log_path_;
    // Joinable, not detached: see join_reader(). Only the engine thread creates
    // it, and every join happens after the process handle is gone.
    std::thread reader_;
};

} // namespace ninfer::supervisor
