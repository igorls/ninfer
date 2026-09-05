#pragma once

#include "config.hpp"
#include "engine_child.hpp"
#include "logic.hpp"
#include "tray_prefs.hpp"

#include <chrono>
#include <cstdint>
#include <string>

namespace ninfer::supervisor {

// What the tray icon says at a glance. In monitor-only mode this still reflects
// the OBSERVED health of the engine rather than a neutral "not my process":
// EngineChild keeps st_.health current for unmanaged engines too, so the colour
// is measured, not invented. The tooltip carries the managed/observing
// distinction instead, because that belongs in words rather than in a hue.
enum class TrayStatus : std::uint8_t {
    Idle,     // grey  — nothing running, or nothing known yet
    Working,  // green — engine running and answering /health
    Pending,  // amber — starting, stopping, backing off, or unreachable
    Failed,   // red   — halted, crash-looped, or reporting unhealthy
};

class TrayIcon {
public:
    TrayIcon(EngineChild& child, EngineSpec& spec, std::string dashboard_url, bool manages_engine,
             std::string prefs_path);
    ~TrayIcon();
    void run();
    void request_quit();
    void open_dashboard() const;
    EngineChild& child();

    // Repaints the tray icon when the status changes. Called on a timer; cheap
    // because it reads EngineChild's in-memory status and never polls the
    // engine or shells out to nvidia-smi.
    void refresh_icon();

    // Idle unload, evaluated on the same timer. Stops the engine after the
    // configured quiet period so a workstation is not holding 70 GiB of weights
    // while nobody is using it. Reload happens on the next explicit start.
    void tick_idle_unload();

    // Read by the menu builder. Small accessors rather than friendship: the menu
    // lives in an anonymous namespace in the .cpp and only needs to read state
    // and hand back the two choices a user can make.
    [[nodiscard]] TrayStatus menu_status() { return current_status(); }
    [[nodiscard]] bool manages_engine() const noexcept { return manages_engine_; }
    [[nodiscard]] int device_index() const noexcept { return spec_.device; }
    [[nodiscard]] int reserve_gib() const noexcept { return prefs_.desktop_reserve_gib; }
    [[nodiscard]] int idle_minutes() const noexcept { return prefs_.idle_unload_minutes; }
    void on_reserve_chosen(int gib);
    void on_idle_chosen(int minutes);

private:
    TrayStatus current_status();
    std::wstring tooltip(TrayStatus status) const;
    void apply_reserve(int gib);
    void set_idle_minutes(int minutes);

    EngineChild& child_;
    EngineSpec& spec_;
    std::string dashboard_url_;
    bool manages_engine_ = true;
    std::string prefs_path_;
    TrayPrefs prefs_;
    void* hwnd_      = nullptr;
    void* hicon_     = nullptr;
    int last_status_ = -1;
    std::chrono::steady_clock::time_point last_busy_ = std::chrono::steady_clock::now();
};

// Device-wide memory via nvidia-smi, run hidden.
//
// Deliberately NOT cudaMemGetInfo and NOT the DXGI budget: both report an empty
// card while another process holds 70 GiB -- verified this session, and the cause
// of two incidents where the engine filled the card and the desktop stopped
// responding. Anything the tray says about memory has to be device-wide truth.
// Returns ok=false when the tool is missing, and the menu then omits the memory
// line rather than showing a wrong one.
NvidiaSmiMemory query_device_memory_smi(int device);

} // namespace ninfer::supervisor
