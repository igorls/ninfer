#pragma once

// Tray preferences: the memory posture the user chose, persisted next to the
// supervisor config so it survives a restart.
//
// These are deliberately few. The tray asks for a posture, not a tuning: a
// desktop reserve the engine must leave alone, and how long an idle engine
// should stay resident. Everything finer belongs in the config file for the
// handful of people who want it.

#include <nlohmann/json.hpp>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

namespace ninfer::supervisor {

struct TrayPrefs {
    // GiB of device memory the engine must not allocate into. The floor is a
    // measurement, not a guess: on this card a desktop starts missing its 16.7 ms
    // frame budget below ~1.7 GiB free, and shows nothing measurable above ~3 GiB.
    int desktop_reserve_gib = 3;

    // Minutes of inactivity after which the engine is unloaded entirely. 0 = never.
    // Conservative by design -- unloading after five minutes would punish anyone
    // who steps away mid-task, while an hour reliably catches "finished for the
    // day" and little else.
    int idle_unload_minutes = 0;
};

inline std::string tray_prefs_path(const std::string& config_path) {
    const auto dot = config_path.find_last_of('.');
    const std::string base = dot == std::string::npos ? config_path : config_path.substr(0, dot);
    return base + ".tray.json";
}

inline TrayPrefs load_tray_prefs(const std::string& path) {
    TrayPrefs p;
    std::ifstream in(path, std::ios::binary);
    if (!in) { return p; }
    try {
        std::ostringstream buf;
        buf << in.rdbuf();
        const auto j = nlohmann::json::parse(buf.str());
        if (j.contains("desktop_reserve_gib") && j["desktop_reserve_gib"].is_number_integer()) {
            p.desktop_reserve_gib = j["desktop_reserve_gib"].get<int>();
        }
        if (j.contains("idle_unload_minutes") && j["idle_unload_minutes"].is_number_integer()) {
            p.idle_unload_minutes = j["idle_unload_minutes"].get<int>();
        }
    } catch (...) {
        // A corrupt preferences file must never stop the supervisor starting.
        return TrayPrefs{};
    }
    return p;
}

inline void save_tray_prefs(const std::string& path, const TrayPrefs& p) {
    try {
        const nlohmann::json j{{"desktop_reserve_gib", p.desktop_reserve_gib},
                               {"idle_unload_minutes", p.idle_unload_minutes}};
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (out) { out << j.dump(2) << "\n"; }
    } catch (...) {
        // Best effort. Failing to persist a preference is not worth an error path.
    }
}

} // namespace ninfer::supervisor
