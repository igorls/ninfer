#pragma once

#include "logic.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <system_error>
#include <string>
#include <vector>

namespace ninfer::supervisor {

struct EngineSpec {
    std::string executable;
    std::vector<std::string> args;
    std::string workdir;
    std::string api_key_file;
    std::string engine_host = "127.0.0.1";
    int engine_port         = 8010;
    std::string request_log;
    int device      = 0;
    bool unmanaged  = false; // observe an engine this process did not spawn
};

struct SupervisorConfig {
    EngineSpec engine;
    // Where this config was loaded from. The dashboard writes edits back here, so
    // it must be the resolved path rather than whatever relative string the CLI
    // was given -- the supervisor's working directory is not the user's.
    std::string source_path;
    std::string host = "127.0.0.1";
    int port         = 8099;
    bool bind_any    = false;
    bool monitor_only = false; // never spawn/stop/restart; HTTP observe only
    std::string logs_dir;
    bool run_at_login = false;
    RestartPolicy restart;
};

inline bool manages_engine_process(const SupervisorConfig& cfg) noexcept {
    return !cfg.monitor_only && !cfg.engine.unmanaged;
}

inline std::string read_file_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path); }
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

inline std::string read_api_key(const std::string& path) {
    if (path.empty()) { return {}; }
    std::string raw = read_file_text(path);
    while (!raw.empty() && (raw.back() == '\n' || raw.back() == '\r' || raw.back() == ' ' ||
                            raw.back() == '\t')) {
        raw.pop_back();
    }
    return raw;
}

// Serializes a config back to the on-disk shape. Round-trips the fields the loader
// reads and nothing else, so a hand-written file keeps its meaning after an edit
// made through the dashboard.
//
// The API key is not here and never will be: it lives in the file named by
// api_key_file, and the supervisor only ever reads that file to talk to the engine.
inline nlohmann::json config_to_json(const SupervisorConfig& cfg) {
    nlohmann::json engine = {
        {"executable", cfg.engine.executable},
        {"args", cfg.engine.args},
        {"workdir", cfg.engine.workdir},
        {"api_key_file", cfg.engine.api_key_file},
        {"engine_host", cfg.engine.engine_host},
        {"engine_port", cfg.engine.engine_port},
        {"request_log", cfg.engine.request_log},
        {"device", cfg.engine.device},
        {"unmanaged", cfg.engine.unmanaged},
    };
    nlohmann::json supervisor = {
        {"host", cfg.host},
        {"port", cfg.port},
        {"bind_any", cfg.bind_any},
        {"monitor_only", cfg.monitor_only},
        {"logs_dir", cfg.logs_dir},
        {"run_at_login", cfg.run_at_login},
        {"restart",
         {{"max_backoff_s", cfg.restart.max_backoff_s},
          {"crash_loop_window_s", cfg.restart.crash_loop_window_s},
          {"crash_loop_max", cfg.restart.crash_loop_max},
          {"health_fail_threshold", cfg.restart.health_fail_threshold}}},
    };
    return {{"engine", engine}, {"supervisor", supervisor}};
}

// Write to a sibling temp file, then rename over the original. A half-written
// config is a supervisor that will not start at the next login, and the edit that
// produces it is made from a browser where a refresh mid-write is normal.
inline void save_config_json(const std::string& path, const SupervisorConfig& cfg) {
    if (path.empty()) { throw std::runtime_error("no config path to write to"); }
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) { throw std::runtime_error("cannot write " + tmp); }
        out << config_to_json(cfg).dump(2) << "\n";
        if (!out) { throw std::runtime_error("cannot write " + tmp); }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        throw std::runtime_error("cannot replace " + path);
    }
}

// Same as read_api_key but never throws: used on presentation paths where a
// missing or unreadable key file is a fact to report, not an error to raise.
inline std::string read_api_key_quiet(const std::string& path) noexcept {
    try {
        return read_api_key(path);
    } catch (...) {
        return {};
    }
}

inline SupervisorConfig load_config_json(const std::string& json_text,
                                         bool monitor_only_cli = false) {
    const auto body = nlohmann::json::parse(json_text);
    SupervisorConfig cfg;
    if (body.contains("engine") && body.at("engine").is_object()) {
        const auto& e = body.at("engine");
        cfg.engine.executable   = e.value("executable", "");
        cfg.engine.workdir      = e.value("workdir", "");
        cfg.engine.api_key_file = e.value("api_key_file", "");
        cfg.engine.engine_host  = e.value("engine_host", "127.0.0.1");
        cfg.engine.engine_port  = e.value("engine_port", 8010);
        cfg.engine.request_log  = e.value("request_log", "");
        cfg.engine.device       = e.value("device", 0);
        cfg.engine.unmanaged    = e.value("unmanaged", false);
        if (e.contains("args") && e.at("args").is_array()) {
            for (const auto& a : e.at("args")) {
                if (a.is_string()) { cfg.engine.args.push_back(a.get<std::string>()); }
            }
        }
    }
    if (body.contains("supervisor") && body.at("supervisor").is_object()) {
        const auto& s = body.at("supervisor");
        cfg.host         = s.value("host", "127.0.0.1");
        cfg.port         = s.value("port", 8099);
        cfg.bind_any      = s.value("bind_any", false);
        cfg.monitor_only  = s.value("monitor_only", false) || monitor_only_cli;
        cfg.logs_dir      = s.value("logs_dir", "");
        cfg.run_at_login = s.value("run_at_login", false);
        if (s.contains("restart") && s.at("restart").is_object()) {
            const auto& r = s.at("restart");
            cfg.restart.max_backoff_s         = r.value("max_backoff_s", 60);
            cfg.restart.crash_loop_window_s   = r.value("crash_loop_window_s", 60);
            cfg.restart.crash_loop_max        = r.value("crash_loop_max", 5);
            cfg.restart.health_fail_threshold = r.value("health_fail_threshold", 3);
        }
    }
    if (monitor_only_cli) { cfg.monitor_only = true; }
    if (manages_engine_process(cfg) && cfg.engine.executable.empty()) {
        throw std::invalid_argument(
            "engine.executable is required unless monitor_only or engine.unmanaged");
    }
    if (!cfg.bind_any && !is_loopback_host(cfg.host)) {
        throw std::invalid_argument(
            "supervisor host must be loopback unless bind_any is true");
    }
    return cfg;
}

} // namespace ninfer::supervisor
