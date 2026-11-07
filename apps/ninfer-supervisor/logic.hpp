#pragma once
#include "product/config/options.h"
#include "product/config/quantity.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::supervisor {

inline bool is_loopback_host(std::string_view host) {
    return host == "127.0.0.1" || host == "::1" || host == "localhost" || host == "localhost.";
}

inline bool is_loopback_peer(std::string_view addr) {
    if (addr.empty()) { return false; }
    if (is_loopback_host(addr)) { return true; }
    // httplib may report IPv4-mapped IPv6.
    return addr == "::ffff:127.0.0.1";
}

inline constexpr std::string_view kSupervisorControlHeader      = "X-NInfer-Supervisor";
inline constexpr std::string_view kSupervisorControlHeaderValue = "1";

inline std::string_view trim_sv(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' ||
                          s.front() == '\n')) {
        s.remove_prefix(1);
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' ||
                          s.back() == '\n')) {
        s.remove_suffix(1);
    }
    return s;
}

// Split Host into name and optional port. IPv6 literals must be bracketed when a
// port is present (`[::1]:8099`).
inline bool split_host_header(std::string_view host, std::string& name, int& port, bool& has_port) {
    host     = trim_sv(host);
    name.clear();
    port     = 0;
    has_port = false;
    if (host.empty()) { return false; }
    if (host.front() == '[') {
        const auto rb = host.find(']');
        if (rb == std::string_view::npos) { return false; }
        name = std::string(host.substr(0, rb + 1));
        if (rb + 1 == host.size()) { return true; }
        if (host[rb + 1] != ':') { return false; }
        const auto p = host.substr(rb + 2);
        if (p.empty()) { return false; }
        int value = 0;
        for (char c : p) {
            if (c < '0' || c > '9') { return false; }
            value = value * 10 + (c - '0');
            if (value > 65535) { return false; }
        }
        port     = value;
        has_port = true;
        return true;
    }
    const auto colon = host.rfind(':');
    if (colon != std::string_view::npos && host.find(':') == colon) {
        name = std::string(host.substr(0, colon));
        const auto p = host.substr(colon + 1);
        if (p.empty() || name.empty()) { return false; }
        int value = 0;
        for (char c : p) {
            if (c < '0' || c > '9') { return false; }
            value = value * 10 + (c - '0');
            if (value > 65535) { return false; }
        }
        port     = value;
        has_port = true;
        return true;
    }
    name = std::string(host);
    return !name.empty();
}

inline bool is_loopback_host_name(std::string_view name) {
    return is_loopback_host(name) || name == "[::1]";
}

// DNS-rebinding defense: only the listen port's loopback names, plus the
// configured bind host when --bind-any names a specific interface. Binding
// 0.0.0.0 does not open the Host allowlist.
inline bool host_header_allowed(std::string_view host_header, int listen_port,
                                std::string_view bind_host, bool bind_any) {
    std::string name;
    int port         = 0;
    bool has_port    = false;
    if (!split_host_header(host_header, name, port, has_port)) { return false; }
    if (has_port && port != listen_port) { return false; }
    if (is_loopback_host_name(name)) { return true; }
    if (!bind_any) { return false; }
    if (bind_host.empty() || bind_host == "0.0.0.0" || bind_host == "::" || bind_host == "[::]") {
        return false;
    }
    return name == bind_host;
}

inline bool supervisor_control_header_ok(std::string_view value) {
    return trim_sv(value) == kSupervisorControlHeaderValue;
}

struct NvidiaMemory {
    bool ok                 = false;
    int index               = -1;
    std::uint64_t used_mib  = 0;
    std::uint64_t total_mib = 0;
    std::string error;
};

// Parses `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits`.
// Values are mebibytes. Picks the row whose index equals `device`.
inline NvidiaMemory parse_nvidia_smi_memory_csv(std::string_view csv, int device) {
    NvidiaMemory out;
    std::string_view rest = csv;
    bool saw_row          = false;
    while (!rest.empty()) {
        auto nl    = rest.find_first_of("\n\r");
        auto line  = trim_sv(nl == std::string_view::npos ? rest : rest.substr(0, nl));
        rest       = nl == std::string_view::npos ? std::string_view{}
                                                  : rest.substr(nl + 1);
        if (line.empty()) { continue; }
        saw_row = true;
        const auto c1 = line.find(',');
        if (c1 == std::string_view::npos) { continue; }
        const auto c2 = line.find(',', c1 + 1);
        if (c2 == std::string_view::npos) { continue; }
        const auto idx_s  = trim_sv(line.substr(0, c1));
        const auto used_s = trim_sv(line.substr(c1 + 1, c2 - c1 - 1));
        const auto tot_s  = trim_sv(line.substr(c2 + 1));
        int idx           = 0;
        std::uint64_t used = 0;
        std::uint64_t tot  = 0;
        try {
            idx  = std::stoi(std::string(idx_s));
            used = std::stoull(std::string(used_s));
            tot  = std::stoull(std::string(tot_s));
        } catch (...) { continue; }
        if (idx != device) { continue; }
        out.ok       = true;
        out.index    = idx;
        out.used_mib = used;
        out.total_mib = tot;
        return out;
    }
    out.error = saw_row ? "nvidia-smi csv has no row for the configured device"
                        : "nvidia-smi csv is empty";
    return out;
}

inline std::uint64_t mib_to_bytes(std::uint64_t mib) { return mib * 1024ull * 1024ull; }

// Pre-filter that agrees with the engine JSONL schema: the field is "event",
// not "type". A substring on the value without the key name is how a panel
// can look populated while every record is then discarded.
inline bool jsonl_event_is(std::string_view line, std::string_view event) {
    const std::string compact = std::string("\"event\":\"") + std::string(event) + "\"";
    const std::string spaced  = std::string("\"event\": \"") + std::string(event) + "\"";
    return line.find(compact) != std::string_view::npos ||
           line.find(spaced) != std::string_view::npos;
}

struct VramSample {
    std::int64_t t_ms                = 0;
    std::uint64_t budget_bytes       = 0;
    std::uint64_t nvidia_used_bytes  = 0;
};

struct VramSeriesEvent {
    std::int64_t t_ms = 0;
    std::string kind;
    std::string label;
};

// One engine throughput report: tokens per WALL second across everything running
// at once, which is a different number from the average speed of one response.
// Taken from the engine's own `throughput` log event rather than derived here.
//
// Deriving it from completed requests was the obvious approach and it is wrong:
// a request only reaches the log when it finishes, so a window over completions
// reads ZERO for the whole of a long generation -- precisely when the card is
// busiest. Measured that happening: 207.9 tok/s, then 0.0 with four generations
// still streaming. The engine emits this every ~5 s while the work is happening.
struct ThroughputSample {
    std::int64_t t_ms     = 0;
    double decode_tok_s   = 0;
    double prefill_tok_s  = 0;
    int running           = 0;
};

// Ring of engine throughput reports (~0.2 Hz). Separate from the 10 Hz VRAM
// series: this arrives on the engine's own cadence, and folding it into the fast
// series would decuple series.jsonl for no extra information.
struct ThroughputRing {
    explicit ThroughputRing(std::size_t cap = 900) : cap_(cap), buf_(cap) {}

    void clear() noexcept {
        head_ = 0;
        size_ = 0;
    }

    void push(ThroughputSample s) {
        if (cap_ == 0) { return; }
        buf_[head_] = s;
        head_       = (head_ + 1) % cap_;
        if (size_ < cap_) { ++size_; }
    }

    [[nodiscard]] std::vector<ThroughputSample> samples() const {
        std::vector<ThroughputSample> out;
        out.reserve(size_);
        const std::size_t start = size_ < cap_ ? 0 : head_;
        for (std::size_t i = 0; i < size_; ++i) { out.push_back(buf_[(start + i) % cap_]); }
        return out;
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
    std::size_t cap_  = 0;
    std::size_t head_ = 0;
    std::size_t size_ = 0;
    std::vector<ThroughputSample> buf_;
};

// Diff last_transition/last_reason, not held_bytes. A 42 ms release finishes
// between 1 Hz polls; the persisted last_reason is what remains observable.
struct AdminVramCursor {
    bool seen                = false;
    std::string last_transition;
    std::string last_reason;

    // Returns true if this observation is an event after the baseline poll.
    bool observe(std::string_view trans, std::string_view reason, std::string& kind) {
        if (!seen) {
            seen             = true;
            last_transition  = std::string(trans);
            last_reason      = std::string(reason);
            return false;
        }
        if (trans == last_transition && reason == last_reason) { return false; }
        last_transition = std::string(trans);
        last_reason     = std::string(reason);
        if (trans == "release") {
            kind = "vram_release";
        } else if (trans == "reclaim" || trans == "reclaim-failed") {
            kind = "vram_reclaim";
        } else {
            kind = "admin_vram";
        }
        return true;
    }
};

inline bool parse_series_event_line(std::string_view line, VramSeriesEvent& ev);
inline bool parse_series_sample_line(std::string_view line, VramSample& s);

// Ring of raw 10 Hz samples. No averaging. Oldest is dropped on overflow.
struct VramSeriesRing {
    explicit VramSeriesRing(std::size_t cap = 6000) : cap_(cap), buf_(cap) {}

    void push(VramSample s) {
        if (cap_ == 0) { return; }
        buf_[head_] = s;
        head_       = (head_ + 1) % cap_;
        if (size_ < cap_) { ++size_; }
    }

    void push_event(VramSeriesEvent e, std::size_t event_cap = 128) {
        events_.push_back(std::move(e));
        while (events_.size() > event_cap) { events_.pop_front(); }
    }

    [[nodiscard]] std::vector<VramSample> samples() const {
        std::vector<VramSample> out;
        out.reserve(size_);
        const std::size_t start = size_ < cap_ ? 0 : head_;
        for (std::size_t i = 0; i < size_; ++i) { out.push_back(buf_[(start + i) % cap_]); }
        return out;
    }

    [[nodiscard]] std::vector<VramSeriesEvent> events() const {
        return {events_.begin(), events_.end()};
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    void load_jsonl(std::string_view jsonl) {
        std::string_view rest = jsonl;
        while (!rest.empty()) {
            auto nl   = rest.find('\n');
            auto line = trim_sv(nl == std::string_view::npos ? rest : rest.substr(0, nl));
            rest      = nl == std::string_view::npos ? std::string_view{} : rest.substr(nl + 1);
            if (line.empty()) { continue; }
            VramSeriesEvent ev;
            VramSample samp;
            if (parse_series_event_line(line, ev)) {
                push_event(std::move(ev));
            } else if (parse_series_sample_line(line, samp)) {
                push(samp);
            }
        }
    }

private:
    std::size_t cap_  = 0;
    std::size_t head_ = 0;
    std::size_t size_ = 0;
    std::vector<VramSample> buf_;
    std::deque<VramSeriesEvent> events_;
};

inline bool extract_json_i64(std::string_view line, std::string_view key, std::int64_t& out) {
    const std::string pat = "\"" + std::string(key) + "\":";
    auto pos              = line.find(pat);
    if (pos == std::string_view::npos) { return false; }
    pos += pat.size();
    while (pos < line.size() && (line[pos] == ' ')) { ++pos; }
    bool neg = false;
    if (pos < line.size() && line[pos] == '-') {
        neg = true;
        ++pos;
    }
    if (pos >= line.size() || line[pos] < '0' || line[pos] > '9') { return false; }
    std::int64_t v = 0;
    while (pos < line.size() && line[pos] >= '0' && line[pos] <= '9') {
        v = v * 10 + (line[pos] - '0');
        ++pos;
    }
    out = neg ? -v : v;
    return true;
}

inline bool extract_json_str(std::string_view line, std::string_view key, std::string& out) {
    const std::string pat = "\"" + std::string(key) + "\":\"";
    auto pos              = line.find(pat);
    if (pos == std::string_view::npos) { return false; }
    pos += pat.size();
    std::string s;
    while (pos < line.size() && line[pos] != '"') {
        s.push_back(line[pos]);
        ++pos;
    }
    out = std::move(s);
    return true;
}

inline bool parse_series_event_line(std::string_view line, VramSeriesEvent& ev) {
    if (line.find("\"kind\"") == std::string_view::npos) { return false; }
    std::int64_t t = 0;
    if (!extract_json_i64(line, "t_ms", t)) { return false; }
    ev.t_ms = t;
    extract_json_str(line, "kind", ev.kind);
    extract_json_str(line, "label", ev.label);
    return !ev.kind.empty();
}

inline bool parse_series_sample_line(std::string_view line, VramSample& s) {
    if (line.find("\"kind\"") != std::string_view::npos) { return false; }
    std::int64_t t = 0, b = 0, n = 0;
    if (!extract_json_i64(line, "t_ms", t)) { return false; }
    extract_json_i64(line, "budget_bytes", b);
    extract_json_i64(line, "nvidia_used_bytes", n);
    s.t_ms               = t;
    s.budget_bytes       = static_cast<std::uint64_t>(b);
    s.nvidia_used_bytes  = static_cast<std::uint64_t>(n);
    return true;
}

inline std::string format_series_sample_line(const VramSample& s) {
    return std::string("{\"t_ms\":") + std::to_string(s.t_ms) +
           ",\"budget_bytes\":" + std::to_string(s.budget_bytes) +
           ",\"nvidia_used_bytes\":" + std::to_string(s.nvidia_used_bytes) + "}";
}

inline std::string format_series_event_line(const VramSeriesEvent& e) {
    return std::string("{\"t_ms\":") + std::to_string(e.t_ms) + ",\"kind\":\"" + e.kind +
           "\",\"label\":\"" + e.label + "\"}";
}

inline std::string extract_context_capacity_line(std::string_view log) {
    const auto key = std::string_view("Context capacity ");
    const auto pos = log.rfind(key);
    if (pos == std::string_view::npos) { return {}; }
    auto start = log.find_last_of("\n", pos);
    start      = start == std::string_view::npos ? 0 : start + 1;
    auto end   = log.find('\n', pos);
    auto line  = log.substr(start, end == std::string_view::npos ? log.size() - start : end - start);
    if (!line.empty() && line.back() == '\r') { line.remove_suffix(1); }
    return std::string(trim_sv(line));
}

struct RestartPolicy {
    int initial_backoff_s     = 1;
    int max_backoff_s         = 60;
    int crash_loop_max        = 5;
    int crash_loop_window_s   = 60;
    int health_fail_threshold = 3;
};

class RestartGate {
public:
    explicit RestartGate(RestartPolicy policy = {}) : policy_(policy), backoff_s_(policy.initial_backoff_s) {}

    // Record an engine exit. Returns false if auto-restart is halted (crash loop).
    bool note_exit(std::chrono::steady_clock::time_point now) {
        if (halted_) { return false; }
        const auto window = std::chrono::seconds(policy_.crash_loop_window_s);
        while (!exits_.empty() && now - exits_.front() > window) { exits_.pop_front(); }
        exits_.push_back(now);
        if (static_cast<int>(exits_.size()) >= policy_.crash_loop_max) {
            halted_ = true;
            return false;
        }
        return true;
    }

    [[nodiscard]] int backoff_seconds() const noexcept { return backoff_s_; }

    void advance_backoff() {
        if (backoff_s_ < policy_.max_backoff_s) {
            const int next = backoff_s_ * 2;
            backoff_s_     = next > policy_.max_backoff_s ? policy_.max_backoff_s : next;
        }
    }

    void note_healthy() {
        backoff_s_ = policy_.initial_backoff_s;
        health_fails_ = 0;
    }

    bool note_health_fail() {
        ++health_fails_;
        return health_fails_ >= policy_.health_fail_threshold;
    }

    void clear_health_fails() { health_fails_ = 0; }

    void reset_halt() {
        halted_ = false;
        exits_.clear();
        backoff_s_    = policy_.initial_backoff_s;
        health_fails_ = 0;
    }

    [[nodiscard]] bool halted() const noexcept { return halted_; }
    [[nodiscard]] int recent_exits() const noexcept { return static_cast<int>(exits_.size()); }
    [[nodiscard]] int health_fails() const noexcept { return health_fails_; }
    [[nodiscard]] const RestartPolicy& policy() const noexcept { return policy_; }

private:
    RestartPolicy policy_;
    std::deque<std::chrono::steady_clock::time_point> exits_;
    int backoff_s_    = 1;
    int health_fails_ = 0;
    bool halted_      = false;
};

// ---------------------------------------------------------------------------
// Tray logic. Everything below is pure so it can be unit-tested off a GPU and
// off Win32: the tray's bugs so far have all been in arithmetic (menu id
// collisions, argument rewriting), not in the Win32 calls around it.
// ---------------------------------------------------------------------------

// The reserve choices. 0 is "engine default" and means the flag is dropped so
// the engine's own kDefaultDesktopReserveBytes (8 GiB) applies -- the tray must
// not silently lower a reserve nobody asked it to change. These are shortcuts;
// the dashboard accepts custom whole-GiB targets. None is a universal minimum
// or recommendation: app demand depends on the user's workload.
inline constexpr std::array<int, 7> kReserveChoicesGib{0, 2, 3, 4, 6, 8, 12};

// A duration, not a toggle. Unloading after five minutes would punish anyone who
// steps away mid-task; an hour reliably catches "finished for the day".
inline constexpr std::array<int, 4> kIdleChoicesMinutes{0, 15, 60, 240};

// Sentinel for TrayPrefs::desktop_reserve_gib: the user has never picked one, so
// whatever the config file says stays untouched. Distinct from 0 ("I chose the
// engine default"), which does strip the flag.
inline constexpr int kReserveUnset = -1;

// Resolve exactly the same file/argv/default layers as the engine child. No model
// artifact is opened by the startup parser. Isolated helper calls may omit it.
inline product::ServeOptions child_startup_options(const std::vector<std::string>& args) {
    std::vector<std::string> arguments{"ninfer-serve"};
    if (args.empty() || args.front().starts_with("--")) arguments.push_back("model.ninfer");
    arguments.insert(arguments.end(), args.begin(), args.end());
    std::vector<char*> pointers;
    for (auto& argument : arguments) pointers.push_back(argument.data());
    return product::parse_serve_options(static_cast<int>(pointers.size()), pointers.data());
}

// Engine flags that earlier releases accepted and the engine now rejects. The previous stable
// launcher wrote several into every first-run profile, so an upgrade has to replace them or the
// engine never starts again. A value is rewritten as `value + unit`, or as `fixed` when the units
// changed meaning, and an empty replacement means the setting no longer exists.
struct RetiredEngineFlag {
    std::string_view flag;
    std::string_view replacement;
    std::string_view unit;
    std::string_view fixed;
    bool takes_value = true;
};

inline constexpr std::array<RetiredEngineFlag, 26> kRetiredEngineFlags{{
    {"--kv-capacity", "--context-capacity", "", "auto"},
    {"--lm-head-draft", "--proposal-head", "", "optimized", false},
    {"--greedy", "--temperature", "", "0", false},
    {"--desktop-reserve-gib", "--desktop-reserve", "GiB"},
    {"--desktop-reserve-mib", "--desktop-reserve", "MiB"},
    {"--host-context-mib", "--host-cache-capacity", "MiB"},
    {"--host-kv-mib", "--host-cache-capacity", "MiB"},
    {"--max-request-mib", "--max-request-size", "MiB"},
    {"--media-cache-mib", "--media-cache-capacity", "MiB"},
    {"--media-live-mib", "--media-live-capacity", "MiB"},
    {"--response-store-max-mib", "--response-store-max-size", "MiB"},
    {"--expert-trace-max-gib", "--expert-trace-max-size", "GiB"},
    {"--log-stats-interval-ms", "--stats-interval", "ms"},
    {"--pending-timeout-ms", "--pending-timeout", "ms"},
    {"--default-max-tokens", "--max-tokens"},
    {"--default-thinking-budget", "--thinking-budget"},
    {"--kv-slack-floor", "--memory-headroom"},
    {"--kv-slack-floor-mib", "--memory-headroom", "MiB"},
    {"--device-state-slots", ""},
    {"--extra-state-slots", ""},
    {"--host-state-slots", ""},
    {"--max-private-continuations", ""},
    {"--max-shared-prefixes", ""},
    {"--gdn-state-dtype", ""},
    {"--output-head-dtype", ""},
    {"--token-embedding-dtype", ""},
}};

inline bool engine_args_contain(const std::vector<std::string>& args, std::string_view flag) {
    const std::string assigned = std::string(flag) + "=";
    return std::any_of(args.begin(), args.end(), [&](const std::string& token) {
        return token == flag || token.starts_with(assigned);
    });
}

// Rewrites retired flags in place and describes each change. A replacement already on the line
// wins, because it was set for the current engine and the old flag can only restate it.
inline std::vector<std::string> migrate_retired_engine_flags(std::vector<std::string>& args) {
    std::vector<std::string> migrated;
    std::vector<std::string> changes;
    migrated.reserve(args.size());
    for (std::size_t i = 0; i < args.size(); ++i) {
        const auto retired =
            std::find_if(kRetiredEngineFlags.begin(), kRetiredEngineFlags.end(),
                         [&](const RetiredEngineFlag& entry) { return entry.flag == args[i]; });
        if (retired == kRetiredEngineFlags.end()) {
            migrated.push_back(args[i]);
            continue;
        }
        std::string old_value;
        if (retired->takes_value && i + 1 < args.size() && !args[i + 1].starts_with("--")) {
            old_value = args[++i];
        }
        std::string change = std::string(retired->flag);
        if (!old_value.empty()) change += " " + old_value;
        const std::string value = !retired->fixed.empty() ? std::string(retired->fixed)
                                  : old_value.empty()     ? std::string()
                                                          : old_value + std::string(retired->unit);
        if (retired->replacement.empty() || value.empty() ||
            engine_args_contain(args, retired->replacement) ||
            engine_args_contain(migrated, retired->replacement)) {
            changes.push_back(change + " removed");
            continue;
        }
        migrated.emplace_back(retired->replacement);
        migrated.push_back(value);
        changes.push_back(change + " -> " + std::string(retired->replacement) + " " + value);
    }
    args = std::move(migrated);
    return changes;
}

// The tray edits whole GiB. Preserve an explicitly finer-grained quantity.
inline bool desktop_reserve_pinned(const std::vector<std::string>& args) {
    try {
        return child_startup_options(args).launch.engine.desktop_reserve_bytes % (1ULL << 30) != 0;
    } catch (...) { return false; }
}

// ---------------------------------------------------------------------------
// Engine launch parameters as data.
//
// One table drives four things that used to drift apart: parsing the configured
// argument vector, validating an edit, rendering the vector back, and describing
// the form the dashboard draws. Adding a knob means adding a row.
//
// What is deliberately NOT here: the executable path and the working directory.
// Those decide which binary runs, and the dashboard must not be able to change
// them -- the control gate is a loopback peer check plus a header, which is a
// CSRF brake, not an authorization system. Parameters are editable; what gets
// executed is a file-only decision.
// ---------------------------------------------------------------------------

enum class ParamKind : std::uint8_t {
    Int,       // "--flag N"
    Text,      // "--flag STRING"
    Enum,      // "--flag CHOICE"
    Flag,      // "--flag" with no value; present or absent
    CapacityOrAuto, // byte quantity or auto
    Capacity,  // exact bytes or B/KiB/MiB/GiB quantity
    Duration,  // exact milliseconds or ms/s quantity
    IntOrNone, // positive integer or none
};

struct EngineParamSpec {
    std::string_view flag;
    std::string_view key;
    ParamKind kind{};
    std::string_view group;
    std::string_view label;
    std::string_view help;
    long long min_value = 0;
    long long max_value = 0;
    std::array<std::string_view, 6> choices{};
};

// Ranges are the engine's own limits where one exists (max_concurrency throws
// above 8 in the engine, so the form must not offer 16), and otherwise a bound
// wide enough not to be a policy while still catching a typo'd extra digit.
inline const std::vector<EngineParamSpec>& engine_param_specs() {
    static const std::vector<EngineParamSpec> specs = {
        {"--max-context", "max_context", ParamKind::Int, "memory", "Max context",
         "Longest prompt plus reply one request may use (K = 1,024 tokens).", 256, 262144, {}},
        {"--context-capacity", "context_capacity", ParamKind::CapacityOrAuto, "memory", "Context capacity",
         "GPU memory every request shares for its KV cache and state.", 1, 262144LL << 20, {}},
        {"--max-concurrency", "max_concurrency", ParamKind::Int, "memory", "Concurrent requests",
         "Requests that run at once, sharing the pool; 8 is the Engine's fixed limit.", 1, 8, {}},
        {"--max-pending-requests", "max_pending_requests", ParamKind::Int, "requests",
         "Max pending requests", "Queue depth before new requests are rejected.", 1, 4096, {}},
        {"--pending-timeout", "pending_timeout", ParamKind::Duration, "requests",
         "Pending timeout (ms)", "How long a request may wait to start before it fails.", 1,
         3600000, {}},
        {"--prefill-chunk", "prefill_chunk", ParamKind::Int, "requests", "Prefill chunk",
         "Tokens per prefill step. Larger is faster and needs more workspace.", 128, 65536, {}},

        {"--desktop-reserve", "desktop_reserve", ParamKind::Capacity, "memory",
         "Desktop reserve",
         "GPU memory kept free at startup for the desktop and other apps.", 0, 64LL << 30,
         {}},
        {"--kv-dtype", "kv_dtype", ParamKind::Enum, "memory", "KV format",
         "Smaller formats hold more tokens but can change what the model writes.", 0, 0,
         {"bf16", "int8", "fp8", "nvfp4", "k8v4"}},
        {"--checkpoint-interval", "checkpoint_interval", ParamKind::Int, "memory",
         "Checkpoint interval", "Target spacing in tokens for reusable intermediate checkpoints.", 1, 262144, {}},
        {"--host-cache-capacity", "host_cache_capacity", ParamKind::Capacity, "memory", "Host context capacity",
         "Shared RAM budget for checkpoints, KV and paused requests. Use units such as 8GiB.", 0, 262144LL << 20, {}},

        {"--vision", "vision", ParamKind::Flag, "model", "Vision",
         "Loads the vision encoder. Costs about 1.1 GiB.", 0, 0, {}},
        {"--spec", "spec", ParamKind::Enum, "model", "Speculative backend",
         "Drafts several tokens ahead and checks them in one step, which speeds up replies. MTP drafts with the model's own extra head and suits many requests at once; DFlash2 drafts with a separate small model (a companion artifact) and is fastest for one conversation with the 27B.", 0, 0,
         {"none", "mtp", "dflash", "dflash2"}},
        {"--draft-tokens", "draft_tokens", ParamKind::Int, "model", "Draft tokens",
         "Tokens drafted ahead per step. Start with 5 for 27B MTP or 7 for DFlash2; 0 with speculation off.", 0, 15, {}},
        {"--proposal-head", "proposal_head", ParamKind::Enum, "model", "Proposal head",
         "Which output layer scores the drafts: optimized is a smaller, faster one, full is the model's own. Optimized is recommended for the 27B presets.", 0, 0, {"full", "optimized"}},
        {"--no-prefix-reuse", "no_prefix_reuse", ParamKind::Flag, "requests", "Disable prefix reuse",
         "Turns off cross-request prefix caching. Multi-turn TTFT gets much worse.", 0, 0, {}},
        {"--no-cuda-graph", "no_cuda_graph", ParamKind::Flag, "logs", "Disable CUDA graphs",
         "Runs decode eagerly. Slower; use to isolate a graph-related defect.", 0, 0, {}},
        {"--no-thinking", "no_thinking", ParamKind::Flag, "model", "Disable thinking",
         "Server-wide default. Reasoning is normally a per-request choice.", 0, 0, {}},

        {"--model-id", "model_id", ParamKind::Text, "access", "Model id",
         "The name clients send and see. Overrides the artifact's own id.", 0, 0, {}},
        {"--cors", "cors", ParamKind::Flag, "access", "Send CORS headers",
         "Allows browser pages from other origins to call the engine.", 0, 0, {}},
        {"--max-tokens", "max_tokens", ParamKind::Int, "requests", "Default max tokens",
         "Applied when a request omits max_tokens.", 1, 262144, {}},
        {"--thinking-budget", "thinking_budget", ParamKind::IntOrNone, "requests",
         "Default thinking budget", "Caps model-origin reasoning tokens per request; none disables it.", 1, 262144,
         {}},
        {"--stats-interval", "stats_interval", ParamKind::Duration, "logs",
         "Throughput log interval (ms)", "0 disables the periodic throughput lines.", 0, 3600000,
         {}},
        {"--max-request-size", "max_request_size", ParamKind::Capacity, "requests", "Max request size",
         "Largest accepted request body. Use units such as 384MiB.", 1, 4096LL << 20, {}},
    };
    return specs;
}

inline const EngineParamSpec* find_engine_param(std::string_view key) {
    for (const auto& spec : engine_param_specs()) {
        if (spec.key == key) { return &spec; }
    }
    return nullptr;
}

// Flags whose VALUE is a secret. The value is replaced before the argument vector
// is shown anywhere: supervisor.prod.json embeds a real key, and the dashboard is
// a web page. A path is not a secret; the bytes at that path are.
inline bool engine_flag_value_is_secret(std::string_view flag) {
    return flag == "--api-key";
}

inline std::vector<std::string> redact_engine_args(std::vector<std::string> args) {
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i].starts_with("--api-key=")) {
            args[i] = "--api-key=(redacted)";
        } else if (engine_flag_value_is_secret(args[i]) && i + 1 < args.size() && !args[i + 1].empty()) {
            args[i + 1] = "(redacted)";
        }
    }
    return args;
}

struct EngineParam {
    std::string key;
    std::string value; // Flag kind: "true" when present; absent keys are omitted entirely.
};

// Parsed view of the configured argument vector: recognised parameters, plus every
// token this table does not know, preserved verbatim and in order so that rendering
// after an edit cannot silently drop a flag the operator added by hand.
struct ParsedEngineArgs {
    std::string artifact; // first positional (the .ninfer path), if any
    std::vector<EngineParam> params;
    std::vector<std::string> passthrough;
};

inline ParsedEngineArgs parse_engine_args(const std::vector<std::string>& args) {
    ParsedEngineArgs out;
    bool artifact_taken = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& token = args[i];
        if (!token.starts_with("--")) {
            if (!artifact_taken) {
                out.artifact  = token;
                artifact_taken = true;
            } else {
                out.passthrough.push_back(token);
            }
            continue;
        }
        const EngineParamSpec* spec = nullptr;
        for (const auto& candidate : engine_param_specs()) {
            if (candidate.flag == token) {
                spec = &candidate;
                break;
            }
        }
        if (spec == nullptr) {
            out.passthrough.push_back(token);
            // An unknown flag's value must travel with it, or rendering would
            // promote the value to a positional argument.
            if (i + 1 < args.size() && !args[i + 1].starts_with("--")) {
                out.passthrough.push_back(args[++i]);
            }
            continue;
        }
        if (spec->kind == ParamKind::Flag) {
            out.params.push_back({std::string(spec->key), "true"});
            continue;
        }
        if (i + 1 < args.size()) {
            out.params.push_back({std::string(spec->key), args[++i]});
        }
    }
    return out;
}

inline const std::string* find_param_value(const std::vector<EngineParam>& params,
                                           std::string_view key) {
    for (const auto& p : params) {
        if (p.key == key) { return &p.value; }
    }
    return nullptr;
}

// Rebuilds the argument vector: artifact first, then parameters in table order so
// the file stays diffable across edits, then everything unrecognised, unchanged.
inline std::vector<std::string> render_engine_args(const ParsedEngineArgs& parsed) {
    std::vector<std::string> args;
    if (!parsed.artifact.empty()) { args.push_back(parsed.artifact); }
    for (const auto& spec : engine_param_specs()) {
        const std::string* value = find_param_value(parsed.params, spec.key);
        if (value == nullptr) { continue; }
        if (spec.kind == ParamKind::Flag) {
            if (*value == "true") { args.emplace_back(spec.flag); }
            continue;
        }
        if (value->empty()) { continue; }
        args.emplace_back(spec.flag);
        args.push_back(*value);
    }
    for (const auto& token : parsed.passthrough) { args.push_back(token); }
    return args;
}

inline bool parse_long_long(std::string_view text, long long& out) {
    if (text.empty()) { return false; }
    long long value = 0;
    std::size_t i   = 0;
    const bool neg  = text[0] == '-';
    if (neg) { i = 1; }
    if (i >= text.size()) { return false; }
    for (; i < text.size(); ++i) {
        if (text[i] < '0' || text[i] > '9') { return false; }
        value = value * 10 + (text[i] - '0');
        if (value > 4294967296LL) { return false; }
    }
    out = neg ? -value : value;
    return true;
}

// Returns one message per rejected field. An empty result means the edit is safe
// to write; the caller never applies a partially valid patch.
inline std::vector<std::string> validate_engine_params(const std::vector<EngineParam>& params) {
    std::vector<std::string> errors;
    for (const auto& p : params) {
        const EngineParamSpec* spec = find_engine_param(p.key);
        if (spec == nullptr) {
            errors.push_back(p.key + " is not an editable parameter");
            continue;
        }
        switch (spec->kind) {
        case ParamKind::Flag:
            if (p.value != "true" && p.value != "false") {
                errors.push_back(std::string(spec->label) + " must be true or false");
            }
            break;
        case ParamKind::Int: {
            long long value = 0;
            if (!parse_long_long(p.value, value)) {
                errors.push_back(std::string(spec->label) + " must be a whole number");
            } else if (value < spec->min_value || value > spec->max_value) {
                errors.push_back(std::string(spec->label) + " must be between " +
                                 std::to_string(spec->min_value) + " and " +
                                 std::to_string(spec->max_value));
            }
            break;
        }
        case ParamKind::CapacityOrAuto:
            if (p.value == "auto") break;
            [[fallthrough]];
        case ParamKind::Capacity:
        case ParamKind::Duration: {
            try {
                const auto amount = (spec->kind == ParamKind::Capacity || spec->kind == ParamKind::CapacityOrAuto)
                    ? product::parse_capacity_quantity(p.value)
                    : product::parse_duration_quantity(p.value);
                if (amount < static_cast<std::uint64_t>(spec->min_value) ||
                    amount > static_cast<std::uint64_t>(spec->max_value))
                    errors.push_back(std::string(spec->label) + " is outside its supported range");
            } catch (const std::invalid_argument& error) {
                errors.push_back(std::string(spec->label) + ": " + error.what());
            }
            break;
        }
        case ParamKind::IntOrNone: {
            long long amount = 0;
            if (p.value != "none" && (!parse_long_long(p.value, amount) ||
                amount < spec->min_value || amount > spec->max_value))
                errors.push_back(std::string(spec->label) + " must be a positive budget or none");
            break;
        }
        case ParamKind::Enum: {
            bool ok = false;
            for (const auto& choice : spec->choices) {
                if (!choice.empty() && p.value == choice) {
                    ok = true;
                    break;
                }
            }
            if (!ok) { errors.push_back(std::string(spec->label) + " has an unsupported value"); }
            break;
        }
        case ParamKind::Text:
            // A value is passed to CreateProcess as one argv element and is quoted
            // there, so spaces are safe; control characters are not, and a newline
            // would corrupt the config file this is written back into.
            for (char c : p.value) {
                if (static_cast<unsigned char>(c) < 0x20) {
                    errors.push_back(std::string(spec->label) + " must not contain control "
                                                                "characters");
                    break;
                }
            }
            break;
        }
    }
    return errors;
}

// Cross-field rules the per-field pass cannot see.
inline std::vector<std::string> validate_engine_param_combination(
    const std::vector<EngineParam>& params, std::string_view model_identity = {}) {
    std::vector<std::string> errors;
    const std::string* draft_tokens = find_param_value(params, "draft_tokens");
    const std::string* spec_backend = find_param_value(params, "spec");
    if (draft_tokens != nullptr && *draft_tokens != "0" && spec_backend == nullptr) {
        errors.emplace_back("Draft tokens has no effect without a speculative backend");
    }
    const auto* head = find_param_value(params, "proposal_head");
    if (head != nullptr && *head == "optimized" && spec_backend == nullptr) {
        errors.emplace_back("Optimized draft head requires a speculative backend");
    }
    if (model_identity == "qwen3.8-flash-next" && head != nullptr && *head == "optimized") {
        errors.emplace_back("Flash-Next MTP requires the full output head");
    }
    if (spec_backend != nullptr && *spec_backend == "none") {
        if ((draft_tokens && *draft_tokens != "0") || (head && *head == "optimized"))
            errors.emplace_back("Disabled speculation requires zero drafts and the full head");
    } else if (spec_backend != nullptr) {
        const int limit = *spec_backend == "mtp" ? 5 : 15;
        long long count = 0;
        if (draft_tokens && parse_long_long(*draft_tokens, count) && (count < 1 || count > limit)) {
            errors.emplace_back("Draft tokens must be between 1 and " + std::to_string(limit) +
                                " for this model and backend");
        }
        if (!model_identity.empty() &&
            ((*spec_backend == "dflash2" && model_identity != "qwen3.8-27b" &&
              model_identity != "qwen3.8-27b-orcarouter") ||
             (*spec_backend == "dflash" && model_identity != "qwen3.6-35b-a3b"))) {
            errors.emplace_back("Selected speculative backend is not supported by this model");
        }
    }
    return errors;
}

// Keeps the engine's LISTEN address in step with the address the supervisor polls.
//
// These are two different things in the config -- `--host`/`--port` in the argument
// vector decide where the engine listens, while engine_host/engine_port decide where
// health checks and /admin/vram are fetched from -- and nothing kept them in step.
// Editing the port in the dashboard would otherwise leave the supervisor polling a
// port nothing listens on, which presents as an engine that starts and is then
// permanently unhealthy.
inline std::vector<std::string> with_engine_endpoint(std::vector<std::string> args,
                                                     const std::string& host, int port) {
    if (!host.empty()) args = product::set_serve_argument(std::move(args), "server.host", host);
    if (port > 0) args = product::set_serve_argument(std::move(args), "server.port", std::to_string(port));
    return args;
}

// Adds --request-log-jsonl PATH unless the caller already passed one explicitly, in which
// case the explicit flag wins and the config field is ignored -- an operator who wrote the
// flag by hand meant it. An empty path is a no-op, so a config that does not ask for a
// request log does not get one.
inline std::vector<std::string> with_request_log(std::vector<std::string> args,
                                                 const std::string& path) {
    if (path.empty() || product::has_serve_argument(args, "logging.request_jsonl")) return args;
    return product::set_serve_argument(std::move(args), "logging.request_jsonl", path);
}

// Adds --api-key-file PATH so the engine actually authenticates. Without this the
// config's api_key_file was read only by the supervisor's own collector, and the
// engine itself ran wide open: /v1/models answered 200 with no key and with a wrong
// one, on whatever interface it was bound to.
//
// The PATH is passed, never the key. A child's command line is readable by every
// process running as this user, so --api-key would publish the secret to the
// machine; --api-key-file leaves it in the file it already lives in. An explicit
// --api-key or --api-key-file in args wins, as with the request log.
inline std::vector<std::string> with_api_key_file(std::vector<std::string> args,
                                                  const std::string& path) {
    if (path.empty() || product::has_serve_argument(args, "server.api_key") ||
        product::has_serve_argument(args, "server.api_key_file")) return args;
    return product::set_serve_argument(std::move(args), "server.api_key_file", path);
}

inline std::vector<std::string> with_desktop_reserve(std::vector<std::string> args, int gib) {
    if (desktop_reserve_pinned(args)) return args;
    // Zero must override the engine's nonzero default explicitly.
    return product::set_serve_argument(std::move(args), "engine.desktop_reserve", std::to_string(gib) + "GiB");
}

// The reserve the config file already asks for, so the menu can put a checkmark
// on it before the user has ever chosen one. 0 when the flag is absent.
inline int desktop_reserve_from_args(const std::vector<std::string>& args) {
    try {
        return static_cast<int>(child_startup_options(args).launch.engine.desktop_reserve_bytes >> 30);
    } catch (...) { return 0; }
}

// Empty when the saved flags are not a valid engine command line: the state report still has to
// answer, because the dashboard is where the engine's refusal gets read.
inline std::optional<std::uint64_t> desktop_reserve_launch_mib(const std::vector<std::string>& args,
                                                               int preference) {
    try {
        const auto effective = preference >= 0 ? with_desktop_reserve(args, preference) : args;
        return child_startup_options(effective).launch.engine.desktop_reserve_bytes >> 20;
    } catch (const std::exception&) { return std::nullopt; }
}

enum class TrayAction : std::uint8_t {
    None,
    OpenDashboard,
    Start,
    Stop,
    Restart,
    Quit,
    Reserve,
    Idle,
    StartAtLogin,
    OpenLogs,
    CopyUrl,
    OpenFrontend,
};

struct TrayCommand {
    TrayAction action = TrayAction::None;
    int value         = 0;
};

// Menu command ids are INDICES into the choice tables, never the values
// themselves. Encoding minutes directly put "After 4 hours" at 200+240=440,
// past the next base, where the dispatch guard silently discarded it.
inline constexpr unsigned kCmdFixedBase   = 1;
inline constexpr unsigned kCmdReserveBase = 100;
inline constexpr unsigned kCmdIdleBase    = 200;
inline constexpr unsigned kCmdEnd         = 300;
// One "Open" item per configured web frontend; the configuration caps their number at this.
inline constexpr std::size_t kMaxFrontends   = 16;
inline constexpr unsigned kCmdFrontendBase   = 300;
inline constexpr unsigned kCmdFrontendEnd    = kCmdFrontendBase + kMaxFrontends;

inline constexpr unsigned kCmdOpenDashboard = kCmdFixedBase + 0;
inline constexpr unsigned kCmdStart         = kCmdFixedBase + 1;
inline constexpr unsigned kCmdStop          = kCmdFixedBase + 2;
inline constexpr unsigned kCmdRestart       = kCmdFixedBase + 3;
inline constexpr unsigned kCmdQuit          = kCmdFixedBase + 4;
inline constexpr unsigned kCmdStartAtLogin  = kCmdFixedBase + 6;
inline constexpr unsigned kCmdOpenLogs      = kCmdFixedBase + 7;
inline constexpr unsigned kCmdCopyUrl       = kCmdFixedBase + 8;

static_assert(kCmdCopyUrl < kCmdReserveBase, "fixed ids must not reach the reserve range");
static_assert(kCmdReserveBase + kReserveChoicesGib.size() <= kCmdIdleBase,
              "reserve ids must not reach the idle range");
static_assert(kCmdIdleBase + kIdleChoicesMinutes.size() <= kCmdEnd,
              "idle ids must not leave the dispatch range");
static_assert(kCmdEnd <= kCmdFrontendBase, "frontend ids must not overlap the idle range");

inline unsigned tray_cmd_reserve(std::size_t index) {
    return kCmdReserveBase + static_cast<unsigned>(index);
}
inline unsigned tray_cmd_idle(std::size_t index) {
    return kCmdIdleBase + static_cast<unsigned>(index);
}
inline unsigned tray_cmd_frontend(std::size_t index) {
    return kCmdFrontendBase + static_cast<unsigned>(index);
}

inline TrayCommand decode_tray_command(unsigned cmd) {
    if (cmd >= kCmdReserveBase && cmd < kCmdIdleBase) {
        const std::size_t i = cmd - kCmdReserveBase;
        if (i >= kReserveChoicesGib.size()) { return {}; }
        return {TrayAction::Reserve, kReserveChoicesGib[i]};
    }
    if (cmd >= kCmdIdleBase && cmd < kCmdEnd) {
        const std::size_t i = cmd - kCmdIdleBase;
        if (i >= kIdleChoicesMinutes.size()) { return {}; }
        return {TrayAction::Idle, kIdleChoicesMinutes[i]};
    }
    if (cmd >= kCmdFrontendBase && cmd < kCmdFrontendEnd) {
        return {TrayAction::OpenFrontend, static_cast<int>(cmd - kCmdFrontendBase)};
    }
    switch (cmd) {
    case kCmdOpenDashboard: return {TrayAction::OpenDashboard, 0};
    case kCmdStart: return {TrayAction::Start, 0};
    case kCmdStop: return {TrayAction::Stop, 0};
    case kCmdRestart: return {TrayAction::Restart, 0};
    case kCmdQuit: return {TrayAction::Quit, 0};
    case kCmdStartAtLogin: return {TrayAction::StartAtLogin, 0};
    case kCmdOpenLogs: return {TrayAction::OpenLogs, 0};
    case kCmdCopyUrl: return {TrayAction::CopyUrl, 0};
    default: break;
    }
    return {};
}

// What one line of engine stderr tells the supervisor. The engine already
// writes everything needed here (spdlog pattern `[ts] [level] [name] msg`), so
// this costs no extra I/O: the pipe is read for engine.log regardless.
struct EngineLineSignal {
    bool activity = false;
    bool ready    = false;
    bool error    = false;
    bool fatal    = false;
    std::string message;
};

inline bool contains_sv(std::string_view hay, std::string_view needle) {
    return hay.find(needle) != std::string_view::npos;
}

inline EngineLineSignal classify_engine_line(std::string_view raw) {
    EngineLineSignal out;
    const std::string_view line = trim_sv(raw);
    if (line.empty()) { return out; }

    // Activity: a request arriving or finishing, or the periodic throughput
    // report -- which the engine only writes while something is running or
    // waiting. Either is proof the engine is not idle.
    out.activity = contains_sv(line, "request id=") || contains_sv(line, "throughput interval_ms=");
    out.ready    = contains_sv(line, "server status=ready");
    out.fatal    = contains_sv(line, "server status=failed");
    out.error    = out.fatal || contains_sv(line, "] [error] ") ||
                contains_sv(line, "] [critical] ") || line.rfind("ninfer-serve: ", 0) == 0;

    // The message is what a person needs in a balloon. Two producers: the
    // logger (`[ninfer-serve] msg`) and the argument parser, which fails before
    // the logger exists and writes `ninfer-serve: msg` straight to stderr.
    constexpr std::string_view kCerr   = "ninfer-serve: ";
    constexpr std::string_view kLogger = "[ninfer-serve] ";
    std::string_view msg;
    if (const auto p = line.find(kCerr); p != std::string_view::npos) {
        msg = line.substr(p + kCerr.size());
    } else if (const auto q = line.find(kLogger); q != std::string_view::npos) {
        msg = line.substr(q + kLogger.size());
    }
    msg = trim_sv(msg);
    if (msg.size() > 200) { msg = msg.substr(0, 200); }
    out.message = std::string(msg);
    return out;
}

// Is this error about the ENGINE, or about one request?
//
// `[error] [ninfer-serve] request id=812 status=failed ...` is an internal
// failure of a single request; the engine is still up and serving everything
// else. Storing it as the engine's "why" put a stale, unrelated request failure
// in the tooltip and in the "Last error:" clause of a later crash-loop balloon,
// attributing the halt to it. A fatal line stays process-scoped even though it
// can mention a request.
inline bool is_process_scoped_error(const EngineLineSignal& sig) {
    return sig.error && (sig.fatal || !sig.activity);
}

// One request's lifecycle, from the same stderr line the activity signal reads.
//
// This is the hard floor under "unload when idle": the log-cadence signals can
// all be silenced (--log-stats-interval-ms 0 removes the throughput line, and a
// config need not have a request log), and a 20-minute generation would then
// look idle and be terminated mid-answer. A request that opened and has not
// closed is proof of work in flight regardless of any cadence.
//
// Formats (src/serve/operational_log.cpp): open is `request id=<n>
// status=submitted ...`; close is status=done / failed / rejected / cancelled.
// `response request_id=` is deliberately not matched -- it is a second record
// about a request that has its own terminal line.
struct RequestLifecycle {
    bool matched         = false; // the line is about a specific request
    std::uint64_t id     = 0;
    bool opens           = false;
    bool closes          = false;
};

inline RequestLifecycle classify_request_line(std::string_view raw) {
    RequestLifecycle out;
    const std::string_view line = trim_sv(raw);
    constexpr std::string_view kReq = "request id=";
    const auto p                    = line.find(kReq);
    if (p == std::string_view::npos) { return out; }
    std::size_t i          = p + kReq.size();
    std::uint64_t id       = 0;
    std::size_t digits     = 0;
    while (i < line.size() && line[i] >= '0' && line[i] <= '9') {
        // Ignore overflow past 19 digits rather than wrapping: an id that long
        // is not something this engine emits, and a wrapped one could collide.
        if (digits < 19) { id = id * 10 + static_cast<std::uint64_t>(line[i] - '0'); }
        ++digits;
        ++i;
    }
    if (digits == 0 || digits > 19) { return out; }
    out.matched = true;
    out.id      = id;
    constexpr std::string_view kStatus = " status=";
    const auto s                       = line.find(kStatus, i);
    if (s == std::string_view::npos) { return out; }
    const std::size_t b = s + kStatus.size();
    std::size_t e       = b;
    while (e < line.size() && line[e] != ' ') { ++e; }
    const std::string_view status = line.substr(b, e - b);
    out.opens                     = status == "submitted";
    out.closes = status == "done" || status == "failed" || status == "rejected" ||
                 status == "cancelled";
    return out;
}

// Inputs to "unload the model because nobody is using it". Kept as a struct so
// the decision is one pure function: killing a live engine mid-request is the
// worst thing this program can do, so the rule must be testable in isolation.
struct IdleInputs {
    std::int64_t now_ms           = 0;
    std::int64_t last_activity_ms = 0;
    int idle_minutes              = 0;
    bool running                  = false;
    bool healthy                  = false;
    // Requests the engine has opened and not closed. The activity clock depends
    // on the engine's log cadence; this does not, and it is what makes "cannot
    // unload mid-request" absolute rather than conditional on a flag.
    int inflight_requests         = 0;
};

inline bool idle_unload_due(const IdleInputs& in) {
    if (in.idle_minutes <= 0 || !in.running || !in.healthy) { return false; }
    if (in.inflight_requests > 0) { return false; }
    // No activity clock yet (engine just adopted, log not yet seen): never
    // unload. An unknown last-use time is not the same as a long-ago one.
    if (in.last_activity_ms <= 0) { return false; }
    const std::int64_t window_ms = static_cast<std::int64_t>(in.idle_minutes) * 60000;
    return in.now_ms - in.last_activity_ms >= window_ms;
}

// Balloon body for a crash-loop halt. NOTIFYICONDATA::szInfo is 256 wchar_t
// including the terminator, so 255 is the hard ceiling and the error text is
// what gets trimmed, never the counts.
inline std::string format_halt_notice(int recent_exits, int window_s, int exit_code,
                                      std::string_view last_error) {
    constexpr std::size_t kMax = 255;
    const std::string tail     = " Open the dashboard for the log.";
    std::string out = "Crashed " + std::to_string(recent_exits) + " times in " +
                      std::to_string(window_s) + " s (exit code " + std::to_string(exit_code) + ").";
    const std::string_view err = trim_sv(last_error);
    const std::size_t fixed    = out.size() + tail.size() + 14; // " Last error: " + "."
    if (!err.empty() && fixed < kMax) {
        std::string e(err);
        const std::size_t room = kMax - fixed;
        if (e.size() > room) { e.resize(room); }
        out += " Last error: " + e + ".";
    }
    if (out.size() + tail.size() <= kMax) { out += tail; }
    if (out.size() > kMax) { out.resize(kMax); }
    return out;
}

// The model a person recognises, for the menu header. The checkpoint path is the
// one argument that is always present and always specific.
inline std::string model_label_from_args(const std::vector<std::string>& args) {
    std::string picked;
    for (const auto& a : args) {
        if (a.size() > 7 && a.compare(a.size() - 7, 7, ".ninfer") == 0) {
            picked = a;
            break;
        }
    }
    if (picked.empty() && !args.empty() && !args.front().empty() && args.front().front() != '-') {
        picked = args.front();
    }
    if (picked.empty()) { return {}; }
    const auto slash = picked.find_last_of("/\\");
    std::string base = slash == std::string::npos ? picked : picked.substr(slash + 1);
    if (base.size() > 7 && base.compare(base.size() - 7, 7, ".ninfer") == 0) {
        base.resize(base.size() - 7);
    }
    return base;
}

// Named-mutex identity for "a supervisor is already running for this config".
// Keyed on the config path rather than on the executable: two configs are two
// engines and may legitimately coexist, while two supervisors on one config race
// for the same engine port and crash-loop each other. Hashed because a kernel
// object name may not contain a backslash after the namespace prefix.
inline std::string single_instance_name(std::string_view canonical_config_path) {
    std::uint64_t h = 1469598103934665603ull; // FNV-1a 64 offset basis
    for (const char c : canonical_config_path) {
        auto u = static_cast<unsigned char>(c);
        if (u == '\\') { u = '/'; }
        if (u >= 'A' && u <= 'Z') { u = static_cast<unsigned char>(u - 'A' + 'a'); }
        h ^= u;
        h *= 1099511628211ull;
    }
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out              = "Local\\NInferSupervisor.";
    for (int i = 15; i >= 0; --i) {
        out.push_back(kHex[(h >> (i * 4)) & 0xFull]);
    }
    return out;
}

inline std::string run_at_login_command(std::string_view module_path,
                                        std::string_view config_abs_path) {
    return "\"" + std::string(module_path) + "\" --config \"" + std::string(config_abs_path) + "\"";
}

// "Start at login" must mean "at login, for THIS config". There is one Run
// value name, so testing that it merely exists put a checkmark on the menu of
// whichever config happened to be running -- including the one the entry does
// not point at, whose toggle would then delete the other's entry. Compare the
// stored command instead. Whitespace-tolerant because the value may have been
// hand-edited in regedit; slash- and case-insensitive because the installer
// and regedit write native "C:\..." paths while the supervisor spells its
// canonical config path with forward slashes. Comparing those literally made
// the installer's own entry show as unchecked, and a click then replaced or
// removed an entry that was already correct.
inline bool run_at_login_command_matches(std::string_view stored, std::string_view expected) {
    const auto fold = [](std::string_view s) {
        std::string out(trim_sv(s));
        for (char& c : out) {
            if (c == '\\') { c = '/'; }
            if (c >= 'A' && c <= 'Z') { c = static_cast<char>(c - 'A' + 'a'); }
        }
        return out;
    };
    const std::string want = fold(expected);
    return !want.empty() && fold(stored) == want;
}

} // namespace ninfer::supervisor
