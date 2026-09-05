#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
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

struct NvidiaSmiMemory {
    bool ok                 = false;
    int index               = -1;
    std::uint64_t used_mib  = 0;
    std::uint64_t total_mib = 0;
    std::string error;
};

// Parses `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits`.
// Values are mebibytes. Picks the row whose index equals `device`.
inline NvidiaSmiMemory parse_nvidia_smi_memory_csv(std::string_view csv, int device) {
    NvidiaSmiMemory out;
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

inline std::string extract_kv_capacity_line(std::string_view log) {
    const auto key = std::string_view("KV capacity ");
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
// not silently lower a reserve nobody asked it to change. The floor is measured
// rather than chosen: on this card a desktop starts missing its 16.7 ms frame
// budget below ~1.7 GiB free and shows nothing measurable above ~3 GiB. So 2 GiB
// is the smallest honest option, and 4 is the one to recommend -- twice the
// measured cliff, leaving room for a real desktop's several applications rather
// than the one synthetic canary that produced the number.
inline constexpr std::array<int, 7> kReserveChoicesGib{0, 2, 3, 4, 6, 8, 12};

// A duration, not a toggle. Unloading after five minutes would punish anyone who
// steps away mid-task; an hour reliably catches "finished for the day".
inline constexpr std::array<int, 4> kIdleChoicesMinutes{0, 15, 60, 240};

// Sentinel for TrayPrefs::desktop_reserve_gib: the user has never picked one, so
// whatever the config file says stays untouched. Distinct from 0 ("I chose the
// engine default"), which does strip the flag.
inline constexpr int kReserveUnset = -1;

// Rewrites --desktop-reserve-gib in an engine argument vector.
//  - an existing value is replaced in place
//  - a trailing "--desktop-reserve-gib" with no value gets one appended
//  - --desktop-reserve-mib is more specific than anything the menu can express,
//    so its presence makes this a no-op rather than a silent coarsening
//  - gib <= 0 removes the pair, leaving the engine's own default in force
//
// A config that pins --desktop-reserve-mib owns the reserve outright: the menu
// cannot express MiB, so every choice in it would be a silent coarsening. The
// UI needs to know this too -- an editable submenu whose every entry is a no-op
// is worse than a greyed one that says why -- so the test is its own function
// rather than a loop buried in the rewrite.
inline bool desktop_reserve_pinned(const std::vector<std::string>& args) {
    for (const auto& a : args) {
        if (a == "--desktop-reserve-mib") { return true; }
    }
    return false;
}

inline std::vector<std::string> with_desktop_reserve(std::vector<std::string> args, int gib) {
    if (desktop_reserve_pinned(args)) { return args; }
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i] != "--desktop-reserve-gib") { continue; }
        const bool has_value = i + 1 < args.size();
        if (gib <= 0) {
            args.erase(args.begin() + static_cast<std::ptrdiff_t>(i),
                       args.begin() + static_cast<std::ptrdiff_t>(i + (has_value ? 2 : 1)));
            return args;
        }
        if (has_value) {
            args[i + 1] = std::to_string(gib);
        } else {
            args.emplace_back(std::to_string(gib));
        }
        return args;
    }
    if (gib > 0) {
        args.emplace_back("--desktop-reserve-gib");
        args.emplace_back(std::to_string(gib));
    }
    return args;
}

// The reserve the config file already asks for, so the menu can put a checkmark
// on it before the user has ever chosen one. 0 when the flag is absent.
inline int desktop_reserve_from_args(const std::vector<std::string>& args) {
    for (std::size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] != "--desktop-reserve-gib") { continue; }
        try {
            return std::stoi(args[i + 1]);
        } catch (...) { return 0; }
    }
    return 0;
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
    ReleaseCache,
    StartAtLogin,
    OpenLogs,
    CopyUrl,
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

inline constexpr unsigned kCmdOpenDashboard = kCmdFixedBase + 0;
inline constexpr unsigned kCmdStart         = kCmdFixedBase + 1;
inline constexpr unsigned kCmdStop          = kCmdFixedBase + 2;
inline constexpr unsigned kCmdRestart       = kCmdFixedBase + 3;
inline constexpr unsigned kCmdQuit          = kCmdFixedBase + 4;
inline constexpr unsigned kCmdReleaseCache  = kCmdFixedBase + 5;
inline constexpr unsigned kCmdStartAtLogin  = kCmdFixedBase + 6;
inline constexpr unsigned kCmdOpenLogs      = kCmdFixedBase + 7;
inline constexpr unsigned kCmdCopyUrl       = kCmdFixedBase + 8;

static_assert(kCmdCopyUrl < kCmdReserveBase, "fixed ids must not reach the reserve range");
static_assert(kCmdReserveBase + kReserveChoicesGib.size() <= kCmdIdleBase,
              "reserve ids must not reach the idle range");
static_assert(kCmdIdleBase + kIdleChoicesMinutes.size() <= kCmdEnd,
              "idle ids must not leave the dispatch range");

inline unsigned tray_cmd_reserve(std::size_t index) {
    return kCmdReserveBase + static_cast<unsigned>(index);
}
inline unsigned tray_cmd_idle(std::size_t index) {
    return kCmdIdleBase + static_cast<unsigned>(index);
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
    switch (cmd) {
    case kCmdOpenDashboard: return {TrayAction::OpenDashboard, 0};
    case kCmdStart: return {TrayAction::Start, 0};
    case kCmdStop: return {TrayAction::Stop, 0};
    case kCmdRestart: return {TrayAction::Restart, 0};
    case kCmdQuit: return {TrayAction::Quit, 0};
    case kCmdReleaseCache: return {TrayAction::ReleaseCache, 0};
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
// hand-edited in regedit.
inline bool run_at_login_command_matches(std::string_view stored, std::string_view expected) {
    return !trim_sv(expected).empty() && trim_sv(stored) == trim_sv(expected);
}

} // namespace ninfer::supervisor
