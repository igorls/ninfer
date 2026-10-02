#include "collector.hpp"
#include "gpu_processes.hpp"
#include "insights.hpp"
#include "run_at_login.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#include <httplib.h>

#include <windows.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

namespace ninfer::supervisor {
namespace {

// The engine reports throughput every ~5 s. Past this the last report describes
// a period that has ended, so it is shown as idle rather than as a live rate.
constexpr std::int64_t kThroughputStaleMs = 12'000;

// Rolling window for the connected-clients panel. Long enough that an app pausing
// between turns keeps its place, short enough that the list describes now.
constexpr std::int64_t kClientWindowMs = 15 * 60 * 1000;

// The request panel describes the newest request_done records.
constexpr std::size_t kRecentDones = 32;

// History read per poll of the request log while catching up. Each poll also follows what
// was appended, so this bounds how stale live figures can get during a catch-up.
constexpr std::uint64_t kRequestLogHistoryBytesPerPoll = 8ull << 20;

// How often the request log is checked for appended records once caught up.
constexpr auto kRequestLogPollPeriod = std::chrono::milliseconds(500);

// Keeps the newest request_done lines and the latest server_start. Returns whether
// anything changed.
bool note_recent_line(std::string_view line, std::deque<std::string>& dones,
                      std::string& server_start) {
    if (jsonl_event_is(line, "request_done")) {
        dones.emplace_back(line);
        if (dones.size() > kRecentDones) { dones.pop_front(); }
        return true;
    }
    if (jsonl_event_is(line, "server_start")) {
        server_start.assign(line);
        return true;
    }
    return false;
}

// The request panel's per-response figures from the newest request_done lines, oldest first.
void fill_recent_mix(const std::vector<std::string_view>& lines, RequestMix& out) {
    double ttft_sum   = 0;
    double decode_sum = 0;
    int n_ttft        = 0;
    int n_dec         = 0;
    for (const std::string_view line : lines) {
        try {
            const auto j = nlohmann::json::parse(line);
            // The engine writes {"event":"request_done"}, not "type". Reading the wrong
            // key made every record fall through and the panel read a permanent 0.
            if (j.value("event", "") != "request_done") { continue; }
            ++out.done;
            if (j.contains("speculative") && j.at("speculative").is_object()) {
                const auto& sp = j.at("speculative");
                out.mtp_backend      = sp.value("backend", out.mtp_backend);
                out.mtp_draft_window = sp.value("draft_window", out.mtp_draft_window);
                const auto drafted   = sp.value("drafted_tokens", 0);
                const auto accepted  = sp.value("accepted_tokens", 0);
                out.mtp_drafted += drafted;
                out.mtp_accepted += accepted;
                out.mtp_fallback_steps += sp.value("fallback_steps", 0);
                out.mtp_rounds += sp.value("rounds", 0);
                if (drafted > 0) {
                    out.mtp_last_accept_rate =
                        static_cast<double>(accepted) / static_cast<double>(drafted);
                }
                if (sp.contains("accepted_per_position") && sp.at("accepted_per_position").is_array()) {
                    const auto& pos = sp.at("accepted_per_position");
                    if (out.mtp_accepted_per_position.size() < pos.size()) {
                        out.mtp_accepted_per_position.resize(pos.size(), 0);
                    }
                    for (std::size_t p = 0; p < pos.size(); ++p) {
                        out.mtp_accepted_per_position[p] += pos.at(p).get<std::uint64_t>();
                    }
                }
            }
            if (j.contains("timings_seconds") && j.at("timings_seconds").contains("ttft")) {
                ttft_sum += j.at("timings_seconds").at("ttft").get<double>() * 1000.0;
                ++n_ttft;
            }
            const auto& result = j.at("result");
            const double dec_s =
                j.contains("timings_seconds") ? j.at("timings_seconds").value("decode", 0.0) : 0.0;
            const int gen = result.value("completion_tokens", 0);
            if (dec_s > 0.0 && gen > 1) {
                decode_sum += static_cast<double>(gen - 1) / dec_s;
                ++n_dec;
            }
            const std::string reuse = result.value("prefix_reuse_path", "");
            out.last_reuse = reuse;
            if (reuse == "full_reset") {
                ++out.reuse_full_reset;
            } else if (reuse.find("append") != std::string::npos) {
                ++out.reuse_append;
            } else if (reuse.find("seed") != std::string::npos ||
                       reuse.find("restore") != std::string::npos) {
                ++out.reuse_seed;
            } else if (!reuse.empty()) {
                ++out.reuse_other;
            }
        } catch (...) {}
    }
    if (n_ttft != 0) { out.ttft_ms_mean = ttft_sum / n_ttft; }
    if (n_dec != 0) { out.decode_tok_s_mean = decode_sum / n_dec; }
}

// The KV capacity line from the latest server_start, for when the engine's own log has none.
std::string capacity_line_from_server_start(const std::string& line) {
    if (line.empty()) { return {}; }
    try {
        const auto j = nlohmann::json::parse(line);
        const auto& eng = j.at("engine");
        const auto& mem = j.at("memory");
        auto gib = [](const nlohmann::json& obj, const char* key) {
            const auto n = obj.value(key, std::uint64_t{0});
            return std::to_string(n / 1048576) + " MiB";
        };
        return std::string("KV capacity ") + eng.value("kv_capacity_mode", std::string("?")) +
               " resolved=" + std::to_string(eng.value("kv_capacity", 0)) +
               " tokens pages=" + std::to_string(eng.value("kv_capacity_page_groups", 0)) + "/" +
               std::to_string(eng.value("kv_capacity_max_page_groups", 0)) +
               " runtime=" + gib(mem, "runtime_reservation_bytes") +
               " prefix-cache=" + gib(mem, "prefix_cache_bytes") +
               " free-after-weights=" + gib(mem, "available_after_weights_bytes") +
               " free-after-startup=" + gib(mem, "available_after_startup_bytes") +
               " headroom=" + gib(mem, "kv_capacity_headroom_bytes") +
               " slack=" + gib(mem, "planned_slack_bytes") +
               " graphs=" + gib(mem, "cuda_graph_observed_bytes") + "/" +
               gib(mem, "cuda_graph_allowance_bytes") + " (from request-log server_start)";
    } catch (...) { return {}; }
}

// Runs a console command with no visible window and captures its stdout. The supervisor is a
// windowless process, so `_popen` (which goes through cmd.exe) flashed a console on every poll.
bool run_hidden_capture(const std::string& command_line, std::string& output, int& exit_code) {
    SECURITY_ATTRIBUTES sa{};
    sa.nLength        = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE out_r = nullptr;
    HANDLE out_w = nullptr;
    if (!CreatePipe(&out_r, &out_w, &sa, 0)) { return false; }
    SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW si{};
    si.cb         = sizeof(si);
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdOutput = out_w;
    si.hStdError  = out_w;
    si.hStdInput  = nullptr;

    // The command line is ASCII (a fixed nvidia-smi invocation); widen it byte by byte.
    std::vector<wchar_t> cmd_buf;
    cmd_buf.reserve(command_line.size() + 1);
    for (const char c : command_line) {
        cmd_buf.push_back(static_cast<wchar_t>(static_cast<unsigned char>(c)));
    }
    cmd_buf.push_back(L'\0');

    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, cmd_buf.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr,
                        nullptr, &si, &pi)) {
        CloseHandle(out_r);
        CloseHandle(out_w);
        return false;
    }
    CloseHandle(out_w);
    CloseHandle(pi.hThread);

    char buf[512];
    DWORD got = 0;
    while (ReadFile(out_r, buf, sizeof(buf), &got, nullptr) && got > 0) { output.append(buf, got); }
    CloseHandle(out_r);

    WaitForSingleObject(pi.hProcess, 10'000);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    exit_code = static_cast<int>(code);
    return true;
}

std::string load_key(const std::string& path) {
    try {
        return read_api_key(path);
    } catch (...) { return {}; }
}

httplib::Client engine_client(const EngineSpec& spec) {
    httplib::Client cli(engine_connect_host(spec), spec.engine_port);
    cli.set_connection_timeout(1, 0);
    cli.set_read_timeout(2, 0);
    const std::string key = load_key(spec.api_key_file);
    if (!key.empty()) { cli.set_bearer_token_auth(key); }
    return cli;
}

} // namespace

Collector::Collector(EngineSpec spec, std::string logs_dir)
    : spec_(std::move(spec)), logs_dir_(std::move(logs_dir)), series_(6000),
      request_log_path_w_(widen_utf8(spec_.request_log)) {}

void Collector::poll_health(Collected& out) {
    auto cli = engine_client(spec_);
    if (auto res = cli.Get("/health")) {
        out.health_status = res->status;
        out.health_body   = res->body;
    } else {
        out.health_status = 0;
        out.health_body   = "unreachable";
    }
}

void Collector::poll_admin(Collected& out) {
    auto cli = engine_client(spec_);
    if (auto res = cli.Get("/admin/vram")) {
        if (res->status == 200) {
            try {
                out.admin_vram = nlohmann::json::parse(res->body);
            } catch (...) {
                out.admin_vram_note = "admin/vram returned unreadable JSON";
            }
        } else if (res->status == 401 || res->status == 403) {
            out.admin_vram_note = "admin VRAM unavailable (enable --admin-vram and --api-key)";
        } else if (res->status == 404) {
            out.admin_vram_note = "admin VRAM not registered on this engine";
        } else {
            out.admin_vram_note = "admin/vram HTTP " + std::to_string(res->status);
        }
    } else {
        out.admin_vram_note = "engine unreachable for admin/vram";
    }
}

// Device-wide memory: NVML in process (one session for the supervisor's lifetime), with
// nvidia-smi only when NVML cannot load.
void Collector::poll_device_memory(Collected& out) {
    out.nvidia = nvml_.query(spec_.device);
    if (out.nvidia.ok || nvml_.available()) { return; }
    std::string csv;
    int rc = 0;
    if (!run_hidden_capture("nvidia-smi --query-gpu=index,memory.used,memory.total "
                            "--format=csv,noheader,nounits",
                            csv, rc)) {
        out.nvidia.error = "nvidia-smi not found";
        return;
    }
    if (rc != 0 && csv.empty()) {
        out.nvidia.error = "nvidia-smi exited " + std::to_string(rc);
        return;
    }
    out.nvidia = parse_nvidia_smi_memory_csv(csv, spec_.device);
}

void Collector::poll_request_log(Collected& out) {
    if (spec_.request_log.empty()) {
        out.requests.log_error = "request log path not configured";
        return;
    }
    // request_log_thread_ keeps the fold current. Before it exists, read here; the first
    // call reads the whole log.
    if (!request_log_threaded_.load()) {
        while (poll_request_log_file()) {}
    }
    {
        std::lock_guard lock(log_mu_);
        switch (log_poll_.status) {
        case RequestLogStatus::Ok: break;
        case RequestLogStatus::NotPolled: out.requests.log_error = "request log not read yet"; return;
        case RequestLogStatus::Missing: out.requests.log_error = "request log not present"; return;
        case RequestLogStatus::Unreadable:
            out.requests.log_error =
                "request log unreadable (Windows error " + std::to_string(log_poll_.error) + ")";
            return;
        }
        if (recent_cache_version_ != recent_version_) {
            // The window's lines first; history before the window only once it has been read,
            // and only if the window held fewer than the panel shows.
            std::vector<std::string_view> lines;
            if (log_history_ready_ && recent_dones_.size() < kRecentDones) {
                const std::size_t want =
                    std::min(kRecentDones - recent_dones_.size(), older_dones_.size());
                for (std::size_t i = older_dones_.size() - want; i < older_dones_.size(); ++i) {
                    lines.emplace_back(older_dones_[i]);
                }
            }
            for (const auto& line : recent_dones_) { lines.emplace_back(line); }
            recent_mix_cache_ = RequestMix{};
            fill_recent_mix(lines, recent_mix_cache_);
            recent_capacity_cache_ = capacity_line_from_server_start(
                !recent_server_start_.empty() ? recent_server_start_
                : log_history_ready_          ? older_server_start_
                                              : std::string{});
            recent_cache_version_ = recent_version_;
        }
        out.requests             = recent_mix_cache_;
        out.engine_capacity_line = recent_capacity_cache_;
    }
    out.requests.log_available = true;
    // The aggregate comes from the engine's throughput reports, which request_log_thread_
    // folds as they are appended. Recomputing it from the 32 lines above would report one
    // response's speed again under a different name.
    {
        std::lock_guard lock(mu_);
        // The newest engine report, and only if it is recent: a stale one would
        // keep claiming the card is busy long after it went quiet.
        const auto samples = throughput_.samples();
        if (!samples.empty() && now_ms() - samples.back().t_ms <= kThroughputStaleMs) {
            out.requests.decode_tok_s_total  = samples.back().decode_tok_s;
            out.requests.prefill_tok_s_total = samples.back().prefill_tok_s;
            out.requests.running_requests    = samples.back().running;
        }
        out.requests.clients                = summarize_clients(client_window_);
        out.requests.clients_window_minutes = static_cast<int>(kClientWindowMs / 60000);
    }
}

std::int64_t Collector::now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

// The memory series is sampled at 10 Hz and appended to series.jsonl on every tick, but
// only the last 2 MiB (about ten minutes, the size of the in-memory ring) is ever read
// back. Left alone the file grew without bound: 638 MB and 7.6 million lines after two
// weeks, all of it unread. Keep the file the size of what we load.
constexpr std::int64_t kSeriesKeepBytes   = 2LL * 1024 * 1024;
constexpr std::int64_t kSeriesRotateBytes = 32LL * 1024 * 1024;

// Reads the last kSeriesKeepBytes of the series file, aligned to a line boundary.
static std::string read_series_tail(const std::string& path, std::int64_t& size_out) {
    size_out = 0;
    std::ifstream in(path, std::ios::binary);
    if (!in) { return {}; }
    in.seekg(0, std::ios::end);
    size_out = static_cast<std::int64_t>(in.tellg());
    if (size_out > kSeriesKeepBytes) { in.seekg(size_out - kSeriesKeepBytes, std::ios::beg); }
    else {
        in.seekg(0, std::ios::beg);
    }
    std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto nl = body.find('\n');
    if (size_out > kSeriesKeepBytes && nl != std::string::npos) { body.erase(0, nl + 1); }
    return body;
}

// Rewrites the series file to hold only `tail`, through a temp file and rename so a crash
// mid-write leaves either the old file or the new one, never a torn one.
static void rewrite_series_file(const std::string& path, const std::string& tail) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) { return; }
        out << tail;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) { std::filesystem::remove(tmp, ec); }
}

void Collector::load_persisted_series() {
    if (logs_dir_.empty()) { return; }
    std::filesystem::create_directories(logs_dir_);
    series_path_ = (std::filesystem::path(logs_dir_) / "series.jsonl").string();
    std::int64_t size = 0;
    const std::string body = read_series_tail(series_path_, size);
    if (!body.empty()) { series_.load_jsonl(body); }
    if (size > kSeriesRotateBytes) { rewrite_series_file(series_path_, body); }
    series_file_.open(series_path_, std::ios::app);
    series_bytes_since_check_ = 0;
}

// Caller holds mu_. Every so often, if the file has outgrown the rotate threshold, shrink it
// back to the tail we would load anyway. Cheap: one stat per ~10k samples.
void Collector::rotate_series_file_locked() {
    if (!series_file_.is_open() || series_path_.empty()) { return; }
    series_bytes_since_check_ = 0;
    std::error_code ec;
    const auto size = static_cast<std::int64_t>(std::filesystem::file_size(series_path_, ec));
    if (ec || size <= kSeriesRotateBytes) { return; }
    series_file_.close();
    std::int64_t unused = 0;
    rewrite_series_file(series_path_, read_series_tail(series_path_, unused));
    series_file_.open(series_path_, std::ios::app);
}

void Collector::persist_sample(const VramSample& s) {
    if (!series_file_.is_open()) { return; }
    const std::string line = format_series_sample_line(s);
    series_file_ << line << '\n';
    series_file_.flush();
    series_bytes_since_check_ += static_cast<std::int64_t>(line.size()) + 1;
    if (series_bytes_since_check_ >= kSeriesKeepBytes) { rotate_series_file_locked(); }
}

void Collector::persist_event(const VramSeriesEvent& e) {
    if (!series_file_.is_open()) { return; }
    series_file_ << format_series_event_line(e) << '\n';
    series_file_.flush();
}

void Collector::start_series() {
    bool expected = false;
    if (!series_run_.compare_exchange_strong(expected, true)) { return; }
    load_persisted_series();
    series_thread_  = std::thread([this] { series_loop(); });
    observe_thread_ = std::thread([this] { observe_loop(); });
    if (!spec_.request_log.empty()) {
        request_log_threaded_ = true;
        request_log_thread_   = std::thread([this] { request_log_loop(); });
    }
}

void Collector::stop_series() {
    series_run_ = false;
    if (series_thread_.joinable()) { series_thread_.join(); }
    if (observe_thread_.joinable()) { observe_thread_.join(); }
    if (request_log_thread_.joinable()) { request_log_thread_.join(); }
}

// One stat() per second. The request log is appended and flushed per record, so
// its mtime is a precise "last time a request touched this engine" -- and unlike
// the stderr scan it survives a supervisor restart. Computed as a delta against
// the file clock's own now(), which avoids clock_cast and its libstdc++/MSVC
// differences.
std::int64_t Collector::poll_request_log_mtime() const {
    if (spec_.request_log.empty()) { return 0; }
    std::error_code ec;
    const auto ft = std::filesystem::last_write_time(spec_.request_log, ec);
    if (ec) { return 0; }
    const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::filesystem::file_time_type::clock::now() - ft)
                         .count();
    const std::int64_t ms = now_ms() - static_cast<std::int64_t>(age);
    return ms > 0 ? ms : 0;
}

NvidiaMemory Collector::last_nvidia() {
    std::lock_guard lock(mu_);
    return last_nvidia_;
}

std::int64_t Collector::request_log_last_write_unix_ms() {
    std::lock_guard lock(mu_);
    return request_log_mtime_ms_;
}

void Collector::observe_loop() {
    // Health and /admin/vram are HTTP. They do not belong on the 10 Hz DXGI
    // loop. They also cannot live only inside snapshot() — that is demand-driven
    // by /api/state, so a dashboard that is closed records nothing. 1 Hz is the
    // heartbeat: slow enough not to compete with the engine, fast enough that a
    // 5 s reclaim is visible even with nobody watching.
    while (series_run_.load()) {
        const auto t0 = std::chrono::steady_clock::now();
        try {
            Collected tmp;
            poll_health(tmp);
            poll_admin(tmp);
            record_transitions(tmp);
            // Outside mu_: the observer reaches into EngineChild, which takes its
            // own lock and may restart the engine.
            if (health_observer_) { health_observer_(tmp.health_status); }
            if (engine_state_provider_) {
                const auto st = engine_state_provider_();
                note_engine_state(st.first, st.second);
            }
            const std::int64_t log_mtime = poll_request_log_mtime();
            std::lock_guard lock(mu_);
            request_log_mtime_ms_ = log_mtime;
            detector_last_ran_ms_ = now_ms();
        } catch (...) {
            std::lock_guard lock(mu_);
            detector_last_ran_ms_ = now_ms();
        }
        const auto elapsed = std::chrono::steady_clock::now() - t0;
        const auto period  = std::chrono::milliseconds(1000);
        if (elapsed < period) { std::this_thread::sleep_for(period - elapsed); }
    }
}

// Only lines appended after the supervisor first saw the log reach here. Replaying hours of
// history would fill the ring with samples from before this supervisor could have been
// watching, and the chart claims to be live.
void Collector::fold_live_line_locked(std::string_view line) {
    // request_start names the client; request_done carries what it cost. The
    // pair is joined on request_id so the panel can attribute re-prefills to
    // the app responsible for them.
    if (jsonl_event_is(line, "request_start")) {
        try {
            const auto j = nlohmann::json::parse(line);
            const auto& r = j.at("request");
            pending_clients_.insert_or_assign(
                r.value("request_id", std::uint64_t{0}),
                std::pair{r.value("client", std::string{}), r.value("tool_count", 0)});
            // An engine that dies mid-request leaves its starts unmatched. Cap
            // the map so a crash loop cannot grow it without bound.
            if (pending_clients_.size() > 512) { pending_clients_.clear(); }
        } catch (...) {}
        return;
    }
    if (jsonl_event_is(line, "request_done")) {
        try {
            const auto j  = nlohmann::json::parse(line);
            const auto& r = j.at("result");
            const auto id = j.at("request").value("request_id", std::uint64_t{0});
            ClientRequest c;
            c.t_ms          = j.value("timestamp_unix_ms", std::int64_t{0});
            c.prompt_tokens = r.value("prompt_tokens", std::uint64_t{0});
            c.refill_tokens = r.value("computed_prefill_tokens", std::uint64_t{0});
            const std::string path = r.value("prefix_reuse_path", std::string{});
            c.from_root = path == "root" || path == "full_reset";
            c.ttft_ms   = j.contains("timings_seconds")
                              ? j.at("timings_seconds").value("ttft", 0.0) * 1000.0
                              : 0.0;
            if (const auto it = pending_clients_.find(id); it != pending_clients_.end()) {
                c.client     = it->second.first;
                c.tool_count = it->second.second;
                pending_clients_.erase(it);
            }
            if (c.t_ms != 0) { client_window_.push_back(std::move(c)); }
        } catch (...) {}
        return;
    }
    if (!jsonl_event_is(line, "throughput")) { return; }
    try {
        const auto j = nlohmann::json::parse(line);
        if (!j.contains("throughput_tokens_per_second")) { return; }
        const auto& tp = j.at("throughput_tokens_per_second");
        ThroughputSample s;
        s.t_ms          = j.value("timestamp_unix_ms", std::int64_t{0});
        s.decode_tok_s  = tp.value("decode", 0.0);
        s.prefill_tok_s = tp.value("prefill", 0.0);
        if (j.contains("scheduler") && j.at("scheduler").is_object()) {
            s.running = j.at("scheduler").value("running", 0);
        }
        if (s.t_ms != 0) { throughput_.push(s); }
    } catch (...) {}
}

// "Who is on the engine now" means recent, so a client that stopped an hour ago drops off
// rather than holding its place in the list.
void Collector::age_client_window_locked() {
    const std::int64_t horizon = now_ms() - kClientWindowMs;
    while (!client_window_.empty() && client_window_.front().t_ms < horizon) {
        client_window_.pop_front();
    }
}

void Collector::reset_request_log_locked() {
    log_insights_      = RequestLogInsights{};
    log_folded_        = 0;
    log_report_folded_ = ~std::uint64_t{0};
    log_history_ready_ = false;
    recent_dones_.clear();
    recent_server_start_.clear();
    older_dones_.clear();
    older_server_start_.clear();
    ++recent_version_;
}

// Routes what the tail reads. History feeds the insights and the request panel's lines older
// than the tail window; the window and appended lines feed the request panel; appended lines
// also feed the live throughput and client figures. log_mu_ and mu_ are taken one at a time.
struct Collector::RequestLogSink {
    Collector& c;
    RequestLogGeneration gen;
    bool live_folded = false;

    void begin(const RequestLogGeneration& g) {
        gen = g;
        std::lock_guard lock(c.log_mu_);
        c.reset_request_log_locked();
    }
    void seed(std::string_view line) {
        note_recent(line);
        // A file that replaced the one the supervisor started with is new as a whole: its
        // window is live data, as it was when the throughput tail started such a file at 0.
        if (gen.number > 1) { fold_live(line); }
    }
    void live(std::string_view line) {
        note_recent(line);
        fold_live(line);
    }
    void history(std::string_view line, std::uint64_t begin) {
        // Parsed before the lock is taken: parsing is nearly all of the cost of catching up.
        const nlohmann::json record =
            line.empty() ? nlohmann::json() : nlohmann::json::parse(line, nullptr, false);
        std::lock_guard lock(c.log_mu_);
        c.log_insights_.note_line_bytes(line.size());
        c.log_insights_.fold(record);
        ++c.log_folded_;
        if (begin < gen.seed_begin &&
            note_recent_line(line, c.older_dones_, c.older_server_start_)) {
            ++c.recent_version_;
        }
    }
    void gone() {
        std::lock_guard lock(c.log_mu_);
        c.reset_request_log_locked();
    }

private:
    void note_recent(std::string_view line) {
        std::lock_guard lock(c.log_mu_);
        if (note_recent_line(line, c.recent_dones_, c.recent_server_start_)) { ++c.recent_version_; }
    }
    void fold_live(std::string_view line) {
        std::lock_guard lock(c.mu_);
        c.fold_live_line_locked(line);
        live_folded = true;
    }
};

bool Collector::poll_request_log_file() {
    std::lock_guard tail_lock(tail_mu_);
    RequestLogSink sink{*this, tail_.generation()};
    std::uint64_t before = 0;
    {
        std::lock_guard lock(log_mu_);
        before = log_poll_.history_offset;
    }
    const RequestLogPoll p = tail_.poll(request_log_path_w_, kRequestLogHistoryBytesPerPoll, sink);
    {
        std::lock_guard lock(log_mu_);
        log_poll_ = p;
        if (p.status == RequestLogStatus::Ok && p.history_at_end && !log_history_ready_) {
            log_history_ready_ = true;
            ++recent_version_;
        }
    }
    if (sink.live_folded) {
        std::lock_guard lock(mu_);
        age_client_window_locked();
    }
    return p.status == RequestLogStatus::Ok && !p.history_at_end && p.history_offset != before;
}

void Collector::request_log_loop() {
    // Catching up on a large log is a few seconds of one core. It is background work and must
    // not take that core from the engine's host threads.
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    while (series_run_.load()) {
        bool more = false;
        try {
            more = poll_request_log_file();
        } catch (...) {}
        if (more) { continue; }
        const auto until = std::chrono::steady_clock::now() + kRequestLogPollPeriod;
        while (series_run_.load() && std::chrono::steady_clock::now() < until) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
}

nlohmann::json Collector::throughput_series_json() {
    std::lock_guard lock(mu_);
    const auto samples = throughput_.samples();
    // The ring holds 900 reports (over an hour at 5 s), which drew as an unreadable wall
    // next to a memory chart that shows ten minutes. Return the same window the memory
    // series covers so the two charts describe the same span; the ring keeps the rest for
    // the request-log panels.
    std::int64_t window_start_ms = 0;
    {
        const auto memory = series_.samples();
        if (!memory.empty()) { window_start_ms = memory.front().t_ms - 5000; }
        else {
            window_start_ms = now_ms() - 600'000;
        }
    }
    nlohmann::json t_ms    = nlohmann::json::array();
    nlohmann::json decode  = nlohmann::json::array();
    nlohmann::json prefill = nlohmann::json::array();
    for (const auto& s : samples) {
        if (s.t_ms < window_start_ms) { continue; }
        t_ms.push_back(s.t_ms);
        decode.push_back(s.decode_tok_s);
        prefill.push_back(s.prefill_tok_s);
    }
    return {{"source", "engine"},
            {"t_ms", std::move(t_ms)},
            {"decode_tok_s", std::move(decode)},
            {"prefill_tok_s", std::move(prefill)}};
}

void Collector::series_loop() {
    // DXGI is an in-process API call, cheap enough to sample at the full rate --
    // and the budget oscillation IS the finding, so it must not be decimated.
    // Device-wide memory (NVML) and the per-process PDH counters move slowly, so they are
    // sampled at 1 Hz and carried forward into the fast series.
    constexpr int kNvidiaEvery = 10;
    int nvidia_tick            = 0;
    NvidiaMemory nvidia_last;
    GpuProcessSource process_source;
    nlohmann::json process_last;
    while (series_run_.load()) {
        const auto t0 = std::chrono::steady_clock::now();
        try {
            VramSample sample;
            sample.t_ms = now_ms();
            DxgiSnapshot dxgi = query_dxgi_local(spec_.device);
            if (nvidia_tick == 0 && dxgi.ok) {
                Collected nv;
                poll_device_memory(nv);
                nvidia_last = nv.nvidia;
                process_last = process_source.sample(dxgi);
            }
            nvidia_tick = (nvidia_tick + 1) % kNvidiaEvery;
            sample.budget_bytes      = dxgi.budget_bytes;
            sample.nvidia_used_bytes = mib_to_bytes(nvidia_last.used_mib);
            {
                std::lock_guard lock(mu_);
                last_dxgi_   = dxgi;
                last_nvidia_ = nvidia_last;
                last_gpu_processes_ = dxgi.ok ? process_last : nlohmann::json{{"ok", false}, {"error", "Graphics card unavailable."}};
                series_.push(sample);
                persist_sample(sample);
            }
        } catch (...) {}
        const auto elapsed = std::chrono::steady_clock::now() - t0;
        const auto period  = std::chrono::milliseconds(100);
        if (elapsed < period) { std::this_thread::sleep_for(period - elapsed); }
    }
}

void Collector::record_transitions(const Collected& snap) {
    const auto t = now_ms();
    std::lock_guard lock(mu_);
    if (last_health_status_ != -1 && last_health_status_ != snap.health_status) {
        if (snap.health_status == 200) {
            const VramSeriesEvent ev{t, "engine_up", "health 200"};
            series_.push_event(ev);
            persist_event(ev);
        } else if (last_health_status_ == 200) {
            const VramSeriesEvent ev{t, "engine_down",
                                     "health " + std::to_string(snap.health_status)};
            series_.push_event(ev);
            persist_event(ev);
        }
    }
    last_health_status_ = snap.health_status;
    last_health_body_   = snap.health_body;
    last_admin_vram_    = snap.admin_vram;
    last_admin_note_    = snap.admin_vram_note;
    if (snap.admin_vram.is_object()) {
        const std::string trans  = snap.admin_vram.value("last_transition", "");
        const std::string reason = snap.admin_vram.value("last_reason", "");
        std::string kind;
        if (admin_cursor_.observe(trans, reason, kind)) {
            std::string released;
            if (snap.admin_vram.contains("tiers") && snap.admin_vram.at("tiers").is_array()) {
                for (const auto& tier : snap.admin_vram.at("tiers")) {
                    if (tier.value("released", false)) {
                        if (!released.empty()) { released += ","; }
                        released += tier.value("name", "?");
                    }
                }
            }
            std::string label = std::string(trans);
            if (!reason.empty()) {
                if (!label.empty()) { label += " "; }
                label += reason;
            }
            if (!released.empty()) { label += " released=" + released; }
            const VramSeriesEvent ev{t, kind, label};
            series_.push_event(ev);
            persist_event(ev);
            if (kind == "vram_release") { last_release_ms_ = t; }
            if (kind == "vram_reclaim") { last_release_ms_ = 0; }
        }
        bool any_released = false;
        if (snap.admin_vram.contains("tiers") && snap.admin_vram.at("tiers").is_array()) {
            for (const auto& tier : snap.admin_vram.at("tiers")) {
                if (tier.value("released", false)) { any_released = true; }
            }
        }
        if (!any_released && trans == "reclaim") { last_release_ms_ = 0; }
    }
}

void Collector::note_engine_state(const std::string& state, const std::string& last_event) {
    std::lock_guard lock(mu_);
    if (!last_engine_state_.empty() && state != last_engine_state_) {
        const auto t = now_ms();
        if (state == "Running" || state == "Starting") {
            const VramSeriesEvent ev{t, "engine_start", last_event.empty() ? state : last_event};
            series_.push_event(ev);
            persist_event(ev);
        } else if (state == "Stopped" || state == "Stopping" || state == "Halted") {
            const VramSeriesEvent ev{t, "engine_stop", last_event.empty() ? state : last_event};
            series_.push_event(ev);
            persist_event(ev);
        }
    }
    last_engine_state_ = state;
}

nlohmann::json Collector::series_json() {
    std::lock_guard lock(mu_);
    const auto samples = series_.samples();
    nlohmann::json t_ms            = nlohmann::json::array();
    nlohmann::json budget          = nlohmann::json::array();
    nlohmann::json nvidia_used     = nlohmann::json::array();
    for (const auto& s : samples) {
        t_ms.push_back(s.t_ms);
        budget.push_back(s.budget_bytes);
        nvidia_used.push_back(s.nvidia_used_bytes);
    }
    nlohmann::json events = nlohmann::json::array();
    for (const auto& e : series_.events()) {
        events.push_back({{"t_ms", e.t_ms}, {"kind", e.kind}, {"label", e.label}});
    }
    return {{"hz", 10},
            {"raw", true},
            {"t_ms", std::move(t_ms)},
            {"budget_bytes", std::move(budget)},
            {"nvidia_used_bytes", std::move(nvidia_used)},
            {"events", std::move(events)},
            {"detector_last_ran_ms", detector_last_ran_ms_}};
}

nlohmann::json Collector::vram_control_json() {
    std::lock_guard lock(mu_);
    nlohmann::json tiers = nlohmann::json::array();
    bool any_released    = false;
    if (last_admin_vram_.is_object() && last_admin_vram_.contains("tiers") &&
        last_admin_vram_.at("tiers").is_array()) {
        for (const auto& tier : last_admin_vram_.at("tiers")) {
            const bool released = tier.value("released", false);
            if (released) { any_released = true; }
            tiers.push_back({{"name", tier.value("name", "")},
                             {"released", released},
                             {"held_bytes", tier.value("held_bytes", 0)},
                             {"min_bytes", tier.value("min_bytes", 0)},
                             {"max_bytes", tier.value("max_bytes", 0)},
                             {"reclaimable_bytes", tier.value("reclaimable_bytes", 0)}});
        }
    }
    const auto now = now_ms();
    nlohmann::json out = {
        {"last_transition", admin_cursor_.last_transition},
        {"last_reason", admin_cursor_.last_reason},
        {"any_released", any_released},
        {"since_release_s",
         (any_released && last_release_ms_ > 0)
             ? nlohmann::json((now - last_release_ms_) / 1000)
             : nlohmann::json(nullptr)},
        {"tiers", std::move(tiers)},
        {"note", last_admin_note_},
        {"detector_last_ran_ms", detector_last_ran_ms_},
        {"detector_age_s",
         detector_last_ran_ms_ > 0 ? nlohmann::json((now - detector_last_ran_ms_) / 1000)
                                   : nlohmann::json(nullptr)},
    };
    return out;
}

nlohmann::json Collector::request_log_report() {
    if (spec_.request_log.empty()) { return request_log_unconfigured_report(); }
    std::lock_guard lock(log_mu_);
    switch (log_poll_.status) {
    case RequestLogStatus::NotPolled:
        return request_log_reading_report(spec_.request_log, 0, 0);
    case RequestLogStatus::Missing:
        return request_log_missing_report(spec_.request_log);
    case RequestLogStatus::Unreadable:
        // Locked for the moment. What was already read still holds; with nothing read yet
        // there is nothing to report.
        if (!log_history_ready_) {
            nlohmann::json report;
            report["source"] = {{"request_log", "unreadable"},
                                {"path", spec_.request_log},
                                {"error", log_poll_.error}};
            report["insights"] = nlohmann::json::array({insight_unavailable(
                "source.request_log", "Request log cannot be read",
                "the request log exists but could not be opened (Windows error " +
                    std::to_string(log_poll_.error) + "); no request_done records in window",
                {{"path", spec_.request_log}, {"error", log_poll_.error}})});
            return report;
        }
        break;
    case RequestLogStatus::Ok:
        if (!log_history_ready_) {
            return request_log_reading_report(spec_.request_log, log_poll_.history_offset,
                                              log_poll_.size);
        }
        break;
    }
    // Rebuilt only when records were folded since: /api/events asks every second.
    if (log_report_folded_ != log_folded_) {
        log_report_cache_  = log_insights_.report(spec_.request_log);
        log_report_folded_ = log_folded_;
    }
    return log_report_cache_;
}

nlohmann::json Collector::insights_report() {
    auto report = request_log_report();
    nlohmann::json admin;
    std::string note;
    {
        std::lock_guard lock(mu_);
        admin = last_admin_vram_;
        note  = last_admin_note_;
    }
    append_admin_vram_insights(report, admin, note);
    return report;
}

Collected Collector::snapshot() {
    Collected out;
    poll_request_log(out);
    if (series_run_.load()) {
        std::lock_guard lock(mu_);
        out.health_status    = last_health_status_ < 0 ? 0 : last_health_status_;
        out.health_body      = last_health_body_;
        out.admin_vram       = last_admin_vram_;
        out.admin_vram_note  = last_admin_note_;
        out.dxgi             = last_dxgi_;
        out.nvidia           = last_nvidia_;
        out.gpu_processes    = last_gpu_processes_;
        return out;
    }
    poll_health(out);
    poll_admin(out);
    out.dxgi = query_dxgi_local(spec_.device);
    poll_device_memory(out);
    record_transitions(out);
    return out;
}

} // namespace ninfer::supervisor
