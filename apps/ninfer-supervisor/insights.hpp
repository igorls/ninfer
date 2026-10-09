#pragma once

#include "logic.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::supervisor {

inline nlohmann::json insight_unavailable(std::string id, std::string title, std::string statement,
                                          nlohmann::json evidence      = nlohmann::json::object(),
                                          nlohmann::json measured_over = {{"requests", 0}}) {
    return {{"id", std::move(id)},
            {"severity", "notice"},
            {"title", std::move(title)},
            {"statement", std::move(statement)},
            {"evidence", std::move(evidence)},
            {"confidence", "measured"},
            {"measured_over", std::move(measured_over)},
            {"availability", "unavailable"}};
}

inline nlohmann::json insight_available(std::string id, std::string severity, std::string title,
                                        std::string statement, nlohmann::json evidence,
                                        std::string recommendation, std::string confidence,
                                        nlohmann::json measured_over) {
    nlohmann::json out = {{"id", std::move(id)},
                          {"severity", std::move(severity)},
                          {"title", std::move(title)},
                          {"statement", std::move(statement)},
                          {"evidence", std::move(evidence)},
                          {"confidence", std::move(confidence)},
                          {"measured_over", std::move(measured_over)},
                          {"availability", "available"}};
    if (!recommendation.empty()) { out["recommendation"] = std::move(recommendation); }
    return out;
}

inline std::int64_t json_i64(const nlohmann::json& j, const char* key, std::int64_t fallback = 0) {
    if (!j.contains(key)) { return fallback; }
    const auto& v = j.at(key);
    if (v.is_number_integer()) { return v.get<std::int64_t>(); }
    if (v.is_number()) { return static_cast<std::int64_t>(v.get<double>()); }
    return fallback;
}

inline double json_f64(const nlohmann::json& j, const char* key, double fallback = 0) {
    if (!j.contains(key)) { return fallback; }
    const auto& v = j.at(key);
    if (v.is_number()) { return v.get<double>(); }
    return fallback;
}

inline bool has_valid_optional_numbers(const nlohmann::json& object,
                                       std::initializer_list<const char*> keys) {
    if (!object.is_object()) { return false; }
    for (const char* key : keys) {
        if (object.contains(key) && !object.at(key).is_number()) { return false; }
    }
    return true;
}

inline bool request_done_numeric_fields_valid(const nlohmann::json& done) {
    if (done.contains("timestamp_unix_ms") && !done.at("timestamp_unix_ms").is_number()) {
        return false;
    }
    const auto valid_section = [&](const char* name, std::initializer_list<const char*> keys) {
        return !done.contains(name) || has_valid_optional_numbers(done.at(name), keys);
    };
    if (!valid_section("request", {"request_id", "tool_count", "requested_output_tokens",
                                   "message_count", "media_item_count"}) ||
        !valid_section("result", {"completion_tokens", "prompt_tokens", "prefix_cache_hit_tokens",
                                  "computed_prefill_tokens"}) ||
        !valid_section("timings_seconds",
                       {"total", "prepare", "prefill", "decode", "vision", "ttft"})) {
        return false;
    }
    if (!done.contains("speculative") || !done.at("speculative").is_object()) { return true; }
    const auto& speculative = done.at("speculative");
    if (!has_valid_optional_numbers(speculative, {"draft_window", "drafted_tokens",
                                                  "accepted_tokens", "fallback_steps", "rounds"})) {
        return false;
    }
    if (speculative.contains("accepted_per_position")) {
        const auto& positions = speculative.at("accepted_per_position");
        if (!positions.is_array()) { return false; }
        for (const auto& position : positions) {
            if (!position.is_number()) { return false; }
        }
    }
    return true;
}

// Insights over the engine request log (schema_version 10 and later), folded one record at a time
// so the supervisor can follow a log that grows to gigabytes: each record is read once, when it is
// appended, and the report is built from running aggregates instead of a re-read of the file.
//
// The report is the one a whole-file pass produces. Two things need care to keep it so:
//
// - Most sections describe the latest server instance only. Records arrive in file order and the
//   engine writes server_start before anything else of its instance, so the instance-scoped
//   aggregates restart when a newer server_start arrives and everything already folded becomes
//   "another instance".
// - Some figures need more than a running sum. Per-request history is kept only where the report
//   depends on it, and only for the latest server instance: decode and fallback step counts of
//   each speculative request (the first/second half split moves as requests arrive; 16 bytes per
//   request) and a histogram of prompt sizes (median and p90; bounded by max_context distinct
//   values). Everything else is a counter, a maximum, the first or newest few samples, or the
//   state of a trailing run.
//
// Content/reasoning_content are not invented: the engine request log does not write them.
class RequestLogInsights {
public:
    static constexpr std::size_t kSamples = 8;

    // One JSONL line without its terminator. Lines that do not parse are skipped, as the batch
    // reader did.
    void fold_line(std::string_view line) {
        bytes_ += line.size() + 1;
        if (line.empty()) { return; }
        fold(nlohmann::json::parse(line, nullptr, false));
    }

    // One parsed record, in file order. `record` may be a discarded parse result.
    void fold(const nlohmann::json& record) {
        if (!record.is_object()) { return; }
        try {
            fold_record(record);
        } catch (...) {
            // A record with a field of the wrong type is skipped. A whole-file pass failed the
            // entire report on it.
        }
    }

    // Counts a line folded through fold() rather than fold_line(), for the empty-log check.
    void note_line_bytes(std::size_t line_bytes) { bytes_ += line_bytes + 1; }

    [[nodiscard]] nlohmann::json report(std::string_view path) const;

private:
    struct Largest {
        std::int64_t id         = 0;
        std::int64_t prompt     = 0;
        std::int64_t completion = 0;
        std::int64_t total      = 0;
        std::string finish;
    };

    // Speculative decoding, as the engine writes it per request_done: backend, draft_window,
    // rounds, drafted/accepted tokens, fallback_steps and accepted_per_position, where position
    // p counts the draft rounds in which the p-th drafted token was accepted. Flash-Next counts a
    // fallback step as a round too and the 27B path does not, and a round near the output limit
    // drafts fewer than draft_window tokens, so neither "rounds" nor drafted/draft_window is the
    // number of rounds that drafted. max(rounds - fallback, ceil(drafted / window)) is exact on
    // Flash-Next and a lower bound on the 27B path; per-position rates are clamped to 1.
    // The backend and window are launch configuration, so only the latest server instance is
    // measured: a whole-file sum would blend windows 4, 5 and 7 from different launches into one
    // curve that belongs to none of them.
    struct SpecPosition {
        std::uint64_t accepted = 0;
        std::uint64_t rounds   = 0;
    };

    struct SpecSteps {
        std::uint64_t steps    = 0; // draft rounds plus fallback steps
        std::uint64_t fallback = 0;
    };

    struct Spec {
        std::string backend;
        int draft_window              = 0;
        int requests_with_telemetry   = 0;
        int requests_with_drafts      = 0;
        int requests_backend_none     = 0;
        std::uint64_t rounds_reported = 0;
        std::uint64_t draft_rounds    = 0;
        std::uint64_t drafted         = 0;
        std::uint64_t accepted        = 0;
        std::uint64_t fallback        = 0;
        // One entry per request_done with a speculative object, in order. The fallback-share halves
        // split them by position against the instance's request_done count, which keeps growing.
        std::vector<SpecSteps> steps;
        std::vector<SpecPosition> positions;
        std::vector<std::int64_t> sample_ids;
        std::vector<nlohmann::json> low_acceptance_samples;
        std::vector<nlohmann::json> high_fallback_samples;
        std::map<int, int> windows_seen;
        std::int64_t tmin = 0;
        std::int64_t tmax = 0;
    };

    // Reuse collapse: a Host context arena that stays full while every later multi-turn request
    // falls to root with no cache hit. A whole-file pass walks both series backwards from the end;
    // this keeps the trailing run of saturated throughput samples and the multi-turn root misses
    // that count toward the streak. The streak only counts requests stamped at or after the run's
    // first sample, which needs the engine's timestamps to rise in file order; it stamps a record
    // just before appending it, so concurrent requests can invert by a few milliseconds, which only
    // matters for a request that lands on the run's first sample.
    struct Miss {
        std::int64_t ts      = 0;
        std::uint64_t refill = 0;
        std::int64_t id      = 0;
    };

    struct Collapse {
        std::uint64_t capacity  = 0; // host_capacity_bytes of the instance; 0 when not reported
        std::size_t run_samples = 0; // trailing saturated throughput samples
        std::int64_t run_since  = 0;
        std::int64_t run_until  = 0;
        // While a run is active: misses stamped at or after run_since since the last cache hit.
        std::size_t streak          = 0;
        std::uint64_t streak_refill = 0;
        std::deque<std::int64_t> streak_ids; // newest last, at most kSamples
        // While no run is active: misses since the last cache hit, not older than the latest
        // throughput sample, in case the next sample starts a run they belong to.
        std::deque<Miss> waiting;
    };

    // Everything that describes the latest server instance. Replaced when a newer one starts.
    struct Scope {
        std::uint64_t dones      = 0;
        std::int64_t max_context = 0;
        Spec spec;
        std::map<std::int64_t, std::uint64_t> prompts; // prompt tokens -> request_done count
        std::vector<Largest> largest;                  // at most kSamples, largest total first
        int near_limit                = 0;
        std::size_t occupancy_samples = 0;
        std::int64_t kv_now           = 0;
        std::int64_t kv_high          = 0;
        std::int64_t peak_running     = 0;
        std::int64_t peak_waiting     = 0;
        std::int64_t preemptions      = 0;
        std::int64_t spill_pages      = 0;
        int admission_expired         = 0;
        std::vector<std::int64_t> admission_expired_ids;
        Collapse collapse;
    };

    struct Bucket {
        int queued            = 0;
        int prefill           = 0;
        int decode            = 0;
        int mixed             = 0;
        int unpaired          = 0;
        double queue_wait_sum = 0;
        double prepare_sum    = 0;
        double prefill_sum    = 0;
        double decode_sum     = 0;
        double vision_sum     = 0;
        double ttft_sum       = 0;
        double total_sum      = 0;
        std::vector<std::int64_t> queued_ids;
        std::vector<std::int64_t> prefill_ids;
        std::vector<std::int64_t> decode_ids;
    };

    // A request_start whose request has not finished yet.
    static constexpr std::size_t kMaxOpenStarts = 65536;

    void fold_record(const nlohmann::json& j);
    void fold_server_start(const nlohmann::json& j, std::int64_t ts);
    void fold_done(const nlohmann::json& done, std::int64_t ts);
    void fold_throughput(const nlohmann::json& tp, std::int64_t ts);
    void fold_error(const nlohmann::json& err);

    [[nodiscard]] bool in_scope(const nlohmann::json& j) const {
        return latest_instance_.empty() || j.value("server_instance_id", "") == latest_instance_;
    }

    [[nodiscard]] bool is_latest_instance(const nlohmann::json& j) const {
        return !latest_instance_.empty() && j.value("server_instance_id", "") == latest_instance_;
    }

    static std::string start_key(const nlohmann::json& j, std::int64_t request_id) {
        return j.value("server_instance_id", "") + ":" + std::to_string(request_id);
    }

    std::uint64_t bytes_ = 0;
    int parsed_          = 0;
    std::int64_t tmin_   = 0;
    std::int64_t tmax_   = 0;

    std::string latest_instance_;
    std::int64_t latest_start_    = 0;
    nlohmann::json latest_engine_ = nlohmann::json::object();
    Scope scope_;

    std::uint64_t starts_ = 0;
    std::unordered_map<std::string, std::int64_t> open_starts_; // instance:request_id -> start ms
    std::uint64_t dones_       = 0;
    std::uint64_t throughputs_ = 0;
    int max_waiting_           = 0;
    int max_running_           = 0;
    int max_prefill_           = 0;

    Bucket b_;
    int output_limit_thinking_ = 0;
    int output_limit_hit_cap_  = 0;
    int thinking_requests_     = 0;
    int tools_declared_        = 0;
    std::vector<std::int64_t> output_limit_ids_; // the first kSamples
    std::int64_t output_limit_cap_sum_ = 0;
    int reuse_root_single_             = 0;
    int reuse_root_multi_              = 0;
    int reuse_checkpoint_              = 0;
    int reuse_other_                   = 0;
    std::uint64_t multi_prompt_tokens_ = 0;
    std::uint64_t multi_hit_tokens_    = 0;
    std::vector<nlohmann::json> reset_multi_samples_;
};

inline void RequestLogInsights::fold_record(const nlohmann::json& j) {
    const auto ev = j.find("event");
    if (ev == j.end()) { return; }
    const std::string event = ev->get<std::string>();
    if (event.empty()) { return; }
    if (event == "request_done" && !request_done_numeric_fields_valid(j)) { return; }
    ++parsed_;
    const auto ts = json_i64(j, "timestamp_unix_ms");
    if (tmin_ == 0 || ts < tmin_) { tmin_ = ts; }
    if (ts > tmax_) { tmax_ = ts; }
    if (event == "server_start" && j.contains("engine")) {
        fold_server_start(j, ts);
    } else if (event == "request_start" && j.contains("request")) {
        const auto rid = json_i64(j.at("request"), "request_id");
        ++starts_;
        open_starts_[start_key(j, rid)] = ts;
        // A start whose request never finished (an engine that died mid-request) is never
        // claimed. Drop those of earlier instances first; they can no longer finish.
        if (open_starts_.size() > kMaxOpenStarts) {
            const std::string prefix = latest_instance_ + ":";
            std::erase_if(open_starts_,
                          [&](const auto& kv) { return kv.first.rfind(prefix, 0) != 0; });
            if (open_starts_.size() > kMaxOpenStarts) { open_starts_.clear(); }
        }
    } else if (event == "request_done") {
        fold_done(j, ts);
    } else if (event == "throughput") {
        fold_throughput(j, ts);
    } else if (event == "request_error") {
        fold_error(j);
    }
}

inline void RequestLogInsights::fold_server_start(const nlohmann::json& j, std::int64_t ts) {
    const std::string instance = j.value("server_instance_id", "");
    const auto& engine         = j.at("engine");
    if (!instance.empty() && ts >= latest_start_) {
        if (instance != latest_instance_) {
            latest_instance_ = instance;
            scope_           = Scope{};
        }
        latest_start_      = ts;
        latest_engine_     = engine;
        scope_.max_context = json_i64(latest_engine_, "max_context");
    }
    if (!instance.empty() && instance == latest_instance_ && engine.contains("context_cache")) {
        const auto capacity = json_i64(engine.at("context_cache"), "host_capacity_bytes");
        if (capacity > 0) { scope_.collapse.capacity = static_cast<std::uint64_t>(capacity); }
    }
}

inline void RequestLogInsights::fold_done(const nlohmann::json& done, std::int64_t ts) {
    // value() throws on a field of the wrong type, so every such field is read before anything
    // changes and a malformed record leaves no partial trace.
    const auto& req     = done.contains("request") ? done.at("request") : nlohmann::json::object();
    const auto id       = json_i64(req, "request_id");
    const bool thinking = req.value("enable_thinking", false);
    const auto tool_count = json_i64(req, "tool_count");
    const auto& result    = done.contains("result") ? done.at("result") : nlohmann::json::object();
    const std::string finish = result.value("finish_reason", "");
    const int cap            = static_cast<int>(json_i64(req, "requested_output_tokens"));
    const int completion     = static_cast<int>(json_i64(result, "completion_tokens"));
    const int messages       = static_cast<int>(json_i64(req, "message_count"));
    const std::string reuse  = result.value("prefix_reuse_path", "");
    const auto prompt_tokens = static_cast<std::uint64_t>(json_i64(result, "prompt_tokens"));
    const auto hit_tokens = static_cast<std::uint64_t>(json_i64(result, "prefix_cache_hit_tokens"));
    const bool multiturn  = messages >= 2;
    const std::string join = start_key(done, id);
    const bool scoped      = in_scope(done);
    const bool latest      = is_latest_instance(done);
    const auto& timings =
        done.contains("timings_seconds") ? done.at("timings_seconds") : nlohmann::json::object();
    const double total   = json_f64(timings, "total");
    const double prepare = json_f64(timings, "prepare");
    const double prefill = json_f64(timings, "prefill");
    const double decode  = json_f64(timings, "decode");
    const double vision  = json_f64(timings, "vision");
    const double ttft    = json_f64(timings, "ttft");
    const nlohmann::json* sp =
        (scoped && done.contains("speculative") && done.at("speculative").is_object())
            ? &done.at("speculative")
            : nullptr;
    std::string sp_backend;
    int sp_window = 0;
    if (sp != nullptr) {
        sp_backend = sp->value("backend", "");
        sp_window  = static_cast<int>(json_i64(*sp, "draft_window"));
    }

    ++dones_;
    if (thinking) { ++thinking_requests_; }
    if (tool_count > 0) { ++tools_declared_; }
    if (finish == "output_limit" && thinking) {
        ++output_limit_thinking_;
        if (output_limit_ids_.size() < kSamples) { output_limit_ids_.push_back(id); }
        output_limit_cap_sum_ += cap;
        if (cap > 0 && completion >= cap) { ++output_limit_hit_cap_; }
    }
    if (multiturn) {
        multi_prompt_tokens_ += prompt_tokens;
        multi_hit_tokens_ += hit_tokens;
    }
    if (reuse == "root") {
        if (multiturn) {
            ++reuse_root_multi_;
            if (reset_multi_samples_.size() < kSamples) {
                reset_multi_samples_.push_back({{"request_id", id},
                                                {"message_count", messages},
                                                {"prompt_tokens", prompt_tokens},
                                                {"prefix_cache_hit_tokens", hit_tokens}});
            }
        } else {
            ++reuse_root_single_;
        }
    } else if (reuse == "checkpoint") {
        ++reuse_checkpoint_;
    } else if (!reuse.empty()) {
        ++reuse_other_;
    }

    if (scoped) {
        auto& s = scope_;
        ++s.dones;
        if (sp != nullptr) {
            auto& spec = s.spec;
            if (spec.tmin == 0 || ts < spec.tmin) { spec.tmin = ts; }
            if (ts > spec.tmax) { spec.tmax = ts; }
            SpecSteps half;
            if (sp_backend.empty() || sp_backend == "none" || sp_window <= 0) {
                ++spec.requests_backend_none;
            } else {
                ++spec.requests_with_telemetry;
                spec.backend      = sp_backend;
                spec.draft_window = std::max(spec.draft_window, sp_window);
                ++spec.windows_seen[sp_window];
                auto count = [&](const char* key) {
                    return static_cast<std::uint64_t>(
                        std::max<std::int64_t>(0, json_i64(*sp, key)));
                };
                const auto drafted      = count("drafted_tokens");
                const auto accepted     = count("accepted_tokens");
                const auto fallback     = count("fallback_steps");
                const auto rounds       = count("rounds");
                const auto full_windows = (drafted + static_cast<std::uint64_t>(sp_window) - 1) /
                                          static_cast<std::uint64_t>(sp_window);
                const auto draft_rounds = std::max(
                    rounds > fallback ? rounds - fallback : std::uint64_t{0}, full_windows);
                spec.rounds_reported += rounds;
                spec.draft_rounds += draft_rounds;
                spec.drafted += drafted;
                spec.accepted += accepted;
                spec.fallback += fallback;
                half = {draft_rounds + fallback, fallback};
                if (sp->contains("accepted_per_position") &&
                    sp->at("accepted_per_position").is_array()) {
                    const auto& pos = sp->at("accepted_per_position");
                    if (spec.positions.size() < pos.size()) { spec.positions.resize(pos.size()); }
                    for (std::size_t p = 0; p < pos.size(); ++p) {
                        if (!pos.at(p).is_number()) { continue; }
                        spec.positions[p].accepted += pos.at(p).get<std::uint64_t>();
                        if (static_cast<int>(p) < sp_window) {
                            spec.positions[p].rounds += draft_rounds;
                        }
                    }
                }
                if (fallback >= 8 && fallback * 2 >= draft_rounds + fallback &&
                    spec.high_fallback_samples.size() < kSamples) {
                    spec.high_fallback_samples.push_back(
                        {{"request_id", id},
                         {"fallback_steps", fallback},
                         {"draft_rounds", draft_rounds},
                         {"completion_tokens", completion},
                         {"message_count", messages},
                         {"media_item_count", json_i64(req, "media_item_count")},
                         {"enable_thinking", thinking}});
                }
                if (drafted > 0) {
                    ++spec.requests_with_drafts;
                    if (spec.sample_ids.size() < kSamples) { spec.sample_ids.push_back(id); }
                    const double ratio =
                        static_cast<double>(accepted) / static_cast<double>(drafted);
                    if (draft_rounds >= 8 && ratio < 0.25 &&
                        spec.low_acceptance_samples.size() < kSamples) {
                        spec.low_acceptance_samples.push_back({{"request_id", id},
                                                               {"drafted_tokens", drafted},
                                                               {"accepted_tokens", accepted},
                                                               {"fallback_steps", fallback},
                                                               {"completion_tokens", completion}});
                    }
                }
            }
            spec.steps.push_back(half);
        }

        Largest l;
        l.id         = id;
        l.prompt     = std::max<std::int64_t>(0, json_i64(result, "prompt_tokens"));
        l.completion = std::max<std::int64_t>(0, json_i64(result, "completion_tokens"));
        l.total      = l.prompt + l.completion;
        l.finish     = finish;
        ++s.prompts[l.prompt];
        if (s.max_context > 0 &&
            static_cast<double>(l.total) / static_cast<double>(s.max_context) >= 0.90) {
            ++s.near_limit;
        }
        if (s.largest.size() < kSamples || l.total > s.largest.back().total) {
            const auto at = std::upper_bound(
                s.largest.begin(), s.largest.end(), l.total,
                [](std::int64_t total, const Largest& e) { return total > e.total; });
            s.largest.insert(at, std::move(l));
            if (s.largest.size() > kSamples) { s.largest.pop_back(); }
        }
    }

    // Reuse collapse: multi-turn requests of the latest instance only.
    if (latest && json_i64(req, "message_count") >= 2) {
        auto& c         = scope_.collapse;
        const bool miss = reuse == "root" && json_i64(result, "prefix_cache_hit_tokens", -1) == 0;
        if (c.run_samples > 0 && ts < c.run_since) {
            // Stamped before the run began: outside the streak whether it hit or missed.
        } else if (!miss) {
            c.streak        = 0;
            c.streak_refill = 0;
            c.streak_ids.clear();
            c.waiting.clear();
        } else {
            const auto refill = static_cast<std::uint64_t>(
                std::max<std::int64_t>(0, json_i64(result, "computed_prefill_tokens")));
            if (c.run_samples > 0) {
                ++c.streak;
                c.streak_refill += refill;
                c.streak_ids.push_back(id);
                if (c.streak_ids.size() > kSamples) { c.streak_ids.pop_front(); }
            } else {
                c.waiting.push_back({ts, refill, id});
                // Throughput samples prune this; cap it for a log that stops writing them.
                if (c.waiting.size() > 4096) { c.waiting.pop_front(); }
            }
        }
    }

    b_.prepare_sum += prepare;
    b_.prefill_sum += prefill;
    b_.decode_sum += decode;
    b_.vision_sum += vision;
    b_.ttft_sum += ttft;
    b_.total_sum += total;

    const auto it = open_starts_.find(join);
    if (it == open_starts_.end()) {
        ++b_.unpaired;
        return;
    }
    const double wall_s = static_cast<double>(ts - it->second) / 1000.0;
    open_starts_.erase(it);
    double queue_wait = wall_s - total;
    if (queue_wait < 0.0) { queue_wait = 0.0; }
    b_.queue_wait_sum += queue_wait;
    const bool queued = queue_wait >= 0.020 && wall_s > 0.0 && queue_wait >= 0.25 * wall_s;
    if (queued) {
        ++b_.queued;
        if (b_.queued_ids.size() < kSamples) { b_.queued_ids.push_back(id); }
    } else if (total > 0.0 && prefill >= decode && prefill >= 0.4 * total) {
        ++b_.prefill;
        if (b_.prefill_ids.size() < kSamples) { b_.prefill_ids.push_back(id); }
    } else if (total > 0.0 && decode >= 0.4 * total) {
        ++b_.decode;
        if (b_.decode_ids.size() < kSamples) { b_.decode_ids.push_back(id); }
    } else {
        ++b_.mixed;
    }
}

inline void RequestLogInsights::fold_throughput(const nlohmann::json& tp, std::int64_t ts) {
    const bool scoped = in_scope(tp);
    const bool latest = is_latest_instance(tp);
    ++throughputs_;
    if (tp.contains("scheduler")) {
        const auto& sch = tp.at("scheduler");
        max_waiting_    = std::max(max_waiting_, static_cast<int>(json_i64(sch, "waiting")));
        max_running_    = std::max(max_running_, static_cast<int>(json_i64(sch, "running")));
        max_prefill_    = std::max(max_prefill_, static_cast<int>(json_i64(sch, "prefilling")));
    }
    if (!scoped) { return; }
    auto& s = scope_;
    if (tp.contains("scheduler") && tp.at("scheduler").is_object()) {
        const auto& sch = tp.at("scheduler");
        s.peak_running  = std::max(s.peak_running, json_i64(sch, "running"));
        s.peak_waiting  = std::max(s.peak_waiting, json_i64(sch, "waiting"));
    }
    if (tp.contains("scheduling") && tp.at("scheduling").is_object()) {
        s.preemptions += json_i64(tp.at("scheduling"), "preemptions");
    }
    const bool has_cache = tp.contains("context_cache") && tp.at("context_cache").is_object();
    const auto& cache    = has_cache ? tp.at("context_cache") : nlohmann::json::object();
    if (has_cache) {
        if (cache.contains("occupancy") && cache.at("occupancy").is_object() &&
            cache.at("occupancy").contains("device_main_kv_pages")) {
            const auto pages = json_i64(cache.at("occupancy"), "device_main_kv_pages", -1);
            if (pages >= 0) {
                ++s.occupancy_samples;
                s.kv_now  = pages;
                s.kv_high = std::max(s.kv_high, pages);
            }
        }
        if (cache.contains("pressure") && cache.at("pressure").is_object()) {
            // Throughput records carry per-interval deltas, so sums are window totals.
            const auto& pr = cache.at("pressure");
            s.spill_pages += json_i64(pr, "spill_pages");
        }
    }

    auto& c = s.collapse;
    if (!latest || c.capacity == 0) { return; }
    const auto& occupancy =
        cache.contains("occupancy") ? cache.at("occupancy") : nlohmann::json::object();
    const auto occupied = json_i64(occupancy, "host_context_occupied_bytes", -1);
    // Occupied already includes in-flight reserved destinations.
    if (occupied == static_cast<std::int64_t>(c.capacity)) {
        if (c.run_samples == 0) {
            // A run starts: the misses waiting since the last hit join the streak if they were
            // stamped at or after this sample.
            c.run_since     = ts;
            c.run_until     = ts;
            c.streak        = 0;
            c.streak_refill = 0;
            c.streak_ids.clear();
            for (const Miss& m : c.waiting) {
                if (m.ts < ts) { continue; }
                ++c.streak;
                c.streak_refill += m.refill;
                c.streak_ids.push_back(m.id);
                if (c.streak_ids.size() > kSamples) { c.streak_ids.pop_front(); }
            }
            c.waiting.clear();
        }
        // The newest sample with a timestamp ends the run.
        if (ts != 0) { c.run_until = ts; }
        ++c.run_samples;
    } else {
        c.run_samples   = 0;
        c.run_since     = 0;
        c.run_until     = 0;
        c.streak        = 0;
        c.streak_refill = 0;
        c.streak_ids.clear();
        // A later run starts at a later sample, so misses stamped before this one cannot join it.
        while (!c.waiting.empty() && c.waiting.front().ts < ts) { c.waiting.pop_front(); }
    }
}

inline void RequestLogInsights::fold_error(const nlohmann::json& err) {
    // An errored request is finished: its start can no longer be paired with a request_done.
    if (err.contains("request")) {
        open_starts_.erase(start_key(err, json_i64(err.at("request"), "request_id")));
    }
    if (!in_scope(err)) { return; }
    const auto& e = err.contains("error") ? err.at("error") : nlohmann::json::object();
    if (e.value("message", "").find("waiting for admission") == std::string::npos) { return; }
    ++scope_.admission_expired;
    if (scope_.admission_expired_ids.size() < kSamples) {
        const auto& req = err.contains("request") ? err.at("request") : nlohmann::json::object();
        scope_.admission_expired_ids.push_back(json_i64(req, "request_id"));
    }
}

inline nlohmann::json RequestLogInsights::report(std::string_view path) const {
    nlohmann::json report = {
        {"source", {{"request_log", bytes_ == 0 ? "empty" : "ok"}, {"path", std::string(path)}}},
        {"insights", nlohmann::json::array()},
    };
    auto& insights        = report["insights"];
    const double window_s = (tmax_ > tmin_) ? static_cast<double>(tmax_ - tmin_) / 1000.0 : 0.0;

    if (parsed_ == 0) {
        insights.push_back(
            insight_unavailable("source.request_log", "Request log has no usable records",
                                "no request_done records in window",
                                {{"path", std::string(path)}, {"parsed_events", 0}}));
        report["generated_note"] = "unavailable is not a clean zero";
        return report;
    }

    if (dones_ == 0) {
        insights.push_back(
            insight_unavailable("source.request_done", "No completed requests in window",
                                "no request_done records in window",
                                {{"parsed_events", parsed_}, {"request_start", starts_}},
                                {{"requests", 0}, {"parsed_events", parsed_}}));
        return report;
    }

    const Scope& s                     = scope_;
    const Spec& spec                   = s.spec;
    const auto spec_scope_dones        = s.dones;
    const std::uint64_t other_instance = latest_instance_.empty() ? 0 : dones_ - s.dones;
    const int paired                   = static_cast<int>(dones_) - b_.unpaired;

    const auto over = nlohmann::json{{"requests", dones_},
                                     {"paired_requests", paired},
                                     {"unpaired_done", b_.unpaired},
                                     {"window_s", window_s},
                                     {"throughput_events", throughputs_}};

    // A full Host context arena is not itself a fault: pressure can legitimately fill it briefly.
    // The production failure in #15 is the conjunction of a pool that remains full and later
    // multi-turn requests that all fall to Root with no cache hit. Keep the two signals joined so
    // ordinary first-turn Root selections never become a false alarm.
    if (const auto& c = s.collapse; c.capacity > 0 && c.run_samples >= 3 &&
                                    c.run_until - c.run_since >= 60'000 && c.streak >= 3) {
        std::vector<std::int64_t> sample_ids(c.streak_ids.rbegin(), c.streak_ids.rend());
        std::ostringstream statement;
        statement << "Host context occupancy remained at " << c.capacity << "/" << c.capacity
                  << " for " << c.run_samples << " consecutive throughput samples over "
                  << static_cast<double>(c.run_until - c.run_since) / 1000.0
                  << " s, while the latest " << c.streak
                  << " multi-turn requests all selected root with zero prefix-cache hits.";
        insights.push_back(insight_available(
            "prefix.reuse_collapsed", "warning", "Prefix reuse has collapsed", statement.str(),
            {{"server_instance_id", latest_instance_},
             {"host_capacity_bytes", c.capacity},
             {"saturated_samples", c.run_samples},
             {"saturated_since_unix_ms", c.run_since},
             {"saturated_until_unix_ms", c.run_until},
             {"consecutive_multiturn_root_misses", c.streak},
             {"recomputed_prefill_tokens", c.streak_refill},
             {"sample_request_ids", sample_ids}},
            "Restart restores reuse temporarily. Preserve the log and investigate Host context "
            "checkpoint ownership before the pool saturates again.",
            "measured",
            {{"requests", c.streak},
             {"throughput_events", c.run_samples},
             {"window_s", static_cast<double>(c.run_until - c.run_since) / 1000.0}}));
    }

    const double n_done       = static_cast<double>(dones_);
    const double mean_queue   = paired > 0 ? b_.queue_wait_sum / paired : 0.0;
    const double mean_prefill = b_.prefill_sum / n_done;
    const double mean_decode  = b_.decode_sum / n_done;
    const int cause_max       = std::max({b_.queued, b_.prefill, b_.decode, b_.mixed});
    std::string cause         = "mixed";
    std::string cause_id      = "latency.mixed";
    std::vector<std::int64_t> cause_ids;
    if (cause_max == b_.queued && b_.queued > 0) {
        cause     = "queued behind concurrency";
        cause_id  = "latency.queued_behind_concurrency";
        cause_ids = b_.queued_ids;
    } else if (cause_max == b_.prefill && b_.prefill > 0) {
        cause     = "long prefill";
        cause_id  = "latency.prefill_dominated";
        cause_ids = b_.prefill_ids;
    } else if (cause_max == b_.decode && b_.decode > 0) {
        cause     = "decode-dominated";
        cause_id  = "latency.decode_dominated";
        cause_ids = b_.decode_ids;
    }

    std::ostringstream sat;
    sat << paired << " paired of " << dones_ << " request_done: " << b_.queued << " queued, "
        << b_.prefill << " prefill-dominated, " << b_.decode << " decode-dominated, " << b_.mixed
        << " mixed. Dominant cause: " << cause << ". Mean queue wait " << (mean_queue * 1000.0)
        << " ms, mean prefill " << (mean_prefill * 1000.0) << " ms, mean decode "
        << (mean_decode * 1000.0) << " ms. Scheduler peak waiting=" << max_waiting_
        << " running=" << max_running_ << " prefilling=" << max_prefill_ << ".";

    const bool pressure = b_.queued > 0 && (b_.queued * 3 >= paired || max_waiting_ > 0);
    insights.push_back(insight_available(
        cause_id, pressure ? "warning" : "info", "Saturation vs latency", sat.str(),
        {{"queued", b_.queued},
         {"prefill_dominated", b_.prefill},
         {"decode_dominated", b_.decode},
         {"mixed", b_.mixed},
         {"mean_queue_wait_s", mean_queue},
         {"mean_prefill_s", mean_prefill},
         {"mean_decode_s", mean_decode},
         {"scheduler_peak",
          {{"waiting", max_waiting_}, {"running", max_running_}, {"prefilling", max_prefill_}}},
         {"sample_request_ids", cause_ids}},
        pressure ? "Queued wait is a concurrency/backlog problem, not a slow kernel. "
                   "Raise --max-concurrency only if KV/headroom allows; otherwise the "
                   "engine is saturated."
                 : "",
        "measured", over));

    const double mean_prepare = b_.prepare_sum / n_done;
    const double mean_vision  = b_.vision_sum / n_done;
    const double mean_ttft    = b_.ttft_sum / n_done;
    const double ttft_body    = mean_prepare + mean_prefill + mean_vision;
    std::string ttft_cause    = "mixed";
    std::string ttft_id       = "latency.ttft_mixed";
    std::string ttft_rec;
    if (mean_prefill >= mean_prepare && mean_prefill >= mean_vision &&
        mean_prefill >= 0.4 * std::max(ttft_body, mean_ttft)) {
        ttft_cause = "prefill";
        ttft_id    = "latency.ttft_prefill_dominated";
        ttft_rec   = "TTFT is prefill-dominated. --prefill-chunk is the lever, not decode kernels.";
    } else if (mean_prepare >= mean_prefill &&
               mean_prepare >= 0.4 * std::max(ttft_body, mean_ttft)) {
        ttft_cause = "prepare";
        ttft_id    = "latency.ttft_prepare_dominated";
        ttft_rec   = "TTFT is prepare-dominated (tokenize/media), not GPU decode.";
    } else if (mean_vision >= 0.4 * std::max(ttft_body, mean_ttft) && mean_vision > 0.0) {
        ttft_cause = "vision";
        ttft_id    = "latency.ttft_vision_dominated";
        ttft_rec   = "TTFT is vision-preprocess dominated.";
    }
    std::ostringstream ttft_stmt;
    ttft_stmt << "Mean TTFT " << (mean_ttft * 1000.0) << " ms over " << dones_
              << " request_done: prepare " << (mean_prepare * 1000.0) << " ms, prefill "
              << (mean_prefill * 1000.0) << " ms, vision " << (mean_vision * 1000.0)
              << " ms (decode " << (mean_decode * 1000.0)
              << " ms is after first token). Dominant TTFT component: " << ttft_cause << ".";
    insights.push_back(insight_available(ttft_id, "info", "TTFT decomposition", ttft_stmt.str(),
                                         {{"mean_ttft_s", mean_ttft},
                                          {"mean_prepare_s", mean_prepare},
                                          {"mean_prefill_s", mean_prefill},
                                          {"mean_vision_s", mean_vision},
                                          {"mean_decode_s", mean_decode},
                                          {"dominant", ttft_cause}},
                                         ttft_rec, "measured", over));

    const double multi_hit_ratio =
        multi_prompt_tokens_ == 0
            ? 0.0
            : static_cast<double>(multi_hit_tokens_) / static_cast<double>(multi_prompt_tokens_);
    std::ostringstream reuse_stmt;
    reuse_stmt << "Reuse mix over " << dones_ << " request_done: root single-turn "
               << reuse_root_single_ << " (expected), root multi-turn " << reuse_root_multi_
               << ", checkpoint " << reuse_checkpoint_ << ", other " << reuse_other_
               << ". Multi-turn prefix-hit ratio " << (multi_hit_ratio * 100.0) << "% ("
               << multi_hit_tokens_ << "/" << multi_prompt_tokens_ << " tokens).";
    insights.push_back(insight_available("prefix.reuse_mix",
                                         reuse_root_multi_ > 0 ? "notice" : "info",
                                         "Prefix-cache reuse mix", reuse_stmt.str(),
                                         {{"root_single_turn", reuse_root_single_},
                                          {"root_multi_turn", reuse_root_multi_},
                                          {"checkpoint", reuse_checkpoint_},
                                          {"other", reuse_other_},
                                          {"multi_turn_prompt_tokens", multi_prompt_tokens_},
                                          {"multi_turn_hit_tokens", multi_hit_tokens_},
                                          {"multi_turn_hit_ratio", multi_hit_ratio}},
                                         "", "measured", over));
    if (reuse_root_multi_ > 0) {
        std::ostringstream miss;
        miss << reuse_root_multi_ << " of " << dones_
             << " request_done were multi-turn (message_count>=2) on root with "
             << "prefix_cache_hit_tokens often 0. A single-message root is expected; "
             << "a multi-turn root may indicate a missing or evicted checkpoint.";
        insights.push_back(insight_available(
            "prefix.multiturn_root", "warning", "Multi-turn conversations resetting the prefix",
            miss.str(),
            {{"root_multi_turn", reuse_root_multi_},
             {"root_single_turn", reuse_root_single_},
             {"samples", reset_multi_samples_}},
            "Check seed store / turn checkpoints. restore_turn_checkpoint or seed_prefix "
            "should fire when message_count>=2.",
            "measured", over));
    }

    // Speculative decoding: the counters are measured; any draft-window advice is inferred from
    // the position curve and says so. No telemetry or no drafts is unavailable, never 0%.
    {
        constexpr double kMinUsefulPositionRate          = 0.20;
        constexpr std::uint64_t kMinDraftRoundsForAdvice = 50;
        auto pct                                         = [](double ratio) {
            std::ostringstream o;
            o << std::fixed << std::setprecision(1) << (ratio * 100.0) << "%";
            return o.str();
        };
        const auto spec_over = nlohmann::json{
            {"requests", spec.requests_with_drafts},
            {"requests_with_telemetry", spec.requests_with_telemetry},
            {"examined_requests", spec_scope_dones},
            {"other_instance_requests", other_instance},
            {"server_instance_id", latest_instance_},
            {"draft_rounds", spec.draft_rounds},
            {"window_s",
             spec.tmax > spec.tmin ? static_cast<double>(spec.tmax - spec.tmin) / 1000.0 : 0.0}};
        if (spec.requests_with_telemetry == 0) {
            std::ostringstream stmt;
            stmt << "no request_done of the latest server instance carries speculative telemetry "
                 << "with a configured backend; " << spec.requests_backend_none << " of "
                 << spec_scope_dones
                 << " report backend none and the rest have no speculative object";
            insights.push_back(insight_unavailable(
                "speculative.draft_acceptance",
                "Speculative decoding telemetry is not in the window", stmt.str(),
                {{"requests_backend_none", spec.requests_backend_none},
                 {"requests", spec_scope_dones},
                 {"server_instance_id", latest_instance_}},
                {{"requests", 0},
                 {"examined_requests", spec_scope_dones},
                 {"other_instance_requests", other_instance}}));
        } else if (spec.drafted == 0) {
            std::ostringstream stmt;
            stmt << "backend " << spec.backend << " with draft window " << spec.draft_window
                 << " is configured, but the " << spec.requests_with_telemetry
                 << " request_done with telemetry recorded 0 drafted tokens (" << spec.fallback
                 << " fallback steps); acceptance cannot be measured from zero drafts";
            insights.push_back(
                insight_unavailable("speculative.draft_acceptance",
                                    "Speculative decoding has not drafted yet", stmt.str(),
                                    {{"backend", spec.backend},
                                     {"draft_window", spec.draft_window},
                                     {"drafted_tokens", 0},
                                     {"fallback_steps", spec.fallback},
                                     {"requests_with_telemetry", spec.requests_with_telemetry},
                                     {"server_instance_id", latest_instance_}},
                                    spec_over));
        } else {
            std::uint64_t first_half_steps     = 0;
            std::uint64_t first_half_fallback  = 0;
            std::uint64_t second_half_steps    = 0;
            std::uint64_t second_half_fallback = 0;
            for (std::size_t i = 0; i < spec.steps.size(); ++i) {
                const bool second_half = i * 2 >= spec_scope_dones;
                (second_half ? second_half_steps : first_half_steps) += spec.steps[i].steps;
                (second_half ? second_half_fallback : first_half_fallback) +=
                    spec.steps[i].fallback;
            }
            nlohmann::json per_position_accepted = nlohmann::json::array();
            nlohmann::json per_position_rate     = nlohmann::json::array();
            int effective_window                 = 0;
            bool prefix_useful                   = true;
            std::ostringstream curve;
            for (std::size_t p = 0; p < spec.positions.size(); ++p) {
                const auto& pos   = spec.positions[p];
                const double rate = pos.rounds > 0
                                        ? std::min(1.0, static_cast<double>(pos.accepted) /
                                                            static_cast<double>(pos.rounds))
                                        : 0.0;
                per_position_accepted.push_back(pos.accepted);
                per_position_rate.push_back(rate);
                if (prefix_useful && rate >= kMinUsefulPositionRate) {
                    ++effective_window;
                } else {
                    prefix_useful = false;
                }
                curve << (p == 0 ? "" : ", ") << "P" << (p + 1) << " " << pct(rate);
            }
            const double acceptance =
                static_cast<double>(spec.accepted) / static_cast<double>(spec.drafted);
            const std::uint64_t steps = spec.draft_rounds + spec.fallback;
            auto share                = [](std::uint64_t part, std::uint64_t whole) {
                return whole > 0 ? static_cast<double>(part) / static_cast<double>(whole) : 0.0;
            };
            const double fallback_share = share(spec.fallback, steps);
            const double first_share    = share(first_half_fallback, first_half_steps);
            const double second_share   = share(second_half_fallback, second_half_steps);
            const bool fallback_high    = spec.fallback >= 10 && fallback_share >= 0.20;
            const bool fallback_rising =
                second_half_fallback >= 10 && second_share >= first_share + 0.10;
            const bool advise_window = spec.draft_rounds >= kMinDraftRoundsForAdvice &&
                                       effective_window >= 1 &&
                                       effective_window < spec.draft_window;

            nlohmann::json windows_seen = nlohmann::json::object();
            for (const auto& [w, n] : spec.windows_seen) { windows_seen[std::to_string(w)] = n; }

            std::ostringstream stmt;
            stmt << "Backend " << spec.backend << ", draft window " << spec.draft_window << ": "
                 << spec.accepted << " of " << spec.drafted << " drafted tokens accepted ("
                 << pct(acceptance) << ") over " << spec.draft_rounds << " draft rounds in "
                 << spec.requests_with_drafts << " requests with drafts ("
                 << spec.requests_with_telemetry << " with telemetry, " << spec_scope_dones
                 << " examined on the latest server instance). Acceptance by draft position: "
                 << curve.str() << ". Fallback steps " << spec.fallback << " = "
                 << pct(fallback_share) << " of " << steps << " decode steps (first half "
                 << pct(first_share) << ", second half " << pct(second_share) << ").";

            std::ostringstream rec;
            if (fallback_high || fallback_rising) {
                rec << "Measured: the engine decoded without a draft on " << pct(fallback_share)
                    << " of steps";
                if (fallback_rising) {
                    rec << ", rising from " << pct(first_share) << " to " << pct(second_share)
                        << " across the window";
                }
                rec << ". The log records that those steps ran on the ordinary decode path, not "
                       "why. "
                       "Inferred: compare the high_fallback_samples (media, thinking, message "
                       "count) with requests that drafted normally, and check the engine log "
                       "around them, before changing the draft window. ";
            }
            if (advise_window) {
                rec << "Inferred, not measured: draft positions " << (effective_window + 1) << ".."
                    << spec.draft_window << " are accepted in under " << pct(kMinUsefulPositionRate)
                    << " of draft rounds, so a draft window of " << effective_window
                    << " would drop mostly rejected work. Verify with "
                    << "--draft-tokens " << effective_window
                    << " and compare decode tok/s before keeping it.";
            }
            std::string recommendation = rec.str();
            while (!recommendation.empty() && recommendation.back() == ' ') {
                recommendation.pop_back();
            }
            insights.push_back(insight_available(
                "speculative.draft_acceptance",
                (fallback_high || fallback_rising) ? "warning" : "info",
                "Speculative draft acceptance", stmt.str(),
                {{"backend", spec.backend},
                 {"server_instance_id", latest_instance_},
                 {"draft_window", spec.draft_window},
                 {"draft_windows_seen", windows_seen},
                 {"draft_rounds", spec.draft_rounds},
                 {"rounds_reported", spec.rounds_reported},
                 {"drafted_tokens", spec.drafted},
                 {"accepted_tokens", spec.accepted},
                 {"acceptance_ratio", acceptance},
                 {"accepted_per_position", per_position_accepted},
                 {"per_position_rate", per_position_rate},
                 {"min_useful_position_rate", kMinUsefulPositionRate},
                 {"effective_draft_window", effective_window},
                 {"inferred_draft_window",
                  advise_window ? nlohmann::json(effective_window) : nlohmann::json(nullptr)},
                 {"fallback_steps", spec.fallback},
                 {"fallback_share", fallback_share},
                 {"fallback_share_first_half", first_share},
                 {"fallback_share_second_half", second_share},
                 {"requests_with_drafts", spec.requests_with_drafts},
                 {"requests_with_telemetry", spec.requests_with_telemetry},
                 {"sample_request_ids", spec.sample_ids},
                 {"low_acceptance_samples", spec.low_acceptance_samples},
                 {"high_fallback_samples", spec.high_fallback_samples}},
                recommendation, "measured", spec_over));
        }
    }

    // Context and KV capacity pressure on the latest server instance: the configured limits come
    // from its server_start, traffic from its request_done records, occupancy and pressure from
    // its throughput samples. Occupancy counts retained checkpoints while prefix reuse is on, so a
    // full pool is only pressure when the engine also evicted, spilled or queued.
    {
        constexpr double kNearContextRatio = 0.90;
        constexpr double kHighKvRatio      = 0.90;
        auto pct                           = [](double ratio) {
            std::ostringstream o;
            o << std::fixed << std::setprecision(1) << (ratio * 100.0) << "%";
            return o.str();
        };

        const auto max_context     = json_i64(latest_engine_, "max_context");
        const auto max_concurrency = json_i64(latest_engine_, "max_concurrency");
        const auto kv_tokens       = json_i64(latest_engine_, "kv_capacity");
        const auto kv_page_groups  = json_i64(latest_engine_, "kv_capacity_page_groups");
        const bool prefix_reuse    = latest_engine_.value("prefix_reuse", false);
        const bool have_config     = max_context > 0 && kv_page_groups > 0;

        // The k-th smallest prompt (0-based) from the histogram.
        const std::uint64_t n_req = s.dones;
        auto nth_prompt           = [&](std::uint64_t k) {
            std::uint64_t seen = 0;
            for (const auto& [tokens, count] : s.prompts) {
                seen += count;
                if (seen > k) { return tokens; }
            }
            return std::int64_t{0};
        };
        const std::int64_t prompt_median = n_req ? nth_prompt((n_req - 1) / 2) : 0;
        const std::int64_t prompt_p90 =
            n_req ? nth_prompt(std::min(n_req - 1, (n_req * 9) / 10)) : 0;
        const std::int64_t prompt_max    = n_req ? s.prompts.rbegin()->first : 0;
        const std::int64_t largest_total = s.largest.empty() ? 0 : s.largest.front().total;
        const double largest_ratio =
            max_context > 0 ? static_cast<double>(largest_total) / static_cast<double>(max_context)
                            : 0.0;
        const int near_limit           = s.near_limit;
        nlohmann::json largest_samples = nlohmann::json::array();
        for (const auto& l : s.largest) {
            const double ratio =
                max_context > 0 ? static_cast<double>(l.total) / static_cast<double>(max_context)
                                : 0.0;
            largest_samples.push_back({{"request_id", l.id},
                                       {"prompt_tokens", l.prompt},
                                       {"completion_tokens", l.completion},
                                       {"total_tokens", l.total},
                                       {"context_ratio", ratio},
                                       {"finish_reason", l.finish}});
        }

        const std::size_t occupancy_samples        = s.occupancy_samples;
        const std::int64_t kv_now                  = s.kv_now;
        const std::int64_t kv_high                 = s.kv_high;
        const std::int64_t peak_running            = s.peak_running;
        const std::int64_t peak_waiting            = s.peak_waiting;
        const std::int64_t spill_pages             = s.spill_pages;
        const int admission_expired                = s.admission_expired;
        const nlohmann::json admission_expired_ids = s.admission_expired_ids;
        const double kv_now_ratio =
            kv_page_groups > 0 ? static_cast<double>(kv_now) / static_cast<double>(kv_page_groups)
                               : 0.0;
        const double kv_high_ratio =
            kv_page_groups > 0 ? static_cast<double>(kv_high) / static_cast<double>(kv_page_groups)
                               : 0.0;
        const std::int64_t pressure_events = s.preemptions + spill_pages;

        nlohmann::json config = {
            {"max_context", max_context},
            {"max_concurrency", max_concurrency},
            {"kv_capacity_tokens", kv_tokens},
            {"kv_page_groups", kv_page_groups},
            {"tokens_per_page_group", kv_page_groups > 0 ? kv_tokens / kv_page_groups : 0},
            {"prefix_reuse", prefix_reuse}};
        nlohmann::json requests_evidence = {
            {"count", n_req},
            {"prompt_tokens",
             {{"median", prompt_median}, {"p90", prompt_p90}, {"max", prompt_max}}},
            {"largest_total_tokens", largest_total},
            {"largest_context_ratio", largest_ratio},
            {"near_limit_count", near_limit},
            {"near_limit_ratio", kNearContextRatio}};
        nlohmann::json kv_evidence        = {{"pages_now", kv_now},
                                             {"pages_high_water", kv_high},
                                             {"pages_capacity", kv_page_groups},
                                             {"utilization_now", kv_now_ratio},
                                             {"utilization_high_water", kv_high_ratio},
                                             {"samples", occupancy_samples},
                                             {"includes_retained_checkpoints", prefix_reuse}};
        nlohmann::json scheduler_evidence = {
            {"peak_running", peak_running},
            {"peak_waiting", peak_waiting},
            {"admission_expired", admission_expired},
            {"admission_expired_request_ids", admission_expired_ids}};
        nlohmann::json pressure_evidence = {{"preemptions", s.preemptions},
                                            {"spill_pages", spill_pages}};
        const auto cap_over              = nlohmann::json{{"requests", n_req},
                                                          {"throughput_events", occupancy_samples},
                                                          {"server_instance_id", latest_instance_}};

        if (!have_config) {
            insights.push_back(insight_unavailable(
                "capacity.context_kv_pressure", "Capacity limits are not in the log window",
                "the latest server instance has no server_start with max_context and "
                "kv_capacity_page_groups, so traffic cannot be compared to a limit; prompt "
                "sizes are reported without a pressure verdict",
                {{"server_instance_id", latest_instance_},
                 {"requests", requests_evidence},
                 {"largest_requests", largest_samples}},
                cap_over));
        } else if (occupancy_samples == 0) {
            insights.push_back(insight_unavailable(
                "capacity.context_kv_pressure", "KV occupancy has not been sampled yet",
                "the latest server instance has capacity configuration but no throughput record "
                "with context_cache.occupancy.device_main_kv_pages; KV utilization is unknown, "
                "not zero",
                {{"server_instance_id", latest_instance_},
                 {"config", config},
                 {"requests", requests_evidence},
                 {"largest_requests", largest_samples}},
                cap_over));
        } else {
            std::string kind     = "comfortable";
            std::string severity = "info";
            std::ostringstream rec;
            if (s.preemptions > 0) {
                kind     = "preempted";
                severity = "notice";
                rec << "Measured: " << s.preemptions << " pauses made room, with " << spill_pages
                    << " KV pages spilled. Inferred: raise --kv-capacity if VRAM "
                       "allows, or reduce concurrent demand to reduce recovery work.";
            } else if (near_limit > 0) {
                kind     = "large_prompt";
                severity = "warning";
                rec << "Measured: " << near_limit << " request(s) used at least "
                    << pct(kNearContextRatio) << " of max_context " << max_context
                    << "; the largest reached " << largest_total
                    << " tokens. Inferred: raise "
                       "--max-context (and --kv-capacity to hold it) if VRAM allows, or shorten "
                       "those conversations; a request over the limit is rejected at admission.";
            } else if ((peak_waiting > 0 || admission_expired > 0) && max_concurrency > 0 &&
                       peak_running >= max_concurrency) {
                // Every lane was busy while requests waited: the concurrency cap, not the pool.
                kind     = "lanes_full";
                severity = admission_expired > 0 ? "warning" : "notice";
                rec << "Measured: requests waited (peak " << peak_waiting << " waiting, "
                    << admission_expired << " expired before admission) while all "
                    << max_concurrency << " lanes were running; KV occupancy peaked at "
                    << pct(kv_high_ratio)
                    << ". Inferred: the concurrency limit is what queued "
                       "them; raise --max-concurrency only if KV capacity and VRAM leave room "
                       "for another request's context.";
            } else if ((peak_waiting > 0 || admission_expired > 0) &&
                       kv_high_ratio >= kHighKvRatio) {
                // Lanes were free but requests still waited: the pool could not admit them.
                kind     = "kv_admission";
                severity = "warning";
                rec << "Measured: requests waited (peak " << peak_waiting << " waiting, "
                    << admission_expired << " expired before admission) with only " << peak_running
                    << " of " << max_concurrency
                    << " lanes running and KV "
                       "occupancy at "
                    << pct(kv_high_ratio)
                    << ", while the largest single "
                       "request used "
                    << pct(largest_ratio)
                    << " of max_context. Inferred: "
                       "concurrent requests exhausted the KV pool; raise --kv-capacity if VRAM "
                       "allows, otherwise lower --max-context to bound each request's KV growth.";
            } else if (kv_high_ratio >= kHighKvRatio && pressure_events > 0) {
                // No queueing: the engine made room by spilling KV to host.
                kind     = "cache_churn";
                severity = "notice";
                rec << "Measured: KV occupancy peaked at " << pct(kv_high_ratio)
                    << " and the "
                       "engine spilled "
                    << spill_pages
                    << " KV pages with no request waiting. "
                       "Host transfers add recovery work. Inferred: raise --kv-capacity if VRAM "
                       "allows; this alone does not establish a concurrency problem.";
            } else if (kv_high_ratio >= kHighKvRatio) {
                kind = "retained_cache";
                rec << "Measured: KV occupancy peaked at " << pct(kv_high_ratio)
                    << " without preemptions, spills or queueing. With prefix reuse "
                    << (prefix_reuse ? "on" : "off")
                    << " the pool holds retained checkpoints, so a full pool alone is not "
                       "pressure. Nothing to change from this window.";
            }

            std::ostringstream stmt;
            stmt << "Configured max_context " << max_context << " tokens, KV capacity " << kv_tokens
                 << " tokens (" << kv_page_groups << " page groups), max concurrency "
                 << max_concurrency << ", prefix reuse " << (prefix_reuse ? "on" : "off")
                 << ". Over " << n_req
                 << " request_done on the latest server instance: prompt tokens median "
                 << prompt_median << ", p90 " << prompt_p90 << ", max " << prompt_max
                 << "; largest request " << largest_total << " tokens = " << pct(largest_ratio)
                 << " of max_context, " << near_limit << " within " << pct(kNearContextRatio)
                 << " of it. KV pages " << kv_now << " of " << kv_page_groups << " now ("
                 << pct(kv_now_ratio) << "), high-water " << kv_high << " (" << pct(kv_high_ratio)
                 << ") over " << occupancy_samples << " samples"
                 << (prefix_reuse ? ", including retained checkpoints" : "")
                 << ". Scheduler peak running " << peak_running << " of " << max_concurrency
                 << ", waiting " << peak_waiting << ", " << admission_expired
                 << " expired before admission. Pressure over the window: preemptions "
                 << s.preemptions << ", spill pages " << spill_pages << ". Verdict: " << kind
                 << ".";

            insights.push_back(insight_available("capacity.context_kv_pressure", severity,
                                                 "Context and KV capacity pressure", stmt.str(),
                                                 {{"server_instance_id", latest_instance_},
                                                  {"kind", kind},
                                                  {"config", config},
                                                  {"requests", requests_evidence},
                                                  {"largest_requests", largest_samples},
                                                  {"kv", kv_evidence},
                                                  {"scheduler", scheduler_evidence},
                                                  {"pressure", pressure_evidence},
                                                  {"high_kv_ratio", kHighKvRatio}},
                                                 rec.str(), "measured", cap_over));
        }
    }

    // Content/reasoning_content are not in schema_version 10 request logs.
    insights.push_back(insight_unavailable(
        "client.content_fields", "Visitor content is not in the request log",
        "result.content and reasoning_content are not written to request_done; "
        "empty-reply-vs-reasoning cannot be confirmed from this source",
        {{"schema_version", 10},
         {"looked_for", nlohmann::json::array({"content", "reasoning_content"})},
         {"requests", dones_}},
        over));

    if (output_limit_thinking_ > 0) {
        std::ostringstream stmt;
        stmt << output_limit_thinking_ << " of " << dones_
             << " request_done finished on output_limit with enable_thinking=true"
             << " (" << output_limit_hit_cap_ << " also hit requested_output_tokens). "
             << thinking_requests_ << " of " << dones_ << " had thinking enabled.";
        const double cap_mean = static_cast<double>(output_limit_cap_sum_) /
                                static_cast<double>(output_limit_thinking_);
        insights.push_back(insight_available(
            "client.output_limit_while_thinking",
            output_limit_thinking_ * 5 >= static_cast<int>(dones_) ? "warning" : "notice",
            "Thinking requests hitting output_limit", stmt.str(),
            {{"output_limit_thinking", output_limit_thinking_},
             {"hit_requested_cap", output_limit_hit_cap_},
             {"thinking_requests", thinking_requests_},
             {"mean_requested_output_tokens", cap_mean},
             {"sample_request_ids", output_limit_ids_}},
            "Inferred: a thinking model with a small max_tokens can spend the budget on "
            "reasoning and return an empty visitor reply. Content fields are not in this log, "
            "so raise requested_output_tokens and compare finish_reason.",
            "measured", over));
    }

    insights.push_back(insight_unavailable(
        "client.narrated_tool_intent", "Narrated tool intent cannot be scored from JSONL",
        "detecting narrated-intent-with-no-tools needs visitor-facing text; the request log "
        "does not store content. tool_count is measurable and is reported in evidence.",
        {{"requests_with_tools", tools_declared_},
         {"requests", dones_},
         {"requests_without_tools", static_cast<int>(dones_) - tools_declared_}},
        over));

    return report;
}

// Analyzes a whole JSONL blob: every line folded, then one report.
inline nlohmann::json analyze_request_log_jsonl(std::string_view jsonl, std::string_view path) {
    RequestLogInsights insights;
    std::size_t begin = 0;
    while (begin < jsonl.size()) {
        const auto end  = jsonl.find('\n', begin);
        const auto stop = end == std::string_view::npos ? jsonl.size() : end;
        insights.fold_line(jsonl.substr(begin, stop - begin));
        begin = stop + 1;
    }
    return insights.report(path);
}

inline nlohmann::json request_log_unconfigured_report() {
    nlohmann::json report;
    report["source"]   = {{"request_log", "unconfigured"}, {"path", ""}};
    report["insights"] = nlohmann::json::array(
        {insight_unavailable("source.request_log", "Request log is not configured",
                             "no request_done records in window", {{"path", ""}})});
    return report;
}

inline nlohmann::json request_log_missing_report(const std::string& path) {
    nlohmann::json report;
    report["source"]   = {{"request_log", "missing"}, {"path", path}};
    report["insights"] = nlohmann::json::array(
        {insight_unavailable("source.request_log", "Request log is not present",
                             "no request_done records in window", {{"path", path}})});
    return report;
}

// The supervisor reads an existing log once, in the background, before it can report on it.
// Until then a partial fold would describe whichever engine instance the read has reached, so the
// report says it is still reading rather than showing that.
inline nlohmann::json request_log_reading_report(const std::string& path, std::uint64_t bytes_read,
                                                 std::uint64_t bytes_total) {
    const double mib = 1024.0 * 1024.0;
    const double percent =
        bytes_total > 0 ? 100.0 * static_cast<double>(bytes_read) / static_cast<double>(bytes_total)
                        : 0.0;
    std::ostringstream statement;
    statement << std::fixed << std::setprecision(0)
              << "Reading the request log: " << static_cast<double>(bytes_read) / mib << " of "
              << static_cast<double>(bytes_total) / mib << " MiB (" << percent
              << "%). Findings appear once the whole log has been read.";
    nlohmann::json report;
    report["source"]   = {{"request_log", "reading"},
                          {"path", path},
                          {"bytes_read", bytes_read},
                          {"bytes_total", bytes_total}};
    report["insights"] = nlohmann::json::array({insight_unavailable(
        "source.request_log", "Reading the request log", statement.str(),
        {{"path", path}, {"bytes_read", bytes_read}, {"bytes_total", bytes_total}})});
    return report;
}

inline void append_admin_vram_insights(nlohmann::json& report, const nlohmann::json& admin,
                                       const std::string& note) {
    if (!report.contains("insights") || !report["insights"].is_array()) {
        report["insights"] = nlohmann::json::array();
    }
    if (!admin.is_object()) {
        report["insights"].push_back(insight_unavailable(
            "vram.admin", "Admin VRAM is not available",
            note.empty() ? "admin/vram was not readable; cannot tell if any tier is releasable"
                         : note,
            {{"note", note}}));
        return;
    }
    nlohmann::json pinned   = nlohmann::json::array();
    nlohmann::json released = nlohmann::json::array();
    if (admin.contains("tiers") && admin.at("tiers").is_array()) {
        for (const auto& tier : admin.at("tiers")) {
            const auto min_b = json_i64(tier, "min_bytes");
            const auto max_b = json_i64(tier, "max_bytes");
            const bool rel   = tier.value("released", false);
            if (rel) { released.push_back(tier.value("name", "?")); }
            if (min_b > 0 && min_b == max_b) {
                pinned.push_back({{"name", tier.value("name", "")},
                                  {"min_bytes", min_b},
                                  {"max_bytes", max_b},
                                  {"reclaimable_bytes", json_i64(tier, "reclaimable_bytes")},
                                  {"released", rel}});
            }
        }
    }
    const auto over = nlohmann::json{{"requests", 0},
                                     {"admin_tiers", pinned.size() + released.size()},
                                     {"last_transition", admin.value("last_transition", "")},
                                     {"last_reason", admin.value("last_reason", "")}};
    if (!pinned.empty()) {
        report["insights"].push_back(insight_available(
            "vram.tier_pinned_unreleasable", "warning",
            "Admin VRAM is enabled but a tier cannot be released",
            "A tier has min_bytes == max_bytes while --admin-vram is on. "
            "--prefix-cache-mib N pins seed min=max=N, so reclaimable_bytes stays 0 "
            "and the admin surface looks healthy while nothing can be released.",
            {{"pinned_tiers", pinned},
             {"last_transition", admin.value("last_transition", "")},
             {"last_reason", admin.value("last_reason", "")}},
            "Omit --prefix-cache-mib or set a max above min if you want idle release.", "measured",
            over));
    }
    if (!released.empty()) {
        report["insights"].push_back(insight_available(
            "vram.tier_currently_released", "notice", "A VRAM tier is currently released",
            "Released tiers: " + released.dump() +
                ". The engine is serving degraded (no cross-request prefix seeding) until reclaim. "
                "Release is ~120x cheaper than reclaim on this hardware.",
            {{"released", released},
             {"last_transition", admin.value("last_transition", "")},
             {"last_reason", admin.value("last_reason", "")}},
            "Reclaim before a traffic burst; released checkpoints can require more recomputation.",
            "measured", over));
    }
}

} // namespace ninfer::supervisor
