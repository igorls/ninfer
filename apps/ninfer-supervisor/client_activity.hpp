#pragma once

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::supervisor {

// Request observations, not diagnoses inferred from a cache percentage. Identity is scoped
// to an engine instance because request IDs restart at one when the engine restarts.
class ClientActivityLog {
    using Json = nlohmann::json;
    struct Pending { std::string client, instance; int tools = 0; std::int64_t t = 0; };
    struct Event {
        std::int64_t t = 0;
        std::string client, instance, outcome, fallback;
        int tools = 0;
        bool root = false, reuse_disabled = false, admission_timeout = false, first_queue = false;
        std::optional<double> ttft, queue, prefill;
        std::optional<std::uint64_t> prompt, hit, computed;
    };
    std::map<std::string, Pending> pending_;
    std::deque<Event> events_;
    std::deque<std::pair<std::int64_t, std::string>> starts_;
    std::map<std::string, bool> reuse_disabled_;
    bool limited_ = false;

    static std::optional<double> milliseconds(const Json& j, const char* name) {
        const auto it = j.find(name);
        if (it == j.end() || !it->is_number()) return {};
        const double value = it->get<double>();
        return std::isfinite(value) && value >= 0 ? std::optional<double>{value * 1000} : std::nullopt;
    }
    static std::optional<std::uint64_t> tokens(const Json& j, const char* name) {
        const auto it = j.find(name);
        if (it == j.end() || !it->is_number_integer() || it->get<double>() < 0) return {};
        return it->get<std::uint64_t>();
    }
    static Json distribution(const std::vector<double>& values) {
        if (values.empty()) return Json{{"samples", 0}, {"mean", nullptr}, {"p50", nullptr}, {"p95", nullptr}};
        auto ordered = values;
        std::sort(ordered.begin(), ordered.end());
        double sum = 0;
        for (const auto v : ordered) sum += v;
        const auto percentile = [&](double p) { return ordered[static_cast<std::size_t>(std::ceil(p * ordered.size())) - 1]; };
        return Json{{"samples", ordered.size()}, {"mean", sum / ordered.size()}, {"p50", percentile(0.50)}, {"p95", percentile(0.95)}};
    }

public:
    void fold(std::string_view line) {
        const auto j = Json::parse(line, nullptr, false);
        if (!j.is_object()) return;
        try {
            const auto type = j.value("event", "");
            const auto instance = j.value("server_instance_id", "");
            const auto t = j.value("timestamp_unix_ms", std::int64_t{0});
            if (!t) return;
            if (type == "server_start") {
                for (auto it = pending_.begin(); it != pending_.end();) {
                    if (it->second.instance == instance) { ++it; continue; }
                    Event interrupted;
                    interrupted.t = t; interrupted.client = it->second.client;
                    interrupted.instance = it->second.instance; interrupted.tools = it->second.tools;
                    interrupted.outcome = "interrupted";
                    events_.push_back(std::move(interrupted));
                    it = pending_.erase(it);
                }
                starts_.emplace_back(t, instance);
                const auto engine = j.value("engine", Json::object());
                reuse_disabled_[instance] = engine.contains("prefix_reuse") && !engine.value("prefix_reuse", true);
                return;
            }
            if (type != "request_start" && type != "request_done" && type != "request_error" && type != "request_rejected") return;
            const auto r = j.value("request", Json::object());
            const auto id = r.value("request_id", std::uint64_t{0});
            const auto key = instance + ":" + std::to_string(id);
            if (type == "request_start") {
                if (pending_.size() >= 1024) {
                    const auto oldest = std::min_element(pending_.begin(), pending_.end(), [](const auto& a, const auto& b) { return a.second.t < b.second.t; });
                    pending_.erase(oldest);
                    limited_ = true;
                }
                pending_[key] = Pending{r.value("client", ""), instance, r.value("tool_count", 0), t};
                return;
            }
            Event e;
            e.t = t; e.instance = instance;
            e.client = r.value("client", ""); e.tools = r.value("tool_count", 0);
            if (const auto it = pending_.find(key); it != pending_.end()) {
                e.client = it->second.client; e.tools = it->second.tools;
                pending_.erase(it);
            }
            e.outcome = type == "request_done" ? "completed" : "failed";
            if (type == "request_done") {
                const auto result = j.value("result", Json::object());
                if (result.value("finish_reason", "") == "cancelled") e.outcome = "cancelled";
                e.prompt = tokens(result, "prompt_tokens");
                e.hit = tokens(result, "prefix_cache_hit_tokens");
                e.computed = tokens(result, "computed_prefill_tokens");
                const auto path = result.value("prefix_reuse_path", "");
                e.root = path == "root";
                e.reuse_disabled = reuse_disabled_[instance];
                const auto timing = j.value("timings_seconds", Json::object());
                e.ttft = milliseconds(timing, "ttft");
                e.prefill = milliseconds(timing, "prefill");
                const auto first = j.value("first_output_timing", Json());
                // New logs expose the first-output boundary. Older logs expose whole-request
                // queue wait; it is useful separately, but must not be called a TTFT component.
                const auto engine = first.is_object() ? first.value("engine", Json::object()) : j.value("engine_timing", Json::object());
                e.queue = milliseconds(engine, "queue_wait_seconds");
                e.first_queue = first.is_object() && e.queue.has_value();
                const auto generation = j.value("generation", Json::object());
                if (generation.is_object()) {
                    const auto admission = generation.value("admission", Json::object());
                    e.fallback = admission.value("fallback_reason", "");
                    if (e.fallback == "none") e.fallback.clear();
                }
            } else {
                const auto error = j.value("error", Json::object());
                const auto message = error.value("message", "");
                e.admission_timeout = message.find("expired while waiting for admission") != std::string::npos || message.find("expired waiting for admission") != std::string::npos;
            }
            events_.push_back(std::move(e));
            if (events_.size() > 8192) { events_.pop_front(); limited_ = true; }
        } catch (const Json::exception&) { /* malformed records do not become zero-valued measurements */ }
    }

    void expire(std::int64_t horizon) {
        std::erase_if(events_, [&](const auto& e) { return e.t < horizon; });
        std::erase_if(starts_, [&](const auto& e) { return e.first < horizon; });
        std::erase_if(pending_, [&](const auto& e) { return e.second.t < horizon; });
        std::set<std::string> retained;
        for (const auto& e : events_) retained.insert(e.instance);
        for (const auto& e : starts_) retained.insert(e.second);
        std::erase_if(reuse_disabled_, [&](const auto& e) { return !retained.contains(e.first); });
    }

    Json summary() const {
        struct Group {
            Json row = {{"completed", 0}, {"cancelled", 0}, {"failed", 0}, {"interrupted", 0}, {"active", 0}, {"admission_timeouts", 0}, {"full_prefills", 0}, {"unexplained_full_prefills", 0}, {"reuse_disabled_prefills", 0}, {"cache_samples", 0}, {"computed_samples", 0}, {"prompt_tokens", 0}, {"hit_tokens", 0}, {"computed_tokens", 0}, {"tool_count", 0}, {"last_seen_ms", 0}, {"queue_first_output_samples", 0}, {"fallbacks", Json::object()}};
            std::vector<double> ttft, queue, prefill;
            std::set<std::string> instances;
        };
        std::map<std::string, Group> groups;
        for (const auto& e : events_) {
            auto& g = groups[e.client]; auto& r = g.row;
            r[e.outcome] = r[e.outcome].get<int>() + 1;
            r["last_seen_ms"] = std::max(r["last_seen_ms"].get<std::int64_t>(), e.t);
            r["tool_count"] = std::max(r["tool_count"].get<int>(), e.tools);
            g.instances.insert(e.instance);
            if (e.admission_timeout) r["admission_timeouts"] = r["admission_timeouts"].get<int>() + 1;
            if (e.root) {
                r["full_prefills"] = r["full_prefills"].get<int>() + 1;
                const char* field = e.reuse_disabled ? "reuse_disabled_prefills" : "unexplained_full_prefills";
                if (e.reuse_disabled || e.fallback.empty()) r[field] = r[field].get<int>() + 1;
            }
            if (!e.fallback.empty()) r["fallbacks"][e.fallback] = r["fallbacks"].value(e.fallback, 0) + 1;
            if (e.prompt && e.hit && *e.hit <= *e.prompt) {
                r["cache_samples"] = r["cache_samples"].get<int>() + 1;
                r["prompt_tokens"] = r["prompt_tokens"].get<std::uint64_t>() + *e.prompt;
                r["hit_tokens"] = r["hit_tokens"].get<std::uint64_t>() + *e.hit;
            }
            if (e.computed) { r["computed_samples"] = r["computed_samples"].get<int>() + 1; r["computed_tokens"] = r["computed_tokens"].get<std::uint64_t>() + *e.computed; }
            if (e.ttft) g.ttft.push_back(*e.ttft);
            if (e.queue) g.queue.push_back(*e.queue);
            if (e.first_queue) r["queue_first_output_samples"] = r["queue_first_output_samples"].get<int>() + 1;
            if (e.prefill) g.prefill.push_back(*e.prefill);
        }
        for (const auto& [key, p] : pending_) {
            auto& group = groups[p.client];
            group.instances.insert(p.instance);
            auto& row = group.row;
            row["active"] = row["active"].get<int>() + 1;
            row["last_seen_ms"] = std::max(row["last_seen_ms"].get<std::int64_t>(), p.t);
        }
        std::vector<Json> rows;
        for (auto& [name, g] : groups) {
            auto& r = g.row;
            r["name"] = name; r["engine_instances"] = g.instances.size();
            const auto prompt = r["prompt_tokens"].get<std::uint64_t>();
            r["reuse_percent"] = prompt ? Json(100.0 * r["hit_tokens"].get<std::uint64_t>() / prompt) : Json();
            r["ttft_ms"] = distribution(g.ttft); r["queue_ms"] = distribution(g.queue); r["prefill_ms"] = distribution(g.prefill);
            rows.push_back(std::move(r));
        }
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
            const auto count = [](const auto& r) { return r.template value<int>("completed", 0) + r.template value<int>("cancelled", 0) + r.template value<int>("failed", 0) + r.template value<int>("active", 0); };
            return count(a) > count(b);
        });
        return Json{{"clients", rows}, {"engine_starts", starts_.size()}, {"limited", limited_}};
    }
};

} // namespace ninfer::supervisor
