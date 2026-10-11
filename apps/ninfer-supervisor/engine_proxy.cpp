#include "engine_proxy.hpp"
#include "frontends.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
#include <httplib.h>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

namespace ninfer::supervisor {
namespace {

// A request without streaming sends no headers until the whole answer is generated.
constexpr auto kUpstreamReadTimeout = std::chrono::hours(1);
constexpr auto kDisconnectPoll      = std::chrono::milliseconds(100);

// One forwarded request. The upstream call runs on its own detached thread and hands the response
// over through this bridge, so the caller receives each chunk as the engine writes it. Whoever
// sees the caller go away sets `cancelled`; the content receiver then returns false, the client
// closes the engine connection, and the engine stops the request.
struct ProxyBridge {
    std::mutex mutex;
    std::condition_variable cv;
    bool headers_ready = false;
    bool finished      = false;
    bool cancelled     = false;
    int status         = 0;
    httplib::Headers headers;
    std::deque<std::string> chunks;
    std::string error;

    void cancel() {
        std::lock_guard<std::mutex> lock(mutex);
        cancelled = true;
        cv.notify_all();
    }
};

} // namespace

void write_engine_error(httplib::Response& res, int status, const std::string& type,
                        const std::string& message) {
    res.status = status;
    res.set_content(
        nlohmann::json{{"error", {{"message", message}, {"type", type}, {"code", nullptr}}}}.dump(),
        "application/json");
}

void forward_to_engine(const httplib::Request& req, httplib::Response& res,
                       const EngineSpec& engine, const std::atomic<bool>& stopping,
                       const std::string& upstream_target) {
    const std::string key  = read_api_key_quiet(engine.api_key_file);
    const bool replace_key = !key.empty();
    httplib::Headers headers;
    for (const auto& [name, value] : req.headers) {
        if (forward_request_header(name) && !frontend_replaces_credentials(name, replace_key)) {
            headers.emplace(name, value);
        }
    }
    if (replace_key) { headers.emplace("Authorization", "Bearer " + key); }

    auto bridge = std::make_shared<ProxyBridge>();
    std::thread([bridge, host = engine_connect_host(engine), port = engine.engine_port,
                 method = req.method, target = upstream_target, headers = std::move(headers),
                 body = req.body]() mutable {
        httplib::Client client(host, port);
        client.set_connection_timeout(std::chrono::seconds(5));
        client.set_read_timeout(kUpstreamReadTimeout);
        client.set_write_timeout(std::chrono::seconds(60));
        httplib::Request upstream;
        upstream.method  = std::move(method);
        upstream.path    = std::move(target);
        upstream.headers = std::move(headers);
        upstream.body    = std::move(body);
        upstream.response_handler = [bridge](const httplib::Response& response) {
            std::lock_guard<std::mutex> lock(bridge->mutex);
            bridge->status        = response.status;
            bridge->headers       = response.headers;
            bridge->headers_ready = true;
            bridge->cv.notify_all();
            return !bridge->cancelled;
        };
        upstream.content_receiver = [bridge](const char* data, std::size_t size, std::size_t,
                                             std::size_t) {
            std::lock_guard<std::mutex> lock(bridge->mutex);
            if (bridge->cancelled) { return false; }
            bridge->chunks.emplace_back(data, size);
            bridge->cv.notify_all();
            return true;
        };
        const httplib::Result result = client.send(upstream);
        std::lock_guard<std::mutex> lock(bridge->mutex);
        if (!result && !bridge->headers_ready) { bridge->error = httplib::to_string(result.error()); }
        bridge->finished = true;
        bridge->cv.notify_all();
    }).detach();

    // Headers arrive with the first token of a stream, or with the whole answer otherwise. A
    // caller that leaves meanwhile cancels the engine request.
    {
        std::unique_lock<std::mutex> lock(bridge->mutex);
        while (!bridge->headers_ready && !bridge->finished) {
            bridge->cv.wait_for(lock, kDisconnectPoll);
            if (bridge->headers_ready || bridge->finished) { break; }
            lock.unlock();
            const bool closed = req.is_connection_closed() || stopping.load();
            lock.lock();
            if (closed) {
                bridge->cancelled = true;
                return;
            }
        }
        if (!bridge->headers_ready) {
            lock.unlock();
            write_engine_error(res, 502, "server_error",
                               "the NInfer engine is not reachable (" + bridge->error + ")");
            return;
        }
        res.status = bridge->status;
        for (const auto& [name, value] : bridge->headers) {
            if (forward_response_header(name)) { res.set_header(name, value); }
        }
    }
    std::string content_type;
    {
        std::lock_guard<std::mutex> lock(bridge->mutex);
        const auto type = bridge->headers.find("Content-Type");
        if (type != bridge->headers.end()) { content_type = type->second; }
        // An empty answer, such as the engine's 404 for a route it does not have, stays empty
        // and untyped.
        const auto length = bridge->headers.find("Content-Length");
        if (length != bridge->headers.end() && length->second == "0") { return; }
    }
    if (content_type.empty()) { content_type = "application/octet-stream"; }
    res.set_chunked_content_provider(
        content_type,
        [bridge, &stopping](std::size_t, httplib::DataSink& sink) {
            if (stopping.load()) {
                bridge->cancel();
                return false;
            }
            std::unique_lock<std::mutex> lock(bridge->mutex);
            bridge->cv.wait_for(lock, kDisconnectPoll, [&] {
                return !bridge->chunks.empty() || bridge->finished || bridge->cancelled;
            });
            while (!bridge->chunks.empty()) {
                std::string chunk = std::move(bridge->chunks.front());
                bridge->chunks.pop_front();
                lock.unlock();
                if (!sink.write(chunk.data(), chunk.size())) {
                    bridge->cancel();
                    return false;
                }
                lock.lock();
            }
            if (bridge->cancelled) { return false; }
            if (bridge->finished) { sink.done(); }
            return true;
        },
        [bridge](bool success) {
            if (!success) { bridge->cancel(); }
        });
}

void add_engine_bench_routes(httplib::Server& server, EngineSpecProvider engine,
                             const std::atomic<bool>& stopping,
                             std::function<bool(const std::string& remote)> local_peer) {
    const auto handler = [engine = std::move(engine), &stopping, local_peer = std::move(local_peer)](
                             const httplib::Request& req, httplib::Response& res) {
        if (!local_peer(req.remote_addr)) {
            write_engine_error(res, 403, "invalid_request_error", "the API bench is loopback-only");
            return;
        }
        if (!supervisor_control_header_ok(
                req.get_header_value(std::string(kSupervisorControlHeader)))) {
            write_engine_error(res, 403, "invalid_request_error",
                               "missing X-NInfer-Supervisor header");
            return;
        }
        const std::string target = engine_bench_target(req.target);
        if (target.empty()) {
            write_engine_error(res, 404, "invalid_request_error",
                               "the API bench forwards engine routes only");
            return;
        }
        forward_to_engine(req, res, engine(), stopping, target);
    };
    const std::string pattern = std::string(kEngineBenchPrefix) + "(/.*)?";
    server.Get(pattern, handler);
    server.Post(pattern, handler);
    server.Put(pattern, handler);
    server.Patch(pattern, handler);
    server.Delete(pattern, handler);
}

} // namespace ninfer::supervisor
