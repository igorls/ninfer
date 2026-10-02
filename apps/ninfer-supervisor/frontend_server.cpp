#include "frontend_server.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
#include <httplib.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>

namespace ninfer::supervisor {
namespace {

// The engine's default request limit, so an image upload that the engine accepts also passes the
// proxy.
constexpr std::size_t kMaxRequestBytes = 384ULL << 20;
// A request without streaming sends no headers until the whole answer is generated.
constexpr auto kUpstreamReadTimeout = std::chrono::hours(1);
constexpr auto kDisconnectPoll      = std::chrono::milliseconds(100);

const char* state_name(FrontendState state) {
    switch (state) {
    case FrontendState::Serving: return "serving";
    case FrontendState::AssetsMissing: return "assets_missing";
    case FrontendState::PortInUse: return "port_in_use";
    case FrontendState::Stopped: break;
    }
    return "stopped";
}

void write_json_error(httplib::Response& res, int status, const std::string& type,
                      const std::string& message) {
    res.status = status;
    res.set_content(
        nlohmann::json{{"error", {{"message", message}, {"type", type}, {"code", nullptr}}}}.dump(),
        "application/json");
}

std::string read_index(const std::string& dir) {
    std::ifstream in(std::filesystem::path(dir) / "index.html", std::ios::binary);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

// One forwarded request. The upstream call runs on its own detached thread and hands the response
// over through this bridge, so the browser receives each chunk as the engine writes it. Whoever
// sees the browser go away sets `cancelled`; the content receiver then returns false, the client
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

void forward(const httplib::Request& req, httplib::Response& res, const EngineSpec& engine,
             const std::atomic<bool>& stopping) {
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
                 method = req.method, target = req.target, headers = std::move(headers),
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
    // browser that leaves meanwhile cancels the engine request.
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
            write_json_error(res, 502, "server_error",
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

} // namespace

FrontendServer::FrontendServer(FrontendSpec spec, EngineSpecProvider engine)
    : spec_(std::move(spec)), engine_(std::move(engine)) {}

FrontendServer::~FrontendServer() {
    stop();
    delete static_cast<httplib::Server*>(server_);
}

bool FrontendServer::bind() {
    if (const std::string problem = frontend_assets_problem(spec_.dir); !problem.empty()) {
        std::lock_guard<std::mutex> lock(reason_mutex_);
        reason_ = problem;
        state_  = FrontendState::AssetsMissing;
        return false;
    }
    auto* svr = new httplib::Server;
    server_   = svr;
#ifdef _WIN32
    // httplib's default SO_REUSEADDR lets a Windows socket bind a port another process already
    // listens on, so a taken port would look free and the two listeners would split requests.
    svr->set_socket_options([](socket_t sock) {
        const BOOL exclusive = TRUE;
        setsockopt(sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char*>(&exclusive),
                   sizeof(exclusive));
    });
#endif
    svr->set_payload_max_length(kMaxRequestBytes);
    svr->set_file_extension_and_mimetype_mapping("mjs", "text/javascript");
    svr->set_file_extension_and_mimetype_mapping("webmanifest", "application/manifest+json");
    svr->set_file_extension_and_mimetype_mapping("wasm", "application/wasm");

    // Runs before the static files, so a rebinding DNS name or another site's page is turned away
    // before anything is served or forwarded.
    svr->set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        const int port = bound_port_.load();
        if (!host_header_allowed(req.get_header_value("Host"), port, kFrontendHost, false)) {
            write_json_error(res, 403, "invalid_request_error", "host not allowed");
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!frontend_origin_allowed(req.get_header_value("Origin"), port)) {
            write_json_error(res, 403, "invalid_request_error", "cross-origin request refused");
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!frontend_method_allowed(req.method, req.path)) {
            write_json_error(res, 403, "invalid_request_error",
                             "engine admin writes are not available through a frontend");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    // Existing files are served first, `/` as index.html.
    if (!svr->set_mount_point("/", spec_.dir)) {
        std::lock_guard<std::mutex> lock(reason_mutex_);
        reason_ = "directory cannot be served";
        state_  = FrontendState::AssetsMissing;
        return false;
    }
    const auto dispatch = [this](const httplib::Request& req, httplib::Response& res) {
        if (!is_engine_api_path(req.path) &&
            is_page_navigation(req.method, req.get_header_value("Accept"))) {
            // History-API fallback: a client-side route reloaded in the browser.
            res.set_header("Cache-Control", "no-cache");
            res.set_content(read_index(spec_.dir), "text/html; charset=utf-8");
            return;
        }
        forward(req, res, engine_(), stopping_);
    };
    svr->Get(".*", dispatch);
    svr->Post(".*", dispatch);
    svr->Put(".*", dispatch);
    svr->Patch(".*", dispatch);
    svr->Delete(".*", dispatch);

    const int port = spec_.port == 0 ? svr->bind_to_any_port(std::string(kFrontendHost))
                                     : (svr->bind_to_port(std::string(kFrontendHost), spec_.port)
                                            ? spec_.port
                                            : -1);
    if (port <= 0) {
        std::lock_guard<std::mutex> lock(reason_mutex_);
        reason_ = "port " + std::to_string(spec_.port) + " is in use";
        state_  = FrontendState::PortInUse;
        return false;
    }
    bound_port_ = port;
    state_      = FrontendState::Serving;
    return true;
}

void FrontendServer::run() {
    if (server_ == nullptr || state_.load() != FrontendState::Serving) { return; }
    static_cast<httplib::Server*>(server_)->listen_after_bind();
    if (state_.load() == FrontendState::Serving) { state_ = FrontendState::Stopped; }
}

void FrontendServer::stop() {
    stopping_ = true;
    if (server_ != nullptr) { static_cast<httplib::Server*>(server_)->stop(); }
}

nlohmann::json FrontendServer::status_json() const {
    const int port = bound_port_.load() != 0 ? bound_port_.load() : spec_.port;
    std::lock_guard<std::mutex> lock(reason_mutex_);
    return {{"name", spec_.name},
            {"url", frontend_url(port)},
            {"port", port},
            {"state", state_name(state_.load())},
            {"reason", reason_}};
}

FrontendHost::FrontendHost(EngineSpecProvider engine) : engine_(std::move(engine)) {}

FrontendHost::~FrontendHost() { stop(); }

void FrontendHost::shut_down(std::vector<Entry>& entries) {
    for (auto& entry : entries) { entry.server->stop(); }
    for (auto& entry : entries) {
        if (entry.thread.joinable()) { entry.thread.join(); }
    }
    entries.clear();
}

void FrontendHost::apply(const std::vector<FrontendSpec>& specs) {
    std::lock_guard<std::mutex> applying(apply_mutex_);
    // Stopping waits for open connections to drain, so it happens outside `mutex_`: the tray and
    // /api/state keep reading the status meanwhile.
    std::vector<Entry> kept;
    std::vector<Entry> retired;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& entry : entries_) {
            const bool wanted = std::any_of(specs.begin(), specs.end(), [&](const FrontendSpec& s) {
                return same_frontend(s, entry.server->spec());
            });
            if (wanted && entry.server->state() == FrontendState::Serving) {
                kept.push_back(std::move(entry));
            } else {
                retired.push_back(std::move(entry));
            }
        }
        entries_.clear();
        // Kept frontends stay visible while the others stop.
        for (auto& entry : kept) { entries_.push_back(std::move(entry)); }
        kept.clear();
    }
    shut_down(retired);

    std::vector<Entry> next;
    next.reserve(specs.size());
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& spec : specs) {
            const auto running = std::find_if(entries_.begin(), entries_.end(), [&](const Entry& e) {
                return e.server && same_frontend(spec, e.server->spec());
            });
            if (running != entries_.end()) {
                next.push_back(std::move(*running));
                continue;
            }
            Entry entry;
            entry.server = std::make_unique<FrontendServer>(spec, engine_);
            if (entry.server->bind()) {
                entry.thread = std::thread([raw = entry.server.get()] { raw->run(); });
            }
            next.push_back(std::move(entry));
        }
        entries_ = std::move(next);
    }
}

void FrontendHost::stop() {
    std::lock_guard<std::mutex> applying(apply_mutex_);
    std::vector<Entry> all;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        all = std::move(entries_);
        entries_.clear();
    }
    shut_down(all);
}

nlohmann::json FrontendHost::status_json() const {
    std::lock_guard<std::mutex> lock(mutex_);
    nlohmann::json out = nlohmann::json::array();
    for (const auto& entry : entries_) { out.push_back(entry.server->status_json()); }
    return out;
}

std::vector<FrontendLink> FrontendHost::links() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<FrontendLink> out;
    out.reserve(entries_.size());
    for (const auto& entry : entries_) {
        const nlohmann::json status = entry.server->status_json();
        out.push_back(FrontendLink{status.value("name", ""), status.value("url", ""),
                                   entry.server->state(), status.value("reason", "")});
    }
    return out;
}

} // namespace ninfer::supervisor
