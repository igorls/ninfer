#include "frontend_server.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
#include <httplib.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>

namespace ninfer::supervisor {
namespace {

// The engine's default request limit, so an image upload that the engine accepts also passes the
// proxy.
constexpr std::size_t kMaxRequestBytes = 384ULL << 20;
const char* state_name(FrontendState state) {
    switch (state) {
    case FrontendState::Serving: return "serving";
    case FrontendState::AssetsMissing: return "assets_missing";
    case FrontendState::PortInUse: return "port_in_use";
    case FrontendState::Stopped: break;
    }
    return "stopped";
}

std::string read_index(const std::string& dir) {
    std::ifstream in(std::filesystem::path(dir) / "index.html", std::ios::binary);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
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
            write_engine_error(res, 403, "invalid_request_error", "host not allowed");
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!frontend_origin_allowed(req.get_header_value("Origin"), port)) {
            write_engine_error(res, 403, "invalid_request_error", "cross-origin request refused");
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!frontend_method_allowed(req.method, req.path)) {
            write_engine_error(res, 403, "invalid_request_error",
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
        forward_to_engine(req, res, engine_(), stopping_, req.target);
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
