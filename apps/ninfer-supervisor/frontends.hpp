#pragma once

#include "logic.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace ninfer::supervisor {

// A static web frontend, for example llama.cpp's web UI, served on its own loopback port. The
// port is the frontend's origin: a UI that calls root-relative API paths works unchanged, and
// each UI keeps its browser storage to itself. The same port forwards the engine's API, so the
// page needs neither CORS nor the engine's API key.
struct FrontendSpec {
    std::string name;
    std::string dir; // built static assets, index.html at the root
    int port = 0;
};

// Loopback only: the proxy adds the engine's API key, so a frontend reachable from another
// computer would hand out keyless access to the engine.
inline constexpr std::string_view kFrontendHost = "127.0.0.1";

inline std::string frontend_url(int port) {
    return "http://" + std::string(kFrontendHost) + ":" + std::to_string(port) + "/";
}

// Reads the top-level "frontends" array. Entries are kept even when their directory does not
// exist yet: a moved folder is reported in the status and never stops the supervisor.
inline std::vector<FrontendSpec> parse_frontends(const nlohmann::json& value) {
    std::vector<FrontendSpec> out;
    if (!value.is_array()) { throw std::invalid_argument("frontends must be an array"); }
    for (const auto& item : value) {
        if (!item.is_object()) { throw std::invalid_argument("each frontend must be an object"); }
        FrontendSpec spec;
        spec.name = item.value("name", "");
        spec.dir  = item.value("dir", "");
        const auto port = item.find("port");
        if (port == item.end() || !port->is_number_integer()) {
            throw std::invalid_argument("frontend \"" + spec.name + "\" needs an integer port");
        }
        spec.port = port->get<int>();
        out.push_back(std::move(spec));
    }
    return out;
}

inline nlohmann::json frontends_to_json(const std::vector<FrontendSpec>& frontends) {
    nlohmann::json out = nlohmann::json::array();
    for (const auto& f : frontends) {
        out.push_back({{"name", f.name}, {"dir", f.dir}, {"port", f.port}});
    }
    return out;
}

// A configuration that can never serve is a load error: no name, no directory, a port outside
// 1..65535, or a port the supervisor, the engine or another frontend already uses.
inline void validate_frontends(const std::vector<FrontendSpec>& frontends, int supervisor_port,
                               int engine_port) {
    if (frontends.size() > kMaxFrontends) {
        throw std::invalid_argument("at most " + std::to_string(kMaxFrontends) +
                                    " frontends are supported");
    }
    std::vector<int> ports;
    for (const auto& f : frontends) {
        if (f.name.empty()) { throw std::invalid_argument("every frontend needs a name"); }
        if (f.dir.empty()) {
            throw std::invalid_argument("frontend \"" + f.name + "\" needs a dir");
        }
        if (f.port < 1 || f.port > 65535) {
            throw std::invalid_argument("frontend \"" + f.name + "\" port must be in 1..65535");
        }
        if (f.port == supervisor_port || f.port == engine_port) {
            throw std::invalid_argument("frontend \"" + f.name +
                                        "\" port is already the supervisor or engine port");
        }
        if (std::find(ports.begin(), ports.end(), f.port) != ports.end()) {
            throw std::invalid_argument("frontend \"" + f.name + "\" port is used twice");
        }
        ports.push_back(f.port);
    }
}

// The dashboard's POST /api/frontends body, {"frontends": [{name, dir, port}, ...]}, checked like
// the configuration file. `error` is empty when `frontends` is valid.
struct FrontendsRequest {
    std::vector<FrontendSpec> frontends;
    std::string error;
};

inline FrontendsRequest parse_frontends_request(const nlohmann::json& body, int supervisor_port,
                                                int engine_port) {
    FrontendsRequest out;
    try {
        if (!body.is_object() || !body.contains("frontends")) {
            throw std::invalid_argument("the request needs a frontends list");
        }
        out.frontends = parse_frontends(body.at("frontends"));
        validate_frontends(out.frontends, supervisor_port, engine_port);
    } catch (const std::exception& ex) {
        out.frontends.clear();
        out.error = ex.what();
    }
    return out;
}

inline bool same_frontend(const FrontendSpec& a, const FrontendSpec& b) {
    return a.name == b.name && a.dir == b.dir && a.port == b.port;
}

// Empty when the directory can serve a page, otherwise the reason it cannot.
inline std::string frontend_assets_problem(const std::string& dir) {
    std::error_code ec;
    const std::filesystem::path root(dir);
    if (!std::filesystem::is_directory(root, ec) || ec) { return "directory not found"; }
    if (!std::filesystem::is_regular_file(root / "index.html", ec) || ec) {
        return "no index.html in the directory";
    }
    return {};
}

inline bool path_has_prefix(std::string_view path, std::string_view prefix) {
    return path == prefix ||
           (path.size() > prefix.size() && path.substr(0, prefix.size()) == prefix &&
            path[prefix.size()] == '/');
}

// Routes the engine serves. They are always forwarded and never answered with index.html, so an
// unknown API path keeps the engine's JSON error.
inline bool is_engine_api_path(std::string_view path) {
    for (const std::string_view prefix : {"/v1", "/props", "/health", "/systemone", "/admin"}) {
        if (path_has_prefix(path, prefix)) { return true; }
    }
    return false;
}

// A browser loading a document. Only these fall back to index.html; a script, style or fetch for
// a path that is neither a file nor an engine route goes to the engine and gets its 404. That is
// also the answer for llama.cpp server routes the engine does not have (/slots, /tools, /models,
// /cors-proxy), which the UI handles. The supervisor never implements /cors-proxy: it would fetch
// arbitrary URLs on the page's behalf.
inline bool is_page_navigation(std::string_view method, std::string_view accept) {
    return (method == "GET" || method == "HEAD") && accept.find("text/html") != std::string_view::npos;
}

// The engine's admin routes are reachable for reading only; POST /admin/quiesce stays on the
// engine's own port.
inline bool frontend_method_allowed(std::string_view method, std::string_view path) {
    if (path_has_prefix(path, "/admin")) { return method == "GET" || method == "HEAD"; }
    return true;
}

// Browsers send Origin on cross-origin requests and on same-origin writes. A request from
// another site's page must not reach the engine with the key this proxy adds. A missing Origin is
// a client on this computer that is not a browser page.
inline bool frontend_origin_allowed(std::string_view origin, int port) {
    if (origin.empty()) { return true; }
    const std::string suffix = ":" + std::to_string(port);
    for (const std::string_view host : {"127.0.0.1", "localhost", "[::1]"}) {
        if (origin == "http://" + std::string(host) + suffix) { return true; }
    }
    return false;
}

inline std::string lower_ascii(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

// Hop-by-hop and framing headers belong to each connection. The proxy also drops the page's own
// credentials and context (cookies, Origin, Referer) and compression negotiation.
inline bool forward_request_header(std::string_view name) {
    const std::string n = lower_ascii(name);
    for (const std::string_view drop :
         {"host", "connection", "keep-alive", "proxy-connection", "proxy-authorization", "te",
          "trailer", "transfer-encoding", "upgrade", "content-length", "accept-encoding",
          "cookie", "origin", "referer", "remote_addr", "remote_port", "local_addr",
          "local_port"}) {
        if (n == drop) { return false; }
    }
    return true;
}

// The page and the proxy share one origin, so the engine's CORS headers (present when it runs
// with --cors) would only widen who may read a response.
inline bool forward_response_header(std::string_view name) {
    const std::string n = lower_ascii(name);
    if (n.rfind("access-control-", 0) == 0) { return false; }
    for (const std::string_view drop : {"connection", "keep-alive", "transfer-encoding",
                                        "content-length", "content-type", "set-cookie"}) {
        if (n == drop) { return false; }
    }
    return true;
}

// A key the supervisor holds replaces the page's credentials, so the UI needs none. Without one,
// whatever the page sends passes through, for an engine whose key the supervisor does not read.
inline bool frontend_replaces_credentials(std::string_view name, bool supervisor_has_key) {
    if (!supervisor_has_key) { return false; }
    const std::string n = lower_ascii(name);
    return n == "authorization" || n == "x-api-key";
}

} // namespace ninfer::supervisor
