#pragma once

#include "config.hpp"

#include <atomic>
#include <functional>
#include <string>

namespace httplib {
class Server;
struct Request;
struct Response;
} // namespace httplib

namespace ninfer::supervisor {

// The engine a proxied request goes to, read for every request so a key file or port changed
// through the dashboard applies without restarting the supervisor.
using EngineSpecProvider = std::function<EngineSpec()>;

// An OpenAI-shaped error body, the shape the engine itself answers with.
void write_engine_error(httplib::Response& res, int status, const std::string& type,
                        const std::string& message);

// Forwards one request to the engine at `upstream_target` (path and query). The supervisor's key,
// when it holds one, replaces the caller's credentials; the answer streams back chunk by chunk as
// the engine writes it, and a caller that leaves, or `stopping`, cancels the engine request.
void forward_to_engine(const httplib::Request& req, httplib::Response& res,
                       const EngineSpec& engine, const std::atomic<bool>& stopping,
                       const std::string& upstream_target);

// The dashboard's API test bench: /api/engine/<engine route> forwards to that engine route.
// Every method needs a local caller and the supervisor control header, like the engine controls,
// so another site's page cannot reach the engine through the bench. Unlike a frontend, the bench
// may call the engine's admin writes; the page asks before sending them.
void add_engine_bench_routes(httplib::Server& server, EngineSpecProvider engine,
                             const std::atomic<bool>& stopping,
                             std::function<bool(const std::string& remote)> local_peer);

} // namespace ninfer::supervisor
