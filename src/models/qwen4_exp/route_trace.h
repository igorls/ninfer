#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::qwen4_exp::execution {

struct RouteTraceColumn {
    std::uint32_t lane = 0;
    std::uint32_t position = 0;
    std::uint32_t source_column = 0;
};

// Owned Host data carried by a completed execution; no rank-local view crosses retirement.
struct RouteTraceRecord {
    ExpertRouteKind kind = ExpertRouteKind::Decode;
    std::uint16_t first_layer = 0;
    std::uint16_t layer_count = 0;
    std::vector<RouteTraceColumn> columns;
    std::vector<std::uint16_t> experts; // [column][layer][rank]
};

} // namespace ninfer::models::qwen4_exp::execution
