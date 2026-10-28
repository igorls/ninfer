#include "ops/gated_residual/mix/select.h"
#include "ops/gated_residual/mix/workspace.h"
#include <algorithm>
#include <array>
#include <limits>
#include <span>

namespace ninfer::ops::detail::hc::mix {
namespace {
struct Route {
    int first, last;
    Plan plan;
};

constexpr int kLast = std::numeric_limits<int>::max();

// Wide prefill keeps four K partitions across window sizes. Changing this reduction
// order at the former wave boundaries changed the observable BF16 low-rank state.
// Decode and target verification share one low-rank reduction and up schedule.
constexpr std::array block_routes{
    Route{1, 48, {Down::Simt, Up::Folded16x16}},
    Route{49, 64, {Down::Split8x32, Up::Folded32x32}},
    Route{65, 96, {Down::Split8x32, Up::TmaFolded32x64}},
    Route{97, 128, {Down::TmaSplit4, Up::TmaFolded32x64}},
    Route{129, 160, {Down::Split8x32, Up::TmaFolded32x64}},
    Route{161, 192, {Down::TmaSplit4, Up::TmaFolded16x64}},
    Route{193, 224, {Down::Split8x32, Up::TmaFolded32x64}},
    Route{225, 256, {Down::TmaSplit2, Up::TmaFolded16x64}},
    Route{257, 320, {Down::Tma96Split4, Up::TmaFolded16x64}},
    Route{321, 448, {Down::TmaSplit4, Up::TmaFolded16x64}},
    Route{449, kLast, {Down::Tma96Split4, Up::TmaFolded16x64}},
};
constexpr std::array final_routes{
    Route{1, 48, {Down::Simt, Up::Folded16x16}},
    Route{49, 64, {Down::Split8x32, Up::Folded32x32}},
    Route{65, 128, {Down::Split8x32, Up::TmaFolded32x64}},
    Route{129, 192, {Down::Tma96Split8, Up::TmaFolded16x64}},
    Route{193, 256, {Down::TmaSplit4, Up::TmaFolded32x64}},
    Route{257, 384, {Down::Tma96Split4, Up::TmaFolded16x64}},
    Route{385, 512, {Down::TmaSplit4, Up::TmaFolded16x64}},
    Route{513, kLast, {Down::Tma96Split4, Up::TmaFolded16x64}},
};

constexpr bool closed(std::span<const Route> routes) {
    std::int64_t next = 1;
    for (const auto& route : routes) {
        if (route.first != next || route.last < route.first) return false;
        next = std::int64_t(route.last) + 1;
    }
    return next == std::int64_t(kLast) + 1;
}

static_assert(closed(block_routes) && closed(final_routes));

std::span<const Route> routes(Kind kind) {
    return kind == Kind::Block ? std::span<const Route>{block_routes}
                               : std::span<const Route>{final_routes};
}
} // namespace

Plan select_plan(Kind kind, std::int32_t columns) {
    validate_interval(columns, columns);
    for (const auto& route : routes(kind))
        if (columns >= route.first && columns <= route.last) return route.plan;
    throw std::logic_error("hc_mix: missing production route");
}

std::size_t capacity(Weights weights, std::int32_t first, std::int32_t last) {
    describe_weights(weights);
    validate_interval(first, last);
    std::size_t peak = 0;
    // Each route's allocation grows monotonically; the complete interval need not.
    for (const auto& route : routes(weights.kind)) {
        if (route.last < first || route.first > last) continue;
        peak = std::max(peak, layout(weights.kind, route.plan, std::min(last, route.last)).bytes);
    }
    return peak;
}
} // namespace ninfer::ops::detail::hc::mix
