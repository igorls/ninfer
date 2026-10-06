#include "models/qwen4_exp/program/planning/graph_profiles.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::models::qwen4_exp::detail {
namespace {

// Resource tiers bound the indexer's inactive scoring CTAs; they are not topology boundaries
// except where the identity route ends.
constexpr std::array<std::uint32_t, 8> kVisibleTiers{4096,  8192,   16384,  32768,
                                                     65536, 131072, 262144, 524288};

} // namespace

std::int32_t indexer_envelope_blocks(std::uint32_t max_frontier, std::uint32_t indexer_block) {
    if (indexer_block == 0) { throw std::invalid_argument("indexer block width must be positive"); }
    return static_cast<std::int32_t>((static_cast<std::uint64_t>(max_frontier) + 1U) /
                                     indexer_block);
}

std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity,
                                                           std::uint32_t indexer_block,
                                                           std::uint32_t indexer_selected_blocks) {
    if (capacity == 0 || indexer_block == 0 || indexer_selected_blocks == 0) {
        throw std::invalid_argument("invalid ordinary graph profile dimensions");
    }
    const std::uint32_t max_frontier = capacity - 1U;
    // The identity route covers every frontier whose complete blocks fit the selection budget.
    const std::uint64_t identity_end64 =
        static_cast<std::uint64_t>(indexer_selected_blocks + 1U) * indexer_block - 2U;
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    const std::uint32_t identity_end =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(identity_end64, max_frontier));
    out.push_back({begin, identity_end, 0});
    if (identity_end == max_frontier) { return out; }
    begin = identity_end + 1U;
    for (const std::uint32_t visible : kVisibleTiers) {
        const std::uint32_t end = std::min(visible - 1U, max_frontier);
        if (end < begin) { continue; }
        out.push_back({begin, end, 1});
        if (end == max_frontier) { return out; }
        begin = end + 1U;
    }
    out.push_back({begin, max_frontier, 1});
    return out;
}

} // namespace ninfer::models::qwen4_exp::detail
