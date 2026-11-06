#include "runtime/engine/context_cache/placement_policy.h"
#include "core/wide_multiply.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::runtime {
namespace {

using WideTokens = WideProduct;

WideTokens add_tokens(std::uint64_t a, std::uint64_t b) {
    const auto low = a + b;
    return {low < a ? 1ULL : 0ULL, low};
}

// Peaks contain at most two uint64 token counts, so these small factors fit 128 bits.
WideTokens scaled_peak(WideTokens value, std::uint64_t factor) {
    auto result = multiply_wide(value.low, factor);
    result.high += value.high * factor;
    return result;
}

struct Choice {
    std::size_t candidate;
    const PlacementLoad* load;
    WideTokens peak;
    bool balanced = false;
};

} // namespace

std::vector<std::size_t> order_placements(std::span<const PlacementLoad> loads,
                                          std::span<const PlacementCandidate> candidates,
                                          std::uint64_t request_tokens, PlacementId first_tie) {
    WideTokens current_peak{};
    for (auto it = loads.begin(); it != loads.end(); ++it) {
        if (std::any_of(loads.begin(), it, [&](const auto& previous) {
                return previous.placement == it->placement;
            })) {
            throw std::invalid_argument("placement loads contain a duplicate placement");
        }
        current_peak = std::max(current_peak, WideTokens{0, it->tokens});
    }

    std::vector<Choice> choices;
    choices.reserve(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const auto& candidate = candidates[i];
        const auto load       = std::find_if(loads.begin(), loads.end(), [&](const auto& value) {
            return value.placement == candidate.placement;
        });
        if (load == loads.end()) {
            throw std::invalid_argument("placement candidate has no load entry");
        }
        if (!candidate.feasible) continue;
        choices.push_back(
            {i, &*load, std::max(current_peak, add_tokens(load->tokens, request_tokens))});
    }
    if (choices.empty()) return {};

    const auto minimum =
        std::min_element(choices.begin(), choices.end(), [](const auto& a, const auto& b) {
            return a.peak < b.peak;
        })->peak;
    for (auto& choice : choices)
        choice.balanced = scaled_peak(choice.peak, 2) <= scaled_peak(minimum, 3);
    std::stable_sort(choices.begin(), choices.end(), [&](const Choice& a, const Choice& b) {
        if (a.balanced != b.balanced) return a.balanced;
        if (!a.balanced && a.peak != b.peak) return a.peak < b.peak;
        const auto a_prefix = candidates[a.candidate].local_prefix_tokens;
        const auto b_prefix = candidates[b.candidate].local_prefix_tokens;
        if (a_prefix != b_prefix) return a_prefix > b_prefix;
        if (a.load->tokens != b.load->tokens) return a.load->tokens < b.load->tokens;
        if (a.load->requests != b.load->requests) return a.load->requests < b.load->requests;
        const auto a_rotation = std::uint32_t(a.load->placement.value - first_tie.value);
        const auto b_rotation = std::uint32_t(b.load->placement.value - first_tie.value);
        return a_rotation < b_rotation;
    });

    std::vector<std::size_t> result;
    result.reserve(choices.size());
    for (const auto& choice : choices) result.push_back(choice.candidate);
    return result;
}

} // namespace ninfer::runtime
