#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// Qwen3.8-Flash-Next publishes 262144 as the native window and 1000000 as the
// static YaRN extension (factor 4, original_max_position_embeddings 262144).
// 1000001 is rejected. 1500000 is not a supported pool.
inline constexpr std::uint32_t kNativeContextTokens = 262'144;
inline constexpr std::uint32_t kMaxContextTokens    = 1'000'000;
// Selected-block attention and the QSA attention wrapper accept at most this
// many query tokens in one call. Prefill stays chunked under that limit while
// max_context itself may reach kMaxContextTokens.
inline constexpr std::uint32_t kMaxPrefillCallTokens = 262'144;
inline constexpr std::uint32_t kIndexerBlockTokens   = 4;
inline constexpr std::int32_t kMaxIndexerBlocks =
    static_cast<std::int32_t>(kMaxContextTokens / kIndexerBlockTokens); // 250000

inline constexpr int kRotaryDim   = 64;
inline constexpr int kRotaryPairs = kRotaryDim / 2;
inline constexpr double kRopeTheta = 10'000'000.0;
inline constexpr double kYarnFactor   = 4.0;
inline constexpr double kYarnBetaFast = 32.0;
inline constexpr double kYarnBetaSlow = 1.0;

static_assert(kMaxContextTokens % kIndexerBlockTokens == 0);
static_assert(kMaxIndexerBlocks == 250'000);
static_assert(kRotaryPairs == 32);

[[nodiscard]] inline constexpr bool
flash_next_context_allowed(std::uint32_t max_context) noexcept {
    return max_context >= 1 && max_context <= kMaxContextTokens;
}

// Full indexer envelope. The previous planner clamped this to 65536, which
// covers only the native 262144 tokens. 262145 is 65537 blocks; 1000000 is
// 250000 blocks of 4.
[[nodiscard]] inline constexpr std::uint32_t
flash_next_indexer_blocks(std::uint32_t max_context) noexcept {
    return (max_context + kIndexerBlockTokens - 1U) / kIndexerBlockTokens;
}

[[nodiscard]] inline constexpr bool
flash_next_yarn_enabled(std::uint32_t max_context) noexcept {
    return max_context > kNativeContextTokens;
}

// Host-side NeoX pair. Matches the device store: dims [0, 32) against [32, 64).
struct FlashNextRotatedPair {
    double first  = 0;
    double second = 0;
};

[[nodiscard]] inline FlashNextRotatedPair
flash_next_rotate_pair(double first, double second, std::int32_t position, double inv_freq,
                       double attention_factor) {
    const double angle  = static_cast<double>(position) * inv_freq;
    const double sine   = std::sin(angle) * attention_factor;
    const double cosine = std::cos(angle) * attention_factor;
    return {first * cosine - second * sine, second * cosine + first * sine};
}

[[nodiscard]] inline double flash_next_default_inv_freq(int pair) {
    return std::pow(kRopeTheta, (-2.0 * static_cast<double>(pair)) / static_cast<double>(kRotaryDim));
}

// Hugging Face `_compute_yarn_parameters` with the published Flash-Next fields
// omitted values left at the library defaults: beta_fast 32, beta_slow 1,
// truncate true, and attention_factor = 0.1 * ln(factor) + 1. The blend is a
// property of the frequency index, not of the token position, so it applies at
// every position once YaRN is enabled.
[[nodiscard]] inline double flash_next_yarn_attention_factor() {
    return 0.1 * std::log(kYarnFactor) + 1.0;
}

inline void flash_next_yarn_correction_bounds(double& low, double& high) {
    const auto find_dim = [](double num_rotations) {
        return (static_cast<double>(kRotaryDim) *
                std::log(static_cast<double>(kNativeContextTokens) /
                         (num_rotations * 2.0 * std::acos(-1.0)))) /
               (2.0 * std::log(kRopeTheta));
    };
    low  = std::floor(find_dim(kYarnBetaFast));
    high = std::ceil(find_dim(kYarnBetaSlow));
    low  = std::max(low, 0.0);
    high = std::min(high, static_cast<double>(kRotaryDim - 1));
    if (low == high) { high += 0.001; }
}

[[nodiscard]] inline double flash_next_yarn_inv_freq(int pair) {
    double low  = 0;
    double high = 0;
    flash_next_yarn_correction_bounds(low, high);
    const double linear = (static_cast<double>(pair) - low) / (high - low);
    const double ramp   = std::clamp(linear, 0.0, 1.0);
    const double extrapolation_factor = 1.0 - ramp;
    const double pos_freq =
        std::pow(kRopeTheta, (2.0 * static_cast<double>(pair)) / static_cast<double>(kRotaryDim));
    const double inv_extrapolation = 1.0 / pos_freq;
    const double inv_interpolation = 1.0 / (kYarnFactor * pos_freq);
    return inv_interpolation * (1.0 - extrapolation_factor) +
           inv_extrapolation * extrapolation_factor;
}

// Passed by value into the QSA attention and indexer kernels. yarn == 0 keeps
// the historical device frequency expression and does not scale cos/sin.
struct FlashNextRopeScaling {
    float inv_freq[kRotaryPairs]{};
    float attention_factor = 1.0F;
    int yarn               = 0;
};

static_assert(std::is_trivially_copyable_v<FlashNextRopeScaling>);
static_assert(std::is_standard_layout_v<FlashNextRopeScaling>);

[[nodiscard]] inline FlashNextRopeScaling flash_next_rope_scaling(std::uint32_t max_context) {
    FlashNextRopeScaling scaling{};
    if (!flash_next_yarn_enabled(max_context)) { return scaling; }
    scaling.yarn              = 1;
    scaling.attention_factor  = static_cast<float>(flash_next_yarn_attention_factor());
    for (int pair = 0; pair < kRotaryPairs; ++pair) {
        scaling.inv_freq[pair] = static_cast<float>(flash_next_yarn_inv_freq(pair));
    }
    return scaling;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
