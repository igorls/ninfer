#pragma once

#include <compare>
#include <cstdint>
#include <limits>

namespace ninfer {

// Exact unsigned 64x64 -> 128-bit product, portable to MSVC (which has no unsigned __int128).
struct WideProduct {
    std::uint64_t high = 0;
    std::uint64_t low  = 0;

    friend constexpr auto operator<=>(const WideProduct&, const WideProduct&) noexcept = default;
};

[[nodiscard]] constexpr WideProduct multiply_wide(std::uint64_t a, std::uint64_t b) noexcept {
    constexpr std::uint64_t kMask = 0xFFFFFFFFULL;
    const std::uint64_t a_lo = a & kMask;
    const std::uint64_t a_hi = a >> 32U;
    const std::uint64_t b_lo = b & kMask;
    const std::uint64_t b_hi = b >> 32U;
    const std::uint64_t ll   = a_lo * b_lo;
    const std::uint64_t lh   = a_lo * b_hi;
    const std::uint64_t hl   = a_hi * b_lo;
    const std::uint64_t hh   = a_hi * b_hi;
    const std::uint64_t mid  = (ll >> 32U) + (lh & kMask) + (hl & kMask);
    return {hh + (lh >> 32U) + (hl >> 32U) + (mid >> 32U), (mid << 32U) | (ll & kMask)};
}

[[nodiscard]] constexpr std::uint64_t saturating_multiply(std::uint64_t a,
                                                          std::uint64_t b) noexcept {
    const WideProduct product = multiply_wide(a, b);
    return product.high != 0 ? std::numeric_limits<std::uint64_t>::max() : product.low;
}

[[nodiscard]] constexpr std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) noexcept {
    return b > std::numeric_limits<std::uint64_t>::max() - a
               ? std::numeric_limits<std::uint64_t>::max()
               : a + b;
}

}  // namespace ninfer
