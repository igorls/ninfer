#include "core/wide_multiply.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>

namespace {

constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();

static_assert(ninfer::multiply_wide(kMax, kMax) == ninfer::WideProduct{kMax - 1U, 1U});
static_assert(ninfer::multiply_wide(1ULL << 32U, 1ULL << 32U) == ninfer::WideProduct{1U, 0U});
static_assert(ninfer::multiply_wide(0xFFFFFFFFULL, 0xFFFFFFFFULL) ==
              ninfer::WideProduct{0U, 0xFFFFFFFE00000001ULL});
static_assert(ninfer::saturating_multiply(1ULL << 32U, 1ULL << 32U) == kMax);
static_assert(ninfer::saturating_multiply(1ULL << 31U, 1ULL << 32U) == 1ULL << 63U);
static_assert(ninfer::saturating_add(kMax - 1U, 1U) == kMax);
static_assert(ninfer::saturating_add(kMax, 1U) == kMax);

// Independent oracle: schoolbook multiplication over 16-bit limbs with explicit carries.
ninfer::WideProduct oracle(std::uint64_t a, std::uint64_t b) {
    std::array<std::uint32_t, 8> limbs{};
    for (unsigned i = 0; i < 4; ++i) {
        std::uint32_t carry = 0;
        const std::uint32_t x = static_cast<std::uint32_t>((a >> (16U * i)) & 0xFFFFU);
        for (unsigned j = 0; j < 4; ++j) {
            const std::uint32_t y = static_cast<std::uint32_t>((b >> (16U * j)) & 0xFFFFU);
            const std::uint64_t sum =
                static_cast<std::uint64_t>(x) * y + limbs[i + j] + carry;
            limbs[i + j] = static_cast<std::uint32_t>(sum & 0xFFFFU);
            carry        = static_cast<std::uint32_t>(sum >> 16U);
        }
        limbs[i + 4] += carry;
    }
    ninfer::WideProduct out;
    for (unsigned k = 0; k < 4; ++k) {
        out.low |= static_cast<std::uint64_t>(limbs[k]) << (16U * k);
        out.high |= static_cast<std::uint64_t>(limbs[k + 4]) << (16U * k);
    }
    return out;
}

} // namespace

int main() {
    std::mt19937_64 random(0x5eedU);
    const std::array<std::uint64_t, 8> edges{0U, 1U, 0xFFFFFFFFULL, 1ULL << 32U,
                                             (1ULL << 32U) + 1U, 1ULL << 63U, kMax - 1U, kMax};
    int failures = 0;
    const auto check = [&](std::uint64_t a, std::uint64_t b) {
        if (ninfer::multiply_wide(a, b) != oracle(a, b)) {
            std::cerr << "multiply_wide mismatch for " << a << " * " << b << '\n';
            ++failures;
        }
    };
    for (const auto a : edges) {
        for (const auto b : edges) { check(a, b); }
    }
    for (int i = 0; i < 100000; ++i) {
        const std::uint64_t a = random() >> (random() % 64U);
        const std::uint64_t b = random() >> (random() % 64U);
        check(a, b);
    }
    return failures == 0 ? 0 : 1;
}
