#include "core/u4z8.h"

#include <bit>
#include <cmath>
#include <stdexcept>

namespace ninfer {
void decode_u4z8_row(std::span<const std::byte> encoded, std::span<std::uint16_t> output) {
    const auto k = output.size();
    if (!k || k % 16 || encoded.size() != k / 2 + k / 16 * 2)
        throw std::invalid_argument("invalid U4Z8 row geometry");
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(encoded.data());
    for (std::size_t group = 0; group < k / 16; ++group) {
        const auto s = std::uint16_t(bytes[k / 2 + 2 * group]) |
                       (std::uint16_t(bytes[k / 2 + 2 * group + 1]) << 8);
        if ((s & 0x8000) || (s & 0x7c00) == 0x7c00)
            throw std::invalid_argument("U4Z8 requires nonnegative finite FP16 scales");
        const unsigned exponent = (s >> 10) & 31, fraction = s & 1023;
        const float scale = exponent ? std::bit_cast<float>(
            std::uint32_t((exponent + 112) << 23) | (fraction << 13))
                                     : std::ldexp(float(fraction), -24);
        for (unsigned j = 0; j < 16; ++j) {
            const auto code = (bytes[group * 8 + j / 2] >> (4 * (j & 1))) & 15;
            if (!s && code != 8)
                throw std::invalid_argument("zero U4Z8 scale requires zero-point codes");
            const auto bits = std::bit_cast<std::uint32_t>(float(int(code) - 8) * scale);
            output[group * 16 + j] = static_cast<std::uint16_t>(
                (bits + 0x7fffU + ((bits >> 16) & 1)) >> 16);
        }
    }
}
}
