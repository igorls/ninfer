#include "core/u4z8.h"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template <class F> void rejects(F fn) {
    try { fn(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid U4Z8 input accepted");
}
// Independent FP64 oracle: interpret the represented FP16 scale mathematically,
// then round the exact product on the BF16 exponent/significand grid.
std::uint16_t oracle(unsigned scale, int code) {
    const auto e = scale >> 10, f = scale & 1023;
    const double multiplier = e ? std::ldexp(double(1024 + f), int(e) - 25)
                                  : std::ldexp(double(f), -24);
    const double value = (code - 8) * multiplier;
    if (!value) return std::signbit(value) ? 0x8000 : 0;
    int exponent;
    (void)std::frexp(std::abs(value), &exponent);
    --exponent;
    auto significand = static_cast<unsigned>(std::nearbyint(
        std::ldexp(std::abs(value), 7 - exponent)));
    if (significand == 256) { significand = 128; ++exponent; }
    return static_cast<std::uint16_t>((std::signbit(value) ? 0x8000 : 0) |
        ((exponent + 127) << 7) | (significand - 128));
}
}
int main() {
    try {
        std::array<std::byte, 10> encoded{};
        std::array<std::uint16_t, 16> output{};
        for (unsigned s = 0; s < 0x7c00; ++s) {
            for (unsigned j = 0; j < 8; ++j)
                encoded[j] = std::byte(s ? 2*j | ((2*j+1) << 4) : 0x88);
            encoded[8] = std::byte(s & 255); encoded[9] = std::byte(s >> 8);
            ninfer::decode_u4z8_row(encoded, output);
            for (unsigned j = 0; j < output.size(); ++j)
                require(output[j] == oracle(s, s ? int(j) : 8), "U4Z8 exact FP64 oracle differs");
        }
        for (unsigned s : {0x8000, 0xbc00, 0x7c00, 0x7e00}) {
            encoded[8] = std::byte(s & 255); encoded[9] = std::byte(s >> 8);
            rejects([&] { ninfer::decode_u4z8_row(encoded, output); });
        }
        encoded[8] = encoded[9] = std::byte{}; encoded[0] = std::byte{};
        rejects([&] { ninfer::decode_u4z8_row(encoded, output); });
        rejects([&] { ninfer::decode_u4z8_row(std::span(encoded).first(9), output); });
        std::cout << "U4Z8: every finite nonnegative FP16 scale and all codes match FP64 oracle\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
