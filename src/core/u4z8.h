#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer {
// Low-nibble-first U4 codes followed by one little-endian binary16 scale per 16
// values. Decode (code-8)*scale to BF16 with round-to-nearest, ties-to-even.
void decode_u4z8_row(std::span<const std::byte> encoded, std::span<std::uint16_t> output);
}
