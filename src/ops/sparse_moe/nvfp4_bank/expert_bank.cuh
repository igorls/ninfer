#pragma once

// Device view of one NVFP4 expert bank. Every kernel reaches an expert's code plane, scale plane
// and divisor only through nvfp4_expert(); nothing else knows how experts are placed. A future
// expert residency scheme (for example a VRAM cache over host-resident banks) replaces this one
// mapping instead of address arithmetic scattered through the kernels.

#include "ninfer/ops/sparse_moe.h"

#include <cstdint>

namespace ninfer::ops::detail {

struct Nvfp4ExpertBankDevice {
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    const float* divisors;
    std::uint64_t code_stride;
    std::uint64_t scale_stride;
};

struct Nvfp4ExpertPlanes {
    const std::uint8_t* codes;  // [N,K/2] row-major codes of this expert
    const std::uint8_t* scales; // this expert's M128x4-swizzled K16 scale plane
    float inverse_divisor;      // 1 / stored FP32 weight divisor
};

inline Nvfp4ExpertBankDevice nvfp4_expert_bank_device(const Nvfp4ExpertBankWeight& bank) {
    return {reinterpret_cast<const std::uint8_t*>(bank.codes),
            reinterpret_cast<const std::uint8_t*>(bank.scales), bank.weight_scale_divisors,
            bank.code_bytes_per_expert, bank.scale_bytes_per_expert};
}

__device__ __forceinline__ Nvfp4ExpertPlanes nvfp4_expert(const Nvfp4ExpertBankDevice& bank,
                                                          int expert) {
    const auto index = static_cast<std::uint64_t>(expert);
    // IEEE division: the reciprocal is the exact-rounded inverse of the stored divisor.
    return {bank.codes + index * bank.code_stride, bank.scales + index * bank.scale_stride,
            1.0F / bank.divisors[expert]};
}

// Byte offset of the scale word holding groups [4*tile, 4*tile+4) of `row` inside one expert's
// scale plane with K/64 scale tiles per row (storage-layouts.md, M128x4 swizzle).
__device__ __forceinline__ std::int64_t nvfp4_expert_scale_word(int row, int tile,
                                                                int tiles_per_row) {
    return static_cast<std::int64_t>((row >> 7) * tiles_per_row + tile) * 512 + (row & 31) * 16 +
           ((row & 127) >> 5) * 4;
}

} // namespace ninfer::ops::detail
