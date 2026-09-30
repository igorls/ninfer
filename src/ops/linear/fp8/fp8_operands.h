#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops::detail {
// Scale is the stored row-multiplier word: BF16 for FP8_E4M3FN_ROW_BF16 and FP32 for
// FP8_E4M3FN_ROW_FP32. Kernels decode it exactly to FP32 before the row scaling.
template <class Scale>
struct Fp8A16OperandsT {
    const __nv_bfloat16* x;
    const std::uint8_t* codes;
    const Scale* scales;
    int rows, k, tokens;
};

template <class Scale>
struct Fp8A8OperandsT {
    const std::uint8_t* x;
    const float* x_scales;
    const std::uint8_t* codes;
    const Scale* scales;
    int rows, k, tokens;
};

using Fp8A16Operands = Fp8A16OperandsT<__nv_bfloat16>;
using Fp8A8Operands  = Fp8A8OperandsT<__nv_bfloat16>;

template <class Scale = __nv_bfloat16>
inline Fp8A16OperandsT<Scale> fp8_a16_operands(const Tensor& x, const Weight& w) {
    return {static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const Scale*>(w.scales),
            w.n,
            w.k,
            x.ne[1]};
}

template <class Scale = __nv_bfloat16>
inline Fp8A8OperandsT<Scale> fp8_a8_operands(const Weight& w, Fp8A8Workspace x, int tokens) {
    return {x.codes,
            x.scales,
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const Scale*>(w.scales),
            w.n,
            w.k,
            tokens};
}

// Private template admission is independent of the public finite shape registry.
template <class Schedule, class Operands>
inline void validate_fp8_operands(const Operands& p) {
    const auto aligned = [](const void* v, int alignment) {
        return v && reinterpret_cast<std::uintptr_t>(v) % alignment == 0;
    };
    if (p.rows <= 0 || p.k <= 0 || p.tokens <= 0 || p.k % 32 || !aligned(p.x, 16) ||
        !aligned(p.codes, 16) || !aligned(p.scales, static_cast<int>(sizeof(*p.scales))))
        throw std::invalid_argument("FP8 templates require positive N/K/T and aligned operands");
    if constexpr (requires { p.x_scales; }) {
        if (!aligned(p.x_scales, 4) || !aligned(p.scales, static_cast<int>(2 * sizeof(*p.scales))))
            throw std::invalid_argument("FP8 A8 requires row-scale pairs aligned to two words");
    }
    if constexpr (Schedule::kStaticK > 0) {
        if (p.k != Schedule::kStaticK)
            throw std::invalid_argument("FP8 template static K does not match operands");
    }
}
} // namespace ninfer::ops::detail
