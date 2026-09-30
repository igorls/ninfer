#pragma once

// Exact decode of the stored FP8 row multiplier: BF16 (FP8_E4M3FN_ROW_BF16) or FP32
// (FP8_E4M3FN_ROW_FP32). Both widen to FP32 without rounding.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

__device__ __forceinline__ float fp8_row_scale(const __nv_bfloat16* scale) {
    return __bfloat162float(__ldg(scale));
}

__device__ __forceinline__ float fp8_row_scale(const float* scale) { return __ldg(scale); }

// Two adjacent row multipliers starting at an even row; the base is aligned to twice the word.
__device__ __forceinline__ float2 fp8_row_scale_pair(const __nv_bfloat16* scale) {
    return bf16x2_bits_to_float2(load_ldg<std::uint32_t>(scale));
}

__device__ __forceinline__ float2 fp8_row_scale_pair(const float* scale) {
    return load_ldg<float2>(scale);
}

} // namespace ninfer::ops::detail
