#pragma once

#include "ops/common/text_mrope_r64.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

// Packed gated-query projection profile: 24 query heads stored as [query_256, gate_256], then
// two 256-wide key heads and two 256-wide value heads (13312 rows per token).
namespace gated_d256 {
inline constexpr int kHeadDim      = 256;
inline constexpr int kQueryHeads   = 24;
inline constexpr int kKeyHeads     = 2;
inline constexpr int kRows         = 13'312;
inline constexpr int kKeyOffset    = 12'288;
inline constexpr int kValueOffset  = 12'800;
inline constexpr int kWarps        = kQueryHeads + kKeyHeads;
inline constexpr int kThreads      = kWarps * 32;
inline constexpr float kEpsilon    = 1.0e-6F;
} // namespace gated_d256

// One warp owns one head. Lane l holds dimensions [8l,8l+8), so rotary pair i (dimensions i and
// i+32, i<32) lives in lanes l and l^4 of the first eight lanes. Normalization runs before the
// CTA's rotation table is ready, so the global loads overlap the phase computation.
__device__ __forceinline__ void gated_d256_normalize(const __nv_bfloat16* __restrict__ source,
                                                     const __nv_bfloat16* __restrict__ weight,
                                                     int lane, float (&values)[8]) {
    using namespace gated_d256;
    const uint4 raw = *reinterpret_cast<const uint4*>(source + lane * 8);
    const uint4 w   = *reinterpret_cast<const uint4*>(weight + lane * 8);
    const auto* x2  = reinterpret_cast<const __nv_bfloat162*>(&raw);
    const auto* w2  = reinterpret_cast<const __nv_bfloat162*>(&w);
    float sum       = 0.0F;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 v    = __bfloat1622float2(x2[i]);
        values[2 * i]     = v.x;
        values[2 * i + 1] = v.y;
        sum               = fmaf(v.x, v.x, sum);
        sum               = fmaf(v.y, v.y, sum);
    }
    sum                 = warp_sum(sum);
    const float inverse = rsqrtf(sum * (1.0F / kHeadDim) + kEpsilon);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 g    = __bfloat1622float2(w2[i]);
        values[2 * i]     = values[2 * i] * inverse * (1.0F + g.x);
        values[2 * i + 1] = values[2 * i + 1] * inverse * (1.0F + g.y);
    }
}

__device__ __forceinline__ void gated_d256_rotate_store(float (&values)[8],
                                                        const float2* __restrict__ rotation,
                                                        __nv_bfloat16* __restrict__ destination,
                                                        int lane) {
    // Every lane participates in the shuffles; only lanes 0..7 hold rotary dimensions.
    float partner[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { partner[i] = __shfl_xor_sync(kFullWarpMask, values[i], 4); }
    if (lane < 8) {
        const bool first = lane < 4;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int pair  = (lane & 3) * 8 + i;
            const float2 sc = rotation[pair];
            values[i]       = first ? values[i] * sc.y - partner[i] * sc.x
                                    : values[i] * sc.y + partner[i] * sc.x;
        }
    }
    uint4 packed;
    auto* out2 = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
    for (int i = 0; i < 4; ++i) { out2[i] = __floats2bfloat162_rn(values[2 * i], values[2 * i + 1]); }
    *reinterpret_cast<uint4*>(destination + lane * 8) = packed;
}

__global__ __launch_bounds__(gated_d256::kThreads) void rmsnorm_rope_gated_d256_kernel(
    const __nv_bfloat16* __restrict__ projected, const std::int32_t* __restrict__ positions,
    const __nv_bfloat16* __restrict__ q_norm, const __nv_bfloat16* __restrict__ k_norm,
    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ gate, __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v, int tokens) {
    using namespace gated_d256;
    __shared__ float2 rotation[kTextMropeR64Pairs];
    const std::int64_t token = blockIdx.x;
    const int warp           = static_cast<int>(threadIdx.x) >> 5;
    const int lane           = static_cast<int>(threadIdx.x) & 31;
    const auto* row          = projected + token * kRows;
    const std::int32_t axis_position =
        threadIdx.x < kTextMropeR64Pairs
            ? text_mrope_axis_position(positions, tokens, token, static_cast<int>(threadIdx.x))
            : 0;
    const bool query         = warp < kQueryHeads;
    const int head           = query ? warp : warp - kQueryHeads;
    // Query warps copy their gate rows; key warps copy their value rows.
    const __nv_bfloat16* source = query ? row + head * 2 * kHeadDim : row + kKeyOffset + head * kHeadDim;
    const __nv_bfloat16* copied = query ? source + kHeadDim : row + kValueOffset + head * kHeadDim;
    __nv_bfloat16* destination  = query ? q + (token * kQueryHeads + head) * kHeadDim
                                        : k + (token * kKeyHeads + head) * kHeadDim;
    __nv_bfloat16* copy_target  = query ? gate + (token * kQueryHeads + head) * kHeadDim
                                        : v + (token * kKeyHeads + head) * kHeadDim;
    *reinterpret_cast<uint4*>(copy_target + lane * 8) =
        *reinterpret_cast<const uint4*>(copied + lane * 8);
    float values[8];
    gated_d256_normalize(source, query ? q_norm : k_norm, lane, values);
    if (threadIdx.x < kTextMropeR64Pairs) {
        const int pair = static_cast<int>(threadIdx.x);
        rotation[pair] = text_mrope_r64_sincos(pair, axis_position);
    }
    __syncthreads(); // the token's rotations are complete
    gated_d256_rotate_store(values, rotation, destination, lane);
}

} // namespace ninfer::ops
