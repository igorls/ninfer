#pragma once

#include "ops/common/text_mrope_r64.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kQsaHeadDim       = 128;
inline constexpr int kQsaQueryHeads    = 4;
inline constexpr int kQsaProjection    = 640;
inline constexpr int kQsaRawKeyOffset  = 512;
inline constexpr int kQsaBlocksPerPage = 16;
inline constexpr int kQsaMaxSelected   = 512;
inline constexpr float kQsaEpsilon     = 1.0e-6F;
inline constexpr float kQsaScoreScale  = 0.08838834764831845F; // 1/sqrt(128)

// Block b of one table row: BF16 [128] at keys[:, b%16, table[b/16]].
__device__ __forceinline__ std::int64_t qsa_block_offset(const std::int32_t* table_row, int block) {
    const int page = table_row[block / kQsaBlocksPerPage];
    return (static_cast<std::int64_t>(page) * kQsaBlocksPerPage + block % kQsaBlocksPerPage) *
           kQsaHeadDim;
}

// One 128-thread CTA normalizes (one-centered RMSNorm) and rotates (interleaved MRoPE R64) one
// 128-wide vector held one dimension per thread. `scratch` holds at least 128 + 4 floats and
// `rotation` the 32 (sin,cos) pairs. Every thread must call it; it returns this thread's rotated
// dimension. Barriers: the first separates the warp partial sums from their readers, the second
// publishes the normalized vector before partner reads, and the trailing one keeps a following
// call from overwriting `scratch` while a slow thread still reads its partner.
__device__ __forceinline__ float qsa_norm_rotate(float x, float weight, const float2* rotation,
                                                 float* scratch) {
    const int dim  = static_cast<int>(threadIdx.x);
    const int lane = dim & 31;
    const int warp = dim >> 5;
    float* partial = scratch + kQsaHeadDim;
    const float squares = warp_sum(x * x);
    if (lane == 0) { partial[warp] = squares; }
    __syncthreads();
    const float sum      = partial[0] + partial[1] + partial[2] + partial[3];
    const float inverse  = rsqrtf(sum * (1.0F / kQsaHeadDim) + kQsaEpsilon);
    const float n        = x * inverse * (1.0F + weight);
    scratch[dim]         = n;
    __syncthreads();
    float out = n;
    if (dim < 32) {
        const float2 sc = rotation[dim];
        out             = n * sc.y - scratch[dim + 32] * sc.x;
    } else if (dim < 64) {
        const float2 sc = rotation[dim - 32];
        out             = n * sc.y + scratch[dim - 32] * sc.x;
    }
    __syncthreads();
    return out;
}

} // namespace ninfer::ops::detail
