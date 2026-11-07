#pragma once

#include "ops/rope/apply/dispatch.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// D256/R64 Text RoPE with host-resolved YaRN frequencies. One CTA owns one token: the first warp
// forms the 32 pair phases in FP64 and reduces them modulo 2*pi before the FP32 sincos, so the
// rotation stays accurate to FP32 rounding across the whole I32 position domain. The magnitude
// scales the shared coefficients; one warp then rotates a head, lane i owning dimensions i and
// i+32, so no storage alignment beyond BF16 is required.
template <int Axes>
__global__ void rope_yarn_d256r64_kernel(const std::int32_t* positions, __nv_bfloat16* q,
                                         __nv_bfloat16* k, std::int32_t tokens,
                                         std::int32_t q_heads, std::int32_t k_heads,
                                         std::int64_t q_token_stride, std::int64_t k_token_stride,
                                         YarnR64Table table) {
    static_assert(Axes == 1 || Axes == 3);
    constexpr double kTwoPi    = 6.283185307179586476925286766559;
    constexpr double kInvTwoPi = 0.15915494309189533576888376337251;
    const int token            = static_cast<int>(blockIdx.x);
    if (token >= tokens) return;

    __shared__ float cos_cache[kYarnR64Pairs];
    __shared__ float sin_cache[kYarnR64Pairs];
    if (threadIdx.x < kYarnR64Pairs) {
        const int pair = static_cast<int>(threadIdx.x);
        const int axis = Axes == 3 ? pair % 3 : 0;
        const double phase =
            static_cast<double>(positions[static_cast<std::int64_t>(axis) * tokens + token]) *
            table.frequency[pair];
        const double reduced = fma(-kTwoPi, rint(phase * kInvTwoPi), phase);
        float sine           = 0.0F;
        float cosine         = 0.0F;
        sincosf(static_cast<float>(reduced), &sine, &cosine);
        cos_cache[pair] = cosine * table.magnitude;
        sin_cache[pair] = sine * table.magnitude;
    }
    __syncthreads();

    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int block_warps = static_cast<int>(blockDim.x) >> 5;
    const float cosine    = cos_cache[lane];
    const float sine      = sin_cache[lane];
    for (int combined = warp; combined < q_heads + k_heads; combined += block_warps) {
        const bool query          = combined < q_heads;
        const int head            = query ? combined : combined - q_heads;
        const std::int64_t stride = query ? q_token_stride : k_token_stride;
        __nv_bfloat16* data       = (query ? q : k) + static_cast<std::int64_t>(token) * stride +
                              static_cast<std::int64_t>(head) * 256;
        const float first  = __bfloat162float(data[lane]);
        const float second = __bfloat162float(data[lane + kYarnR64Pairs]);
        data[lane]         = __float2bfloat16_rn(first * cosine - second * sine);
        data[lane + kYarnR64Pairs] = __float2bfloat16_rn(second * cosine + first * sine);
    }
}

} // namespace ninfer::ops::detail
