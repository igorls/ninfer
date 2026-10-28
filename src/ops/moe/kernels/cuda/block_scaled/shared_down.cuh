#pragma once

#include "ops/moe/kernels/cuda/block_scaled/launch.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::detail::moe_cuda::block_scaled {

__device__ __forceinline__ float shared_down_dot(const float* activation,
                                                 const __nv_bfloat16* weights, int row,
                                                 int lane) {
    float sum = 0;
    for (int k = lane * 8; k < kIntermediate; k += 32 * 8) {
        const float4 x0 = common::load_vec<float4>(activation + k);
        const float4 x1 = common::load_vec<float4>(activation + k + 4);
        const auto packed = common::load_vec<uint4>(weights + std::int64_t(row) * kIntermediate + k);
        const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
        const float x[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const auto w = common::bf16x2_bits_to_float2(words[j]);
            sum = fmaf(x[j * 2], w.x, sum);
            sum = fmaf(x[j * 2 + 1], w.y, sum);
        }
    }
    return common::warp_reduce_sum(sum);
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled
