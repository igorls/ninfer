#pragma once

#include "ops/common/warp.cuh"
#include "ops/softmax_attention/selected_block/common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Merge of the batched route's partitions: every (column, head) combines its partitions by their
// maxima and denominators before the only BF16 rounding. Empty partitions (zero denominator)
// contribute nothing; an empty visible set yields exact zero.
__global__ __launch_bounds__(kSelectedHeadDim) void selected_block_decode_merge_kernel(
    const float* __restrict__ partial, __nv_bfloat16* __restrict__ output, int splits) {
    __shared__ float factors[32];
    __shared__ float denominator;
    const std::int64_t row =
        static_cast<std::int64_t>(blockIdx.y) * kSelectedQueryHeads + blockIdx.x;
    const float* values = partial + row * splits * kSelectedPartialStride;
    if (threadIdx.x < 32) {
        const int lane = static_cast<int>(threadIdx.x);
        const float split_max =
            lane < splits ? values[lane * kSelectedPartialStride + kSelectedHeadDim] : -CUDART_INF_F;
        const float split_sum =
            lane < splits ? values[lane * kSelectedPartialStride + kSelectedHeadDim + 1] : 0.0F;
        const float maximum = warp_max(split_max);
        const float factor  = split_sum > 0.0F ? expf(split_max - maximum) : 0.0F;
        factors[lane]       = factor;
        const float total   = warp_sum(factor * split_sum);
        if (lane == 0) { denominator = total; }
    }
    __syncthreads();
    float value = 0.0F;
    for (int split = 0; split < splits; ++split) {
        value = fmaf(values[split * kSelectedPartialStride + threadIdx.x], factors[split], value);
    }
    output[row * kSelectedHeadDim + threadIdx.x] =
        __float2bfloat16_rn(denominator > 0.0F ? value / denominator : 0.0F);
}

} // namespace ninfer::ops::detail
