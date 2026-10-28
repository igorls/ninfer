#pragma once
#include "ops/gated_residual/kernel/combine.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/memory.cuh"

namespace ninfer::ops::detail::hc {
// A CTA owns every read and write of one (stream,column), including exact in-place updates.
template <Form Kind, int Block>
__launch_bounds__(Block) __global__ void norm_kernel(Operands o, std::int64_t first) {
    constexpr int kPairs  = kWidth / 2;
    constexpr int kValues = kPairs / Block;
    static_assert(kPairs % Block == 0);
    const std::int64_t row = first + blockIdx.x;
    const auto m           = row / kStreams;
    const int s            = int(row % kStreams);
    const auto base        = row * kPairs;
    const auto* input      = static_cast<const __nv_bfloat162*>(o.input);
    const auto* block      = static_cast<const __nv_bfloat162*>(o.block);
    const auto* weight     = static_cast<const __nv_bfloat162*>(o.weight) + s * kPairs;
    auto* residual         = static_cast<__nv_bfloat162*>(o.residual);
    auto* normalized       = static_cast<__nv_bfloat162*>(o.normalized);
    const float j          = Kind == Form::CombineNorm ? o.injection[row] : 1.0f;
    __nv_bfloat162 values[kValues];
    __nv_bfloat162 weights[kValues];
    float sum = 0;
#pragma unroll
    for (int v = 0; v < kValues; ++v) {
        const int d = threadIdx.x + v * Block;
        auto value  = input[(Kind == Form::ExpandNorm ? m * kPairs : base) + d];
        if constexpr (Kind == Form::CombineNorm || Kind == Form::AddNorm)
            value = update<Kind>(value, block[m * kPairs + d], j);
        values[v]    = value;
        weights[v]   = weight[d];
        const auto x = __bfloat1622float2(value);
        sum += x.x * x.x + x.y * x.y;
    }
    __shared__ float warp_sums[Block / common::kWarpSize];
    __shared__ float inverse;
    const float total = common::block_reduce_sum<Block>(sum, warp_sums);
    if (threadIdx.x == 0) inverse = rsqrtf(total * (1.0f / kWidth) + o.epsilon);
    __syncthreads();
#pragma unroll
    for (int v = 0; v < kValues; ++v) {
        const int d  = threadIdx.x + v * Block;
        const auto x = __bfloat1622float2(values[v]);
        const auto w = __bfloat1622float2(weights[v]);
        if constexpr (Kind != Form::GroupedNorm) residual[base + d] = values[v];
        normalized[base + d] =
            __floats2bfloat162_rn(x.x * inverse * (1.0f + w.x), x.y * inverse * (1.0f + w.y));
    }
}

} // namespace ninfer::ops::detail::hc
