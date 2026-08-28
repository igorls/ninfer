#include "targets/qwen3_8_flash_next/impl/hyper_connection_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

constexpr int kHidden      = 2'560;
constexpr int kStreams     = 4;
constexpr int kConcat      = kHidden * kStreams;
constexpr int kLowRank     = 320;
constexpr int kNormThreads = 256;
constexpr int kLinearWarps = 8;

__global__ void group_norm_kernel(const __nv_bfloat16* __restrict__ hidden,
                                  const __nv_bfloat16* __restrict__ norm,
                                  __nv_bfloat16* __restrict__ normalized) {
    __shared__ float warp_sums[kNormThreads / 32];
    const int stream  = static_cast<int>(blockIdx.x);
    const int token   = static_cast<int>(blockIdx.y);
    const auto* input = hidden + static_cast<std::int64_t>(token) * kConcat + stream * kHidden;
    float sum         = 0.0F;
    for (int column = static_cast<int>(threadIdx.x); column < kHidden; column += kNormThreads) {
        const float value = __bfloat162float(input[column]);
        sum               = fmaf(value, value, sum);
    }
    sum = ops::block_reduce_sum<kNormThreads>(sum, warp_sums);
    __shared__ float inverse_rms;
    if (threadIdx.x == 0) { inverse_rms = rsqrtf(sum / kHidden + 1.0e-6F); }
    __syncthreads();
    auto* output = normalized + static_cast<std::int64_t>(token) * kConcat + stream * kHidden;
    for (int column = static_cast<int>(threadIdx.x); column < kHidden; column += kNormThreads) {
        const int flat = stream * kHidden + column;
        const float value =
            __bfloat162float(input[column]) * inverse_rms * (1.0F + __bfloat162float(norm[flat]));
        output[column] = __float2bfloat16_rn(value);
    }
}

__global__ void low_rank_kernel(const __nv_bfloat16* __restrict__ normalized,
                                const __nv_bfloat16* __restrict__ weight,
                                __nv_bfloat16* __restrict__ low_rank) {
    const int token = static_cast<int>(blockIdx.y);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int row   = static_cast<int>(blockIdx.x) * kLinearWarps + warp;
    const auto* x   = normalized + static_cast<std::int64_t>(token) * kConcat;
    float sum       = 0.0F;
    for (int column = lane; column < kConcat; column += 32) {
        sum = fmaf(__bfloat162float(weight[static_cast<std::int64_t>(row) * kConcat + column]),
                   __bfloat162float(x[column]), sum);
    }
    sum = ops::warp_reduce_sum(sum);
    if (lane == 0) {
        low_rank[static_cast<std::int64_t>(token) * kLowRank + row] =
            __float2bfloat16_rn(ops::silu(sum * 0.25F));
    }
}

__global__ void mix_kernel(const __nv_bfloat16* __restrict__ normalized,
                           const __nv_bfloat16* __restrict__ low_rank,
                           const __nv_bfloat16* __restrict__ weight,
                           __nv_bfloat16* __restrict__ block_input) {
    __shared__ float contributions[kStreams];
    const int token  = static_cast<int>(blockIdx.y);
    const int hidden = static_cast<int>(blockIdx.x);
    const int stream = static_cast<int>(threadIdx.x) >> 5;
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int row    = stream * kHidden + hidden;
    const auto* x    = low_rank + static_cast<std::int64_t>(token) * kLowRank;
    float sum        = 0.0F;
    for (int column = lane; column < kLowRank; column += 32) {
        sum = fmaf(__bfloat162float(weight[static_cast<std::int64_t>(row) * kLowRank + column]),
                   __bfloat162float(x[column]), sum);
    }
    sum = ops::warp_reduce_sum(sum);
    if (lane == 0) {
        const float mix = ops::sigmoid(sum);
        contributions[stream] =
            mix * __bfloat162float(normalized[static_cast<std::int64_t>(token) * kConcat + row]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const float mean =
            (contributions[0] + contributions[1] + contributions[2] + contributions[3]) * 0.25F;
        block_input[static_cast<std::int64_t>(token) * kHidden + hidden] =
            __float2bfloat16_rn(mean);
    }
}

__global__ void injection_kernel(const __nv_bfloat16* __restrict__ normalized,
                                 const __nv_bfloat16* __restrict__ weight,
                                 float* __restrict__ injection) {
    const int token  = static_cast<int>(blockIdx.x);
    const int stream = static_cast<int>(threadIdx.x) >> 5;
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const auto* x    = normalized + static_cast<std::int64_t>(token) * kConcat;
    const auto* w    = weight + static_cast<std::int64_t>(stream) * kConcat;
    float sum        = 0.0F;
    for (int column = lane; column < kConcat; column += 32) {
        sum = fmaf(__bfloat162float(w[column]), __bfloat162float(x[column]), sum);
    }
    sum = ops::warp_reduce_sum(sum);
    if (lane == 0) {
        injection[static_cast<std::int64_t>(token) * kStreams + stream] =
            2.0F * ops::sigmoid(sum * 0.25F);
    }
}

__global__ void inject_kernel(const __nv_bfloat16* __restrict__ block_output,
                              const float* __restrict__ injection,
                              __nv_bfloat16* __restrict__ hidden, int elements) {
    const int index =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= elements) { return; }
    const int token      = index / kConcat;
    const int flat       = index - token * kConcat;
    const int stream     = flat / kHidden;
    const int hidden_row = flat - stream * kHidden;
    const float update =
        __bfloat162float(block_output[static_cast<std::int64_t>(token) * kHidden + hidden_row]) *
        injection[static_cast<std::int64_t>(token) * kStreams + stream];
    hidden[index] = __float2bfloat16_rn(__bfloat162float(hidden[index]) + update);
}

void launch_mix_common(const Tensor& hidden, const Tensor& norm, const Weight& down,
                       const Weight& up, FlashNextHyperWorkspace& scratch, Tensor& block_input,
                       cudaStream_t stream) {
    const unsigned tokens = static_cast<unsigned>(hidden.ne[1]);
    group_norm_kernel<<<dim3(kStreams, tokens), kNormThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data),
        static_cast<const __nv_bfloat16*>(norm.data),
        static_cast<__nv_bfloat16*>(scratch.normalized.data));
    CUDA_CHECK(cudaGetLastError());
    low_rank_kernel<<<dim3(kLowRank / kLinearWarps, tokens), kLinearWarps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.normalized.data),
        static_cast<const __nv_bfloat16*>(down.qdata),
        static_cast<__nv_bfloat16*>(scratch.low_rank.data));
    CUDA_CHECK(cudaGetLastError());
    mix_kernel<<<dim3(kHidden, tokens), kStreams * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.normalized.data),
        static_cast<const __nv_bfloat16*>(scratch.low_rank.data),
        static_cast<const __nv_bfloat16*>(up.qdata), static_cast<__nv_bfloat16*>(block_input.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void flash_next_hyper_prepare_launch(const Tensor& hidden, const HyperConnectionWeights& weights,
                                     FlashNextHyperWorkspace& scratch, Tensor& block_input,
                                     cudaStream_t stream) {
    launch_mix_common(hidden, weights.norm, weights.input_mix_down, weights.input_mix_up, scratch,
                      block_input, stream);
    injection_kernel<<<hidden.ne[1], kStreams * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.normalized.data),
        static_cast<const __nv_bfloat16*>(weights.block_inject.qdata),
        static_cast<float*>(scratch.injection.data));
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_hyper_mix_launch(const Tensor& hidden, const HyperMixerWeights& weights,
                                 FlashNextHyperWorkspace& scratch, Tensor& block_input,
                                 cudaStream_t stream) {
    launch_mix_common(hidden, weights.norm, weights.input_mix_down, weights.input_mix_up, scratch,
                      block_input, stream);
}

void flash_next_hyper_inject_launch(const Tensor& block_output, const Tensor& injection,
                                    Tensor& hidden, cudaStream_t stream) {
    const int elements    = hidden.ne[0] * hidden.ne[1];
    constexpr int threads = 256;
    inject_kernel<<<(elements + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(block_output.data),
        static_cast<const float*>(injection.data), static_cast<__nv_bfloat16*>(hidden.data),
        elements);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
