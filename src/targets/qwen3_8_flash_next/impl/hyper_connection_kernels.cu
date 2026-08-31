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
constexpr int kConcat      = kHidden * kStreams; // 10,240
constexpr int kLowRank     = 320;
constexpr int kNormThreads = 256;

// =========================================================================
// Kernel 1: Vectorized 4-Stream Group RMSNorm
// Grid: dim3(4, tokens), Block: 256 threads
// =========================================================================
__global__ void __launch_bounds__(kNormThreads)
group_norm_vectorized_kernel(const __nv_bfloat16* __restrict__ hidden,
                             const __nv_bfloat16* __restrict__ norm,
                             __nv_bfloat16* __restrict__ normalized,
                             int tokens) {
    __shared__ float s_warp_sums[kNormThreads / 32];
    __shared__ float s_inv_rms;

    const int stream = static_cast<int>(blockIdx.x);
    const int token  = static_cast<int>(blockIdx.y);
    const int tid    = static_cast<int>(threadIdx.x);

    if (token >= tokens || stream >= kStreams) { return; }

    const int stream_offset = stream * kHidden;
    const auto* in_stream   = hidden + static_cast<std::int64_t>(token) * kConcat + stream_offset;

    // 2560 elements = 320 ulonglong2 (8 BF16) chunks
    float sum_sq = 0.0F;
    for (int chunk = tid; chunk < (kHidden / 8); chunk += kNormThreads) {
        const auto raw_in = *reinterpret_cast<const ulonglong2*>(in_stream + chunk * 8);
        const auto* bf_in = reinterpret_cast<const __nv_bfloat16*>(&raw_in);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float v = __bfloat162float(bf_in[i]);
            sum_sq        = fmaf(v, v, sum_sq);
        }
    }

    sum_sq = ops::block_reduce_sum<kNormThreads>(sum_sq, s_warp_sums);
    if (tid == 0) {
        s_inv_rms = rsqrtf(sum_sq / static_cast<float>(kHidden) + 1.0e-6F);
    }
    __syncthreads();

    const float inv_rms = s_inv_rms;
    const auto* norm_stream = norm + stream_offset;
    auto* out_stream        = normalized + static_cast<std::int64_t>(token) * kConcat + stream_offset;

    for (int chunk = tid; chunk < (kHidden / 8); chunk += kNormThreads) {
        const int col_base  = chunk * 8;
        const auto raw_in   = *reinterpret_cast<const ulonglong2*>(in_stream + col_base);
        const auto* bf_in   = reinterpret_cast<const __nv_bfloat16*>(&raw_in);
        const auto raw_norm = *reinterpret_cast<const ulonglong2*>(norm_stream + col_base);
        const auto* bf_norm = reinterpret_cast<const __nv_bfloat16*>(&raw_norm);

        ulonglong2 raw_out;
        auto* bf_out = reinterpret_cast<__nv_bfloat16*>(&raw_out);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float v = __bfloat162float(bf_in[i]);
            const float n = __bfloat162float(bf_norm[i]);
            bf_out[i]     = __float2bfloat16_rn(v * inv_rms * (1.0F + n));
        }
        *reinterpret_cast<ulonglong2*>(out_stream + col_base) = raw_out;
    }
}

// =========================================================================
// Kernel 2: Fused Down Projection & Injection Gates
// Grid: dim3(total_rows, tokens) where total_rows = 324 (or 320)
// Block: 256 threads (8 warps)
// 1 CTA per row -> 324 CTAs at T=1 saturate all SMs.
// Vectorized 128-bit memory loads across 10,240 elements (40 steps x 8 elements).
// =========================================================================
__global__ void __launch_bounds__(256)
low_rank_and_injection_kernel(const __nv_bfloat16* __restrict__ normalized,
                              const __nv_bfloat16* __restrict__ down_weight,
                              const __nv_bfloat16* __restrict__ inject_weight,
                              __nv_bfloat16* __restrict__ low_rank,
                              float* __restrict__ injection,
                              int tokens, int total_rows) {
    __shared__ float s_warp_sums[8]; // 256 / 32

    const int row   = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int tid   = static_cast<int>(threadIdx.x);

    if (token >= tokens || row >= total_rows) { return; }

    const auto* x_token = normalized + static_cast<std::int64_t>(token) * kConcat;
    const __nv_bfloat16* w_row = (row < kLowRank)
        ? (down_weight + static_cast<std::int64_t>(row) * kConcat)
        : (inject_weight + static_cast<std::int64_t>(row - kLowRank) * kConcat);

    // 10,240 elements = 1,280 ulonglong2 chunks.
    // With 256 threads, each thread loads 1280 / 256 = 5 chunks (40 BF16 elements).
    float sum = 0.0F;
    #pragma unroll
    for (int chunk = tid; chunk < (kConcat / 8); chunk += 256) {
        const int col_base = chunk * 8;
        const auto w_raw   = *reinterpret_cast<const ulonglong2*>(w_row + col_base);
        const auto* w_bf   = reinterpret_cast<const __nv_bfloat16*>(&w_raw);
        const auto x_raw   = *reinterpret_cast<const ulonglong2*>(x_token + col_base);
        const auto* x_bf   = reinterpret_cast<const __nv_bfloat16*>(&x_raw);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            sum = fmaf(__bfloat162float(w_bf[i]), __bfloat162float(x_bf[i]), sum);
        }
    }

    sum = ops::block_reduce_sum<256>(sum, s_warp_sums);

    if (tid == 0) {
        if (row < kLowRank) {
            low_rank[static_cast<std::int64_t>(token) * kLowRank + row] =
                __float2bfloat16_rn(ops::silu(sum * 0.25F));
        } else if (injection != nullptr) {
            injection[static_cast<std::int64_t>(token) * kStreams + (row - kLowRank)] =
                2.0F * ops::sigmoid(sum * 0.25F);
        }
    }
}

// =========================================================================
// Kernel 3: Mix Up Projection & Stream Reduction
// Grid: dim3(kHidden = 2560, tokens), Block: 128 threads (4 warps = 1 warp per stream)
// 2,560 CTAs at T=1 saturate all SMs.
// Warp s computes stream s dot-product (320 elements), applies sigmoid * normalized,
// thread 0 averages the 4 streams into block_input.
// =========================================================================
__global__ void __launch_bounds__(128)
mix_up_and_reduce_kernel(const __nv_bfloat16* __restrict__ normalized,
                         const __nv_bfloat16* __restrict__ low_rank,
                         const __nv_bfloat16* __restrict__ up_weight,
                         __nv_bfloat16* __restrict__ block_input,
                         int tokens) {
    __shared__ float s_contrib[kStreams];

    const int hidden  = static_cast<int>(blockIdx.x);
    const int token   = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int stream  = tid >> 5; // 0, 1, 2, 3
    const int lane_id = tid & 31; // 0..31

    if (token >= tokens || hidden >= kHidden) { return; }

    const int row              = stream * kHidden + hidden;
    const __nv_bfloat16* w_row = up_weight + static_cast<std::int64_t>(row) * kLowRank;
    const auto* lr_token       = low_rank + static_cast<std::int64_t>(token) * kLowRank;

    // 320 elements:
    // Step 0: 32 lanes x 8 elements = 256 elements (columns 0..255)
    float sum = 0.0F;
    {
        const int col_base = lane_id * 8;
        const auto w_raw   = *reinterpret_cast<const ulonglong2*>(w_row + col_base);
        const auto* w_bf   = reinterpret_cast<const __nv_bfloat16*>(&w_raw);
        const auto x_raw   = *reinterpret_cast<const ulonglong2*>(lr_token + col_base);
        const auto* x_bf   = reinterpret_cast<const __nv_bfloat16*>(&x_raw);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            sum = fmaf(__bfloat162float(w_bf[i]), __bfloat162float(x_bf[i]), sum);
        }
    }
    // Step 1: 32 lanes x 2 elements = 64 elements (columns 256..319)
    {
        const int col_base = 256 + lane_id * 2;
        const auto w_raw   = *reinterpret_cast<const std::uint32_t*>(w_row + col_base);
        const auto* w_bf   = reinterpret_cast<const __nv_bfloat16*>(&w_raw);
        const auto x_raw   = *reinterpret_cast<const std::uint32_t*>(lr_token + col_base);
        const auto* x_bf   = reinterpret_cast<const __nv_bfloat16*>(&x_raw);

        sum = fmaf(__bfloat162float(w_bf[0]), __bfloat162float(x_bf[0]), sum);
        sum = fmaf(__bfloat162float(w_bf[1]), __bfloat162float(x_bf[1]), sum);
    }

    sum = ops::warp_reduce_sum(sum);

    if (lane_id == 0) {
        const float mix_gate = ops::sigmoid(sum);
        const float norm_val =
            __bfloat162float(normalized[static_cast<std::int64_t>(token) * kConcat + row]);
        s_contrib[stream]    = mix_gate * norm_val;
    }
    __syncthreads();

    if (tid == 0) {
        const float mean =
            (s_contrib[0] + s_contrib[1] + s_contrib[2] + s_contrib[3]) * 0.25F;
        block_input[static_cast<std::int64_t>(token) * kHidden + hidden] =
            __float2bfloat16_rn(mean);
    }
}

// =========================================================================
// Kernel 4: Vectorized In-Place Injection (hidden += block_output * injection)
// Grid: dim3((10240 / 8 + 255) / 256, tokens), Block: 256 threads
// =========================================================================
__global__ void __launch_bounds__(256)
hyper_inject_vectorized_kernel(const __nv_bfloat16* __restrict__ block_output,
                               const float* __restrict__ injection,
                               __nv_bfloat16* __restrict__ hidden,
                               int tokens) {
    const int tid   = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int token = static_cast<int>(blockIdx.y);

    if (token >= tokens || tid >= (kConcat / 8)) { return; }

    const int col_base   = tid * 8;
    const int stream     = col_base / kHidden;
    const int hidden_col = col_base - stream * kHidden;

    const float inj_scale = injection[static_cast<std::int64_t>(token) * kStreams + stream];
    const auto* out_ptr   = block_output + static_cast<std::int64_t>(token) * kHidden + hidden_col;
    auto* hid_ptr         = hidden + static_cast<std::int64_t>(token) * kConcat + col_base;

    const auto out_raw = *reinterpret_cast<const ulonglong2*>(out_ptr);
    const auto* out_bf = reinterpret_cast<const __nv_bfloat16*>(&out_raw);
    const auto hid_raw = *reinterpret_cast<const ulonglong2*>(hid_ptr);
    const auto* hid_bf = reinterpret_cast<const __nv_bfloat16*>(&hid_raw);

    ulonglong2 res_raw;
    auto* res_bf = reinterpret_cast<__nv_bfloat16*>(&res_raw);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float out_f = __bfloat162float(out_bf[i]);
        const float hid_f = __bfloat162float(hid_bf[i]);
        const float upd   = fmaf(out_f, inj_scale, hid_f);
        res_bf[i]         = __float2bfloat16_rn(upd);
    }

    *reinterpret_cast<ulonglong2*>(hid_ptr) = res_raw;
}

} // namespace

void flash_next_hyper_prepare_launch(const Tensor& hidden, const HyperConnectionWeights& weights,
                                     FlashNextHyperWorkspace& scratch, Tensor& block_input,
                                     cudaStream_t stream) {
    const int tokens = static_cast<int>(hidden.ne[1]);

    // 1. Group Norm (4 CTAs per token)
    group_norm_vectorized_kernel<<<dim3(kStreams, tokens), kNormThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data),
        static_cast<const __nv_bfloat16*>(weights.norm.data),
        static_cast<__nv_bfloat16*>(scratch.normalized.data), tokens);
    CUDA_CHECK(cudaGetLastError());

    // 2. Fused Down Projection + Injection Gates (324 CTAs per token)
    constexpr int kTotalRows = kLowRank + kStreams; // 324
    low_rank_and_injection_kernel<<<dim3(kTotalRows, tokens), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.normalized.data),
        static_cast<const __nv_bfloat16*>(weights.input_mix_down.qdata),
        static_cast<const __nv_bfloat16*>(weights.block_inject.qdata),
        static_cast<__nv_bfloat16*>(scratch.low_rank.data),
        static_cast<float*>(scratch.injection.data), tokens, kTotalRows);
    CUDA_CHECK(cudaGetLastError());

    // 3. Mix Up Projection & 4-Stream Reduction (2,560 CTAs per token)
    mix_up_and_reduce_kernel<<<dim3(kHidden, tokens), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.normalized.data),
        static_cast<const __nv_bfloat16*>(scratch.low_rank.data),
        static_cast<const __nv_bfloat16*>(weights.input_mix_up.qdata),
        static_cast<__nv_bfloat16*>(block_input.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_hyper_mix_launch(const Tensor& hidden, const HyperMixerWeights& weights,
                                 FlashNextHyperWorkspace& scratch, Tensor& block_input,
                                 cudaStream_t stream) {
    const int tokens = static_cast<int>(hidden.ne[1]);

    // 1. Group Norm (4 CTAs per token)
    group_norm_vectorized_kernel<<<dim3(kStreams, tokens), kNormThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data),
        static_cast<const __nv_bfloat16*>(weights.norm.data),
        static_cast<__nv_bfloat16*>(scratch.normalized.data), tokens);
    CUDA_CHECK(cudaGetLastError());

    // 2. Down Projection only (320 CTAs per token)
    low_rank_and_injection_kernel<<<dim3(kLowRank, tokens), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.normalized.data),
        static_cast<const __nv_bfloat16*>(weights.input_mix_down.qdata),
        nullptr,
        static_cast<__nv_bfloat16*>(scratch.low_rank.data),
        nullptr, tokens, kLowRank);
    CUDA_CHECK(cudaGetLastError());

    // 3. Mix Up Projection & 4-Stream Reduction (2,560 CTAs per token)
    mix_up_and_reduce_kernel<<<dim3(kHidden, tokens), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.normalized.data),
        static_cast<const __nv_bfloat16*>(scratch.low_rank.data),
        static_cast<const __nv_bfloat16*>(weights.input_mix_up.qdata),
        static_cast<__nv_bfloat16*>(block_input.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_hyper_inject_launch(const Tensor& block_output, const Tensor& injection,
                                    Tensor& hidden, cudaStream_t stream) {
    const int tokens = static_cast<int>(hidden.ne[1]);
    dim3 block(256);
    dim3 grid((kConcat / 8 + block.x - 1) / block.x, tokens);
    hyper_inject_vectorized_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(block_output.data),
        static_cast<const float*>(injection.data),
        static_cast<__nv_bfloat16*>(hidden.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
