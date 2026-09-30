#pragma once

// Four-stream hyper-connection kernels. The decode chain (T <= 8) is CUDA-core: a group norm
// over (stream, token) CTAs, one CTA per low-rank/injection row, and one CTA per hidden column
// applying its four up rows to every token. The tensor-core route reuses the BF16 Linear MMA
// contraction for the down and up projections (launch.cu).

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kHyperHidden      = 2560;
inline constexpr int kHyperStreams     = 4;
inline constexpr int kHyperConcat      = kHyperHidden * kHyperStreams;
inline constexpr int kHyperLowRank     = 320;
inline constexpr int kHyperNormThreads = 256;

__device__ __forceinline__ ulonglong2 hyper_normalize_chunk(ulonglong2 raw_in, ulonglong2 raw_norm,
                                                            float inv_rms) {
    const auto* in   = reinterpret_cast<const __nv_bfloat16*>(&raw_in);
    const auto* gain = reinterpret_cast<const __nv_bfloat16*>(&raw_norm);
    ulonglong2 raw_out;
    auto* out = reinterpret_cast<__nv_bfloat16*>(&raw_out);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i] = __float2bfloat16_rn(__bfloat162float(in[i]) * inv_rms *
                                     (1.0F + __bfloat162float(gain[i])));
    }
    return raw_out;
}

__device__ __forceinline__ float hyper_dot_chunk(ulonglong2 w_raw, ulonglong2 x_raw, float sum) {
    const auto* w = reinterpret_cast<const __nv_bfloat16*>(&w_raw);
    const auto* x = reinterpret_cast<const __nv_bfloat16*>(&x_raw);
#pragma unroll
    for (int i = 0; i < 8; ++i) sum = fmaf(__bfloat162float(w[i]), __bfloat162float(x[i]), sum);
    return sum;
}

// Grid (4, T), 256 threads: one CTA normalizes one 2560-wide stream of one token.
__global__ void __launch_bounds__(kHyperNormThreads)
    hyper_group_norm_decode_kernel(const __nv_bfloat16* __restrict__ hidden,
                                   const __nv_bfloat16* __restrict__ norm,
                                   __nv_bfloat16* __restrict__ normalized) {
    __shared__ float warp_sums[kHyperNormThreads / 32];
    __shared__ float inv_rms;
    const int stream = static_cast<int>(blockIdx.x);
    const int token  = static_cast<int>(blockIdx.y);
    const int tid    = static_cast<int>(threadIdx.x);
    const auto base  = static_cast<std::int64_t>(token) * kHyperConcat + stream * kHyperHidden;

    float sum_sq = 0.0F;
    for (int chunk = tid; chunk < kHyperHidden / 8; chunk += kHyperNormThreads) {
        const auto raw = *reinterpret_cast<const ulonglong2*>(hidden + base + chunk * 8);
        const auto* v  = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            sum_sq = fmaf(__bfloat162float(v[i]), __bfloat162float(v[i]), sum_sq);
        }
    }
    // One reduction per launch: warp_sums is never rewritten after it, the total lives in warp 0
    // and inv_rms is published behind its own barrier.
    sum_sq = block_reduce_sum<kHyperNormThreads>(sum_sq, warp_sums);
    if (tid == 0) inv_rms = rsqrtf(sum_sq / static_cast<float>(kHyperHidden) + 1.0e-6F);
    __syncthreads();

    const float scale = inv_rms;
    for (int chunk = tid; chunk < kHyperHidden / 8; chunk += kHyperNormThreads) {
        const int column = chunk * 8;
        const auto raw   = *reinterpret_cast<const ulonglong2*>(hidden + base + column);
        const auto gain =
            *reinterpret_cast<const ulonglong2*>(norm + stream * kHyperHidden + column);
        *reinterpret_cast<ulonglong2*>(normalized + base + column) =
            hyper_normalize_chunk(raw, gain, scale);
    }
}

// Grid (rows, T), 256 threads: rows [0,320) are the SiLU'd low-rank rows, rows [320,324) the
// injection gates (absent in the mixer form, rows == 320).
__global__ void __launch_bounds__(256)
    hyper_low_rank_decode_kernel(const __nv_bfloat16* __restrict__ normalized,
                                 const __nv_bfloat16* __restrict__ down,
                                 const __nv_bfloat16* __restrict__ inject,
                                 __nv_bfloat16* __restrict__ low_rank,
                                 float* __restrict__ injection) {
    __shared__ float warp_sums[8];
    const int row   = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int tid   = static_cast<int>(threadIdx.x);
    const auto* x   = normalized + static_cast<std::int64_t>(token) * kHyperConcat;
    const auto* w   = row < kHyperLowRank
                          ? down + static_cast<std::int64_t>(row) * kHyperConcat
                          : inject + static_cast<std::int64_t>(row - kHyperLowRank) * kHyperConcat;
    float sum       = 0.0F;
#pragma unroll
    for (int chunk = tid; chunk < kHyperConcat / 8; chunk += 256) {
        sum = hyper_dot_chunk(*reinterpret_cast<const ulonglong2*>(w + chunk * 8),
                              *reinterpret_cast<const ulonglong2*>(x + chunk * 8), sum);
    }
    // One reduction per launch; only thread 0 (in warp 0) consumes it.
    sum = block_reduce_sum<256>(sum, warp_sums);
    if (tid == 0) {
        if (row < kHyperLowRank) {
            low_rank[static_cast<std::int64_t>(token) * kHyperLowRank + row] =
                __float2bfloat16_rn(silu(sum * 0.25F));
        } else {
            injection[static_cast<std::int64_t>(token) * kHyperStreams + (row - kHyperLowRank)] =
                2.0F * sigmoid(sum * 0.25F);
        }
    }
}

// Grid 2560, 128 threads: warp s loads its 320-wide up row once and applies it to every token;
// thread t < Tokens then averages the four stream contributions of token t.
template <int Tokens>
__global__ void __launch_bounds__(128)
    hyper_mix_up_decode_kernel(const __nv_bfloat16* __restrict__ normalized,
                               const __nv_bfloat16* __restrict__ low_rank,
                               const __nv_bfloat16* __restrict__ up,
                               __nv_bfloat16* __restrict__ block_input) {
    __shared__ float contribution[Tokens][kHyperStreams];
    const int column = static_cast<int>(blockIdx.x);
    const int tid    = static_cast<int>(threadIdx.x);
    const int stream = tid >> 5;
    const int lane   = tid & 31;
    const int row    = stream * kHyperHidden + column;
    const auto* w    = up + static_cast<std::int64_t>(row) * kHyperLowRank;

    // 320 values: columns [0,256) as 32 lanes x 8, columns [256,320) as 32 lanes x 2.
    const auto w_lo_raw = *reinterpret_cast<const ulonglong2*>(w + lane * 8);
    const auto w_hi_raw = *reinterpret_cast<const std::uint32_t*>(w + 256 + lane * 2);
    const auto* w_lo    = reinterpret_cast<const __nv_bfloat16*>(&w_lo_raw);
    const auto* w_hi    = reinterpret_cast<const __nv_bfloat16*>(&w_hi_raw);

    float sums[Tokens];
#pragma unroll
    for (int t = 0; t < Tokens; ++t) {
        const auto* l       = low_rank + static_cast<std::int64_t>(t) * kHyperLowRank;
        const auto x_lo_raw = *reinterpret_cast<const ulonglong2*>(l + lane * 8);
        const auto x_hi_raw = *reinterpret_cast<const std::uint32_t*>(l + 256 + lane * 2);
        const auto* x_lo    = reinterpret_cast<const __nv_bfloat16*>(&x_lo_raw);
        const auto* x_hi    = reinterpret_cast<const __nv_bfloat16*>(&x_hi_raw);
        float s             = 0.0F;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s = fmaf(__bfloat162float(w_lo[i]), __bfloat162float(x_lo[i]), s);
        }
        s       = fmaf(__bfloat162float(w_hi[0]), __bfloat162float(x_hi[0]), s);
        s       = fmaf(__bfloat162float(w_hi[1]), __bfloat162float(x_hi[1]), s);
        sums[t] = s;
    }
#pragma unroll
    for (int t = 0; t < Tokens; ++t) {
        const float total = warp_reduce_sum(sums[t]);
        if (lane == 0) {
            contribution[t][stream] =
                sigmoid(total) *
                __bfloat162float(normalized[static_cast<std::int64_t>(t) * kHyperConcat + row]);
        }
    }
    __syncthreads();
    if (tid < Tokens) {
        const float mean = (contribution[tid][0] + contribution[tid][1] + contribution[tid][2] +
                            contribution[tid][3]) *
                           0.25F;
        block_input[static_cast<std::int64_t>(tid) * kHyperHidden + column] =
            __float2bfloat16_rn(mean);
    }
}

// Grid T, 256 threads: 64 threads per stream; two warps per stream reduce with shuffles and one
// per-stream shared pair (no block_reduce_sum).
__global__ void __launch_bounds__(256)
    hyper_group_norm_wide_kernel(const __nv_bfloat16* __restrict__ hidden,
                                 const __nv_bfloat16* __restrict__ norm,
                                 __nv_bfloat16* __restrict__ normalized) {
    __shared__ float warp_sums[kHyperStreams][2];
    __shared__ float inv_rms[kHyperStreams];
    const int token      = static_cast<int>(blockIdx.x);
    const int tid        = static_cast<int>(threadIdx.x);
    const int stream     = tid / 64;
    const int stream_tid = tid % 64;
    const auto base = static_cast<std::int64_t>(token) * kHyperConcat + stream * kHyperHidden;

    float sum_sq = 0.0F;
#pragma unroll
    for (int step = 0; step < 5; ++step) {
        const auto raw =
            *reinterpret_cast<const ulonglong2*>(hidden + base + (stream_tid + step * 64) * 8);
        const auto* v = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            sum_sq = fmaf(__bfloat162float(v[i]), __bfloat162float(v[i]), sum_sq);
        }
    }
    sum_sq = warp_reduce_sum(sum_sq);
    if ((stream_tid & 31) == 0) warp_sums[stream][stream_tid / 32] = sum_sq;
    __syncthreads();
    if (stream_tid == 0) {
        inv_rms[stream] = rsqrtf((warp_sums[stream][0] + warp_sums[stream][1]) /
                                     static_cast<float>(kHyperHidden) +
                                 1.0e-6F);
    }
    __syncthreads();
    const float scale = inv_rms[stream];
#pragma unroll
    for (int step = 0; step < 5; ++step) {
        const int column = (stream_tid + step * 64) * 8;
        const auto raw   = *reinterpret_cast<const ulonglong2*>(hidden + base + column);
        const auto gain =
            *reinterpret_cast<const ulonglong2*>(norm + stream * kHyperHidden + column);
        *reinterpret_cast<ulonglong2*>(normalized + base + column) =
            hyper_normalize_chunk(raw, gain, scale);
    }
}

// Grid T, 128 threads: warp s computes injection gate s of one token.
__global__ void __launch_bounds__(128)
    hyper_injection_wide_kernel(const __nv_bfloat16* __restrict__ normalized,
                                const __nv_bfloat16* __restrict__ inject,
                                float* __restrict__ injection) {
    const int token  = static_cast<int>(blockIdx.x);
    const int tid    = static_cast<int>(threadIdx.x);
    const int stream = tid >> 5;
    const int lane   = tid & 31;
    const auto* x    = normalized + static_cast<std::int64_t>(token) * kHyperConcat;
    const auto* w    = inject + static_cast<std::int64_t>(stream) * kHyperConcat;
    float sum        = 0.0F;
#pragma unroll 4
    for (int step = 0; step < kHyperConcat / 8 / 32; ++step) {
        const int column = (lane + step * 32) * 8;
        sum              = hyper_dot_chunk(*reinterpret_cast<const ulonglong2*>(w + column),
                                           *reinterpret_cast<const ulonglong2*>(x + column), sum);
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        injection[static_cast<std::int64_t>(token) * kHyperStreams + stream] =
            2.0F * sigmoid(sum * 0.25F);
    }
}

// Grid T, 256 threads: block_input = mean over streams of sigmoid(up) * normalized.
__global__ void __launch_bounds__(256)
    hyper_mix_reduce_wide_kernel(const __nv_bfloat16* __restrict__ up,
                                 const __nv_bfloat16* __restrict__ normalized,
                                 __nv_bfloat16* __restrict__ block_input) {
    const int token   = static_cast<int>(blockIdx.x);
    const int tid     = static_cast<int>(threadIdx.x);
    const auto offset = static_cast<std::int64_t>(token) * kHyperConcat;
    for (int chunk = tid; chunk < kHyperHidden / 8; chunk += 256) {
        const int column = chunk * 8;
        float mean[8]    = {};
#pragma unroll
        for (int s = 0; s < kHyperStreams; ++s) {
            const auto z_raw =
                *reinterpret_cast<const ulonglong2*>(up + offset + s * kHyperHidden + column);
            const auto n_raw = *reinterpret_cast<const ulonglong2*>(normalized + offset +
                                                                     s * kHyperHidden + column);
            const auto* z    = reinterpret_cast<const __nv_bfloat16*>(&z_raw);
            const auto* n    = reinterpret_cast<const __nv_bfloat16*>(&n_raw);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                mean[i] += sigmoid(__bfloat162float(z[i])) * __bfloat162float(n[i]);
            }
        }
        ulonglong2 raw_out;
        auto* out = reinterpret_cast<__nv_bfloat16*>(&raw_out);
#pragma unroll
        for (int i = 0; i < 8; ++i) out[i] = __float2bfloat16_rn(mean[i] * 0.25F);
        *reinterpret_cast<ulonglong2*>(block_input +
                                       static_cast<std::int64_t>(token) * kHyperHidden + column) =
            raw_out;
    }
}

inline constexpr int kHyperInjectBlocksPerToken = (kHyperConcat / 8 + 255) / 256;

// Grid 5*T, 256 threads: hidden += block_output * injection over 8-value chunks.
__global__ void __launch_bounds__(256)
    hyper_inject_kernel(const __nv_bfloat16* __restrict__ block_output,
                        const float* __restrict__ injection, __nv_bfloat16* __restrict__ hidden) {
    const int token = static_cast<int>(blockIdx.x / kHyperInjectBlocksPerToken);
    const int chunk = static_cast<int>(blockIdx.x % kHyperInjectBlocksPerToken) * 256 +
                      static_cast<int>(threadIdx.x);
    if (chunk >= kHyperConcat / 8) return;
    const int column   = chunk * 8;
    const int stream   = column / kHyperHidden;
    const float scale  = injection[static_cast<std::int64_t>(token) * kHyperStreams + stream];
    const auto out_raw = *reinterpret_cast<const ulonglong2*>(
        block_output + static_cast<std::int64_t>(token) * kHyperHidden + column -
        stream * kHyperHidden);
    auto* h          = hidden + static_cast<std::int64_t>(token) * kHyperConcat + column;
    const auto h_raw = *reinterpret_cast<const ulonglong2*>(h);
    const auto* o    = reinterpret_cast<const __nv_bfloat16*>(&out_raw);
    const auto* v    = reinterpret_cast<const __nv_bfloat16*>(&h_raw);
    ulonglong2 result;
    auto* r = reinterpret_cast<__nv_bfloat16*>(&result);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        r[i] = __float2bfloat16_rn(fmaf(__bfloat162float(o[i]), scale, __bfloat162float(v[i])));
    }
    *reinterpret_cast<ulonglong2*>(h) = result;
}

} // namespace ninfer::ops::detail
