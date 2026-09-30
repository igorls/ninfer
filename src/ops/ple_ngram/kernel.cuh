#pragma once

// Flash-Next per-layer-embedding kernels: exact u4z8 row decode, the per-stream gated combine and
// group norm, and the dilation-3 four-tap causal convolution with its nine-column BF16 history.

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kPleStreamWidth = 2560;
inline constexpr int kPleStreams     = 4;
inline constexpr int kPleWidth       = kPleStreamWidth * kPleStreams;
inline constexpr int kPleHistory     = 9;
inline constexpr int kPleRowWidth    = 160;
inline constexpr int kPleGateThreads = 256;
inline constexpr int kPleConvThreads = 256;
inline constexpr int kPleConvBlocks  = kPleWidth / kPleConvThreads;

// One thread decodes one group of 16 values of one row: 8 code bytes and one FP16 multiplier in,
// 32 bytes of BF16 out.
__global__ void __launch_bounds__(256)
    ple_decode_kernel(const std::uint8_t* __restrict__ codes, const __half* __restrict__ scales,
                      __nv_bfloat16* __restrict__ out, std::int64_t groups) {
    const std::int64_t group =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + static_cast<std::int64_t>(threadIdx.x);
    if (group >= groups) return;
    const uint2 packed = *reinterpret_cast<const uint2*>(codes + group * 8);
    const float scale  = __half2float(scales[group]);
    const std::uint32_t words[2] = {packed.x, packed.y};
    uint4 result[2];
    auto* values = reinterpret_cast<__nv_bfloat162*>(result);
#pragma unroll
    for (int byte = 0; byte < 8; ++byte) {
        const std::uint32_t code = (words[byte / 4] >> (8 * (byte % 4))) & 0xffU;
        const float low          = static_cast<float>(static_cast<int>(code & 0x0fU) - 8) * scale;
        const float high         = static_cast<float>(static_cast<int>(code >> 4U) - 8) * scale;
        values[byte] = __nv_bfloat162(__float2bfloat16_rn(low), __float2bfloat16_rn(high));
    }
    auto* destination = reinterpret_cast<uint4*>(out + group * 16);
    destination[0]    = result[0];
    destination[1]    = result[1];
}

// Sums N values over the CTA and returns the totals to every thread. `partials` holds one row of N
// values per warp and belongs to exactly one call site of a launch: it is written once, read
// after the barrier and never rewritten, so no trailing barrier is needed.
template <int Threads, int N>
__device__ __forceinline__ void ple_block_sum(float (&values)[N], float (*partials)[N]) {
    constexpr int kWarps = Threads / 32;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int i = 0; i < N; ++i) values[i] = warp_reduce_sum(values[i]);
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < N; ++i) partials[warp][i] = values[i];
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < N; ++i) {
        float total = 0.0F;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) total += partials[w][i];
        values[i] = total;
    }
}

// Grid (4, columns), 256 threads: one CTA owns one stream of one column. Writes the FP32 stream
// gate sigmoid(gate_s) and u = BF16(n(V, conv_norm)); V = gate * value is recomputed in FP32 by
// the convolution kernels.
__global__ void __launch_bounds__(kPleGateThreads)
    ple_gate_norm_kernel(const __nv_bfloat16* __restrict__ hidden,
                         const __nv_bfloat16* __restrict__ key,
                         const __nv_bfloat16* __restrict__ value,
                         const __nv_bfloat16* __restrict__ query_norm,
                         const __nv_bfloat16* __restrict__ key_norm,
                         const __nv_bfloat16* __restrict__ conv_norm,
                         float* __restrict__ gates, __nv_bfloat16* __restrict__ normalized) {
    constexpr int kWarps = kPleGateThreads / 32;
    __shared__ float squares[kWarps][2];
    __shared__ float dots[kWarps][1];
    __shared__ float gated_squares[kWarps][1];
    const int stream        = static_cast<int>(blockIdx.x);
    const int column        = static_cast<int>(blockIdx.y);
    const int tid           = static_cast<int>(threadIdx.x);
    const int stream_offset = stream * kPleStreamWidth;
    const auto wide  = static_cast<std::int64_t>(column) * kPleWidth + stream_offset;
    const auto narrow = static_cast<std::int64_t>(column) * kPleStreamWidth;

    float sums[2] = {0.0F, 0.0F};
    for (int d = tid; d < kPleStreamWidth; d += kPleGateThreads) {
        const float q = __bfloat162float(hidden[wide + d]);
        const float k = __bfloat162float(key[wide + d]);
        sums[0]       = fmaf(q, q, sums[0]);
        sums[1]       = fmaf(k, k, sums[1]);
    }
    ple_block_sum<kPleGateThreads>(sums, squares);
    const float inv_q = rsqrtf(sums[0] / static_cast<float>(kPleStreamWidth) + 1.0e-6F);
    const float inv_k = rsqrtf(sums[1] / static_cast<float>(kPleStreamWidth) + 1.0e-6F);

    float dot[1] = {0.0F};
    for (int d = tid; d < kPleStreamWidth; d += kPleGateThreads) {
        const float q = __bfloat162float(hidden[wide + d]) * inv_q *
                        (1.0F + __bfloat162float(query_norm[stream_offset + d]));
        const float k = __bfloat162float(key[wide + d]) * inv_k *
                        (1.0F + __bfloat162float(key_norm[stream_offset + d]));
        dot[0] = fmaf(q, k, dot[0]);
    }
    ple_block_sum<kPleGateThreads>(dot, dots);
    const float raw  = dot[0] * 0.019764235376052372F; // 1 / sqrt(2560)
    const float sign = raw > 0.0F ? 1.0F : (raw < 0.0F ? -1.0F : 0.0F);
    const float gate = sigmoid(sign * sqrtf(fmaxf(fabsf(raw), 1.0e-6F)));

    float square[1] = {0.0F};
    for (int d = tid; d < kPleStreamWidth; d += kPleGateThreads) {
        const float v = gate * __bfloat162float(value[narrow + d]);
        square[0]     = fmaf(v, v, square[0]);
    }
    ple_block_sum<kPleGateThreads>(square, gated_squares);
    const float inv_v = rsqrtf(square[0] / static_cast<float>(kPleStreamWidth) + 1.0e-6F);
    if (tid == 0) gates[static_cast<std::int64_t>(column) * kPleStreams + stream] = gate;
    for (int d = tid; d < kPleStreamWidth; d += kPleGateThreads) {
        const float v        = gate * __bfloat162float(value[narrow + d]);
        normalized[wide + d] = __float2bfloat16_rn(
            v * inv_v * (1.0F + __bfloat162float(conv_norm[stream_offset + d])));
    }
}

// V of one channel/column, bit-identical to the value the gate kernel normalized.
__device__ __forceinline__ float ple_value(const float* __restrict__ gates,
                                           const __nv_bfloat16* __restrict__ value,
                                           std::int64_t column, int channel) {
    return gates[column * kPleStreams + channel / kPleStreamWidth] *
           __bfloat162float(value[column * kPleStreamWidth + channel % kPleStreamWidth]);
}

__device__ __forceinline__ float ple_conv(const float (&taps)[4], const float (&weights)[4]) {
    float conv = fmaf(taps[0], weights[0], 0.0F);
    conv       = fmaf(taps[1], weights[1], conv);
    conv       = fmaf(taps[2], weights[2], conv);
    return fmaf(taps[3], weights[3], conv);
}

__device__ __forceinline__ void ple_load_weights(const __nv_bfloat16* __restrict__ conv_weight,
                                                 int channel, float (&weights)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) weights[i] = __bfloat162float(conv_weight[i * kPleWidth + channel]);
}

// Grid 40 * T, 256 threads: one thread per (channel, column). Reads history and in-call u only;
// the state update runs in the next launch.
__global__ void __launch_bounds__(kPleConvThreads)
    ple_conv_kernel(const float* __restrict__ gates, const __nv_bfloat16* __restrict__ value,
                    const __nv_bfloat16* __restrict__ normalized,
                    const __nv_bfloat16* __restrict__ conv_weight,
                    const __nv_bfloat16* __restrict__ state, __nv_bfloat16* __restrict__ out) {
    const int column  = static_cast<int>(blockIdx.x / kPleConvBlocks);
    const int channel = static_cast<int>(blockIdx.x % kPleConvBlocks) * kPleConvThreads +
                        static_cast<int>(threadIdx.x);
    float weights[4];
    ple_load_weights(conv_weight, channel, weights);
    float taps[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int tau = column - 9 + 3 * i;
        taps[i] = __bfloat162float(tau >= 0
                                       ? normalized[static_cast<std::int64_t>(tau) * kPleWidth + channel]
                                       : state[(kPleHistory + tau) * kPleWidth + channel]);
    }
    const auto at = static_cast<std::int64_t>(column) * kPleWidth + channel;
    out[at]       = __float2bfloat16_rn(ple_value(gates, value, column, channel) +
                                        silu(ple_conv(taps, weights)));
}

// Grid 40, 256 threads: the new nine-column window. Ascending i reads index T + i >= i + 1 of an
// aliased state before any later write reaches it.
__global__ void __launch_bounds__(kPleConvThreads)
    ple_state_kernel(const __nv_bfloat16* __restrict__ normalized, const __nv_bfloat16* state_in,
                     __nv_bfloat16* state_out, int columns) {
    const int channel = static_cast<int>(blockIdx.x) * kPleConvThreads + static_cast<int>(threadIdx.x);
#pragma unroll 1
    for (int i = 0; i < kPleHistory; ++i) {
        const int tau                        = columns - kPleHistory + i;
        const __nv_bfloat16 value            = tau >= 0
                                                   ? normalized[static_cast<std::int64_t>(tau) * kPleWidth + channel]
                                                   : state_in[(kPleHistory + tau) * kPleWidth + channel];
        state_out[i * kPleWidth + channel] = value;
    }
}

// Grid (40, B), 256 threads: row b scans its W columns from its initial slot, writing a snapshot
// after every valid column and BF16 zero for the invalid tail.
__global__ void __launch_bounds__(kPleConvThreads)
    ple_snapshot_kernel(const float* __restrict__ gates, const __nv_bfloat16* __restrict__ value,
                        const __nv_bfloat16* __restrict__ normalized,
                        const __nv_bfloat16* __restrict__ conv_weight, __nv_bfloat16* states,
                        const std::int32_t* __restrict__ valid_columns,
                        const std::int32_t* __restrict__ initial_slots,
                        const std::int32_t* __restrict__ snapshot_base_slots,
                        __nv_bfloat16* __restrict__ out, int width) {
    const int channel = static_cast<int>(blockIdx.x) * kPleConvThreads + static_cast<int>(threadIdx.x);
    const int row     = static_cast<int>(blockIdx.y);
    const int valid   = valid_columns != nullptr ? valid_columns[row] : width;
    float weights[4];
    ple_load_weights(conv_weight, channel, weights);
    constexpr std::int64_t kSlot = static_cast<std::int64_t>(kPleHistory) * kPleWidth;
    const __nv_bfloat16* initial = states + initial_slots[row] * kSlot + channel;
    __nv_bfloat16 window[kPleHistory];
#pragma unroll
    for (int i = 0; i < kPleHistory; ++i) window[i] = initial[i * kPleWidth];
    const std::int64_t base = static_cast<std::int64_t>(snapshot_base_slots[row]);
#pragma unroll 1
    for (int j = 0; j < width; ++j) {
        const auto at = (static_cast<std::int64_t>(row) * width + j) * kPleWidth + channel;
        if (j >= valid) {
            out[at] = __float2bfloat16_rn(0.0F);
            continue;
        }
        const __nv_bfloat16 u = normalized[at];
        const float taps[4]   = {__bfloat162float(window[0]), __bfloat162float(window[3]),
                                 __bfloat162float(window[6]), __bfloat162float(u)};
        const std::int64_t column = static_cast<std::int64_t>(row) * width + j;
        out[at] = __float2bfloat16_rn(ple_value(gates, value, column, channel) +
                                      silu(ple_conv(taps, weights)));
#pragma unroll
        for (int i = 0; i + 1 < kPleHistory; ++i) window[i] = window[i + 1];
        window[kPleHistory - 1] = u;
        __nv_bfloat16* snapshot = states + (base + j) * kSlot + channel;
#pragma unroll
        for (int i = 0; i < kPleHistory; ++i) snapshot[i * kPleWidth] = window[i];
    }
}

} // namespace ninfer::ops::detail
