#include "targets/qwen3_8_flash_next/impl/moe_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

using GateGeometry = ops::detail::Nvfp4GemvGeometry<1'280, 2'560>;
using GateSchedule =
    ops::detail::Nvfp4GemvSchedule<8, 2, 16, 4, ops::detail::Nvfp4ScaleAccess::Direct,
                                   ops::detail::Nvfp4CodeCache::Default, 1>;
using DownGeometry = ops::detail::Nvfp4GemvGeometry<2'560, 640>;

constexpr int kHidden       = 2'560;
constexpr int kIntermediate = 640;
constexpr int kTopK         = 10;
constexpr int kPaths        = kTopK + 1;
constexpr int kDownWarps    = 8;

template <int K>
__device__ __forceinline__ void dot_bf16_pair(const __nv_bfloat16* x, const __nv_bfloat16* first,
                                              const __nv_bfloat16* second, float& first_result,
                                              float& second_result) {
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    float first_sum  = 0.0F;
    float second_sum = 0.0F;
    for (int column = lane; column < K; column += 32) {
        const float value = __bfloat162float(x[column]);
        first_sum         = fmaf(__bfloat162float(first[column]), value, first_sum);
        second_sum        = fmaf(__bfloat162float(second[column]), value, second_sum);
    }
    first_result  = ops::warp_reduce_sum(first_sum);
    second_result = ops::warp_reduce_sum(second_sum);
}

template <int K>
__device__ __forceinline__ float dot_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float sum      = 0.0F;
    for (int column = lane; column < K; column += 32) {
        sum = fmaf(__bfloat162float(weight[column]), __bfloat162float(x[column]), sum);
    }
    return ops::warp_reduce_sum(sum);
}

__device__ __forceinline__ float dot_nvfp4_down(const __nv_bfloat16* x, const std::uint8_t* codes,
                                                const std::uint8_t* scales,
                                                float inverse_weight_divisor, int row) {
    constexpr int kGroupValues     = 16;
    constexpr int kGroupsPerRow    = kIntermediate / kGroupValues;
    constexpr int kCodeBytesPerRow = kIntermediate / 2;
    const int lane                 = static_cast<int>(threadIdx.x) & 31;
    float sum                      = 0.0F;
    for (int group = lane; group < kGroupsPerRow; group += 32) {
        const float coefficient =
            ops::detail::decode_nvfp4_e4m3(
                scales[ops::detail::nvfp4_scale_offset<DownGeometry>(row, group)]) *
            inverse_weight_divisor;
        const auto* packed = codes + static_cast<std::int64_t>(row) * kCodeBytesPerRow + group * 8;
        const auto* activation = x + group * kGroupValues;
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            const float2 code  = ops::detail::decode_nvfp4_e2m1x2(packed[pair]);
            const float2 value = ops::bf16x2_bits_to_float2(
                reinterpret_cast<const std::uint32_t*>(activation)[pair]);
            sum = fmaf(code.x * coefficient, value.x, sum);
            sum = fmaf(code.y * coefficient, value.y, sum);
        }
    }
    return ops::warp_reduce_sum(sum);
}

__global__ void flash_next_moe_gate_up_kernel(
    const __nv_bfloat16* __restrict__ input, const std::int32_t* __restrict__ ids,
    const std::uint8_t* __restrict__ expert_codes, const std::uint8_t* __restrict__ expert_scales,
    const float* __restrict__ expert_divisors, std::uint64_t code_stride,
    std::uint64_t scale_stride, const __nv_bfloat16* __restrict__ shared_gate,
    const __nv_bfloat16* __restrict__ shared_up, __nv_bfloat16* __restrict__ activations) {
    __shared__ ops::detail::Nvfp4GemvSharedStorage<GateGeometry, GateSchedule> shared;
    const int token = static_cast<int>(blockIdx.y);
    const int path  = static_cast<int>(blockIdx.z);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int row   = static_cast<int>(blockIdx.x) * GateSchedule::kWarpsPerCta + warp;
    if (row >= kIntermediate) { return; }
    const auto* x = input + static_cast<std::int64_t>(token) * kHidden;
    float gate    = 0.0F;
    float up      = 0.0F;
    if (path < kTopK) {
        const int expert    = ids[token * kTopK + path];
        const auto* codes   = expert_codes + static_cast<std::uint64_t>(expert) * code_stride;
        const auto* scales  = expert_scales + static_cast<std::uint64_t>(expert) * scale_stride;
        const float inverse = 1.0F / expert_divisors[expert];
        const int parent_rows[GateSchedule::kRowsPerWarp] = {row, row + kIntermediate};
        float accumulators[GateSchedule::kRowsPerWarp][GateSchedule::kAccumulatorChains] = {};
        ops::detail::compute_nvfp4_rows<GateGeometry, GateSchedule>(
            x, codes, scales, shared, inverse, parent_rows, warp * GateSchedule::kRowsPerWarp, lane,
            accumulators);
#pragma unroll
        for (int chain = 0; chain < GateSchedule::kAccumulatorChains; ++chain) {
            gate += accumulators[0][chain];
            up += accumulators[1][chain];
        }
        gate = ops::warp_reduce_sum(gate);
        up   = ops::warp_reduce_sum(up);
    } else {
        dot_bf16_pair<kHidden>(x, shared_gate + static_cast<std::int64_t>(row) * kHidden,
                               shared_up + static_cast<std::int64_t>(row) * kHidden, gate, up);
    }
    if (lane == 0) {
        activations[(static_cast<std::int64_t>(token) * kPaths + path) * kIntermediate + row] =
            __float2bfloat16_rn(ops::silu(gate) * up);
    }
}

__global__ void flash_next_moe_down_kernel(
    const std::int32_t* __restrict__ ids, const float* __restrict__ alpha,
    const float* __restrict__ shared_scale, const __nv_bfloat16* __restrict__ activations,
    const std::uint8_t* __restrict__ expert_codes, const std::uint8_t* __restrict__ expert_scales,
    const float* __restrict__ expert_divisors, std::uint64_t code_stride,
    std::uint64_t scale_stride, const __nv_bfloat16* __restrict__ shared_down,
    __nv_bfloat16* __restrict__ output) {
    const int token = static_cast<int>(blockIdx.y);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int row   = static_cast<int>(blockIdx.x) * kDownWarps + warp;
    if (row >= kHidden) { return; }

    float routed = 0.0F;
#pragma unroll
    for (int path = 0; path < kTopK; ++path) {
        const int expert = ids[token * kTopK + path];
        const auto* x =
            activations + (static_cast<std::int64_t>(token) * kPaths + path) * kIntermediate;
        const auto* codes   = expert_codes + static_cast<std::uint64_t>(expert) * code_stride;
        const auto* scales  = expert_scales + static_cast<std::uint64_t>(expert) * scale_stride;
        const float inverse = 1.0F / expert_divisors[expert];
        const float value   = dot_nvfp4_down(x, codes, scales, inverse, row);
        if (lane == 0) { routed = fmaf(alpha[token * kTopK + path], value, routed); }
    }
    const auto* shared_x =
        activations + (static_cast<std::int64_t>(token) * kPaths + kTopK) * kIntermediate;
    const float shared_value = dot_bf16<kIntermediate>(
        shared_x, shared_down + static_cast<std::int64_t>(row) * kIntermediate);
    if (lane == 0) {
        output[static_cast<std::int64_t>(token) * kHidden + row] =
            __float2bfloat16_rn(fmaf(shared_scale[token], shared_value, routed));
    }
}

// =========================================================================
// PREFILL KERNELS (T >= 128, Amortized Weight Loading & Grouped GEMMs)
// =========================================================================

// Step 1: Deterministic Expert Histogram, Prefix Scan & Token Grouping
// Step 1: Deterministic Expert Histogram, Prefix Scan, Token Grouping & Active Expert List
// CTA: 1 CTA = 512 threads (16 warps).
__global__ void flash_next_moe_prefill_build_groups_kernel(
    const std::int32_t* __restrict__ ids,
    std::int32_t* __restrict__ expert_counts,
    std::int32_t* __restrict__ expert_offsets,
    std::int32_t* __restrict__ active_experts,
    std::int32_t* __restrict__ active_count_ptr,
    std::int32_t* __restrict__ grouped_tokens,
    std::int32_t* __restrict__ grouped_paths,
    std::int32_t* __restrict__ grouped_experts,
    std::int32_t* __restrict__ task_counter,
    int tokens) {
    const int tid = static_cast<int>(threadIdx.x);
    if (tid >= 512) { return; }

    const int total_items = tokens * kTopK;

    // 1. Histogram: count items for expert 'tid'
    int count = 0;
    for (int i = 0; i < total_items; ++i) {
        if (ids[i] == tid) {
            count++;
        }
    }
    expert_counts[tid] = count;

    // 2. Deterministic prefix scan for expert offsets
    const int lane = tid & 31;
    const int warp = tid >> 5;

    int warp_sum = count;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int val = __shfl_up_sync(0xffffffff, warp_sum, offset);
        if (lane >= offset) warp_sum += val;
    }

    __shared__ int s_warp_totals[16];
    if (lane == 31) { s_warp_totals[warp] = warp_sum; }
    __syncthreads();

    if (warp == 0 && lane < 16) {
        int w_sum = s_warp_totals[lane];
        #pragma unroll
        for (int offset = 1; offset < 16; offset <<= 1) {
            const int val = __shfl_up_sync(0xffff, w_sum, offset);
            if (lane >= offset) w_sum += val;
        }
        s_warp_totals[lane] = w_sum;
    }
    __syncthreads();

    const int base_offset = (warp > 0) ? s_warp_totals[warp - 1] : 0;
    const int exc_offset  = base_offset + warp_sum - count;
    expert_offsets[tid]   = exc_offset;

    if (tid == 511) {
        expert_offsets[512] = exc_offset + count;
    }

    // 3. Compact active experts list using prefix scan of (count > 0)
    const int is_active = (count > 0) ? 1 : 0;
    int act_warp_sum = is_active;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int val = __shfl_up_sync(0xffffffff, act_warp_sum, offset);
        if (lane >= offset) act_warp_sum += val;
    }

    __shared__ int s_act_totals[16];
    if (lane == 31) { s_act_totals[warp] = act_warp_sum; }
    __syncthreads();

    if (warp == 0 && lane < 16) {
        int w_sum = s_act_totals[lane];
        #pragma unroll
        for (int offset = 1; offset < 16; offset <<= 1) {
            const int val = __shfl_up_sync(0xffff, w_sum, offset);
            if (lane >= offset) w_sum += val;
        }
        s_act_totals[lane] = w_sum;
    }
    __syncthreads();

    const int act_base_offset = (warp > 0) ? s_act_totals[warp - 1] : 0;
    const int act_exc_offset  = act_base_offset + act_warp_sum - is_active;
    if (is_active) {
        active_experts[act_exc_offset] = tid;
    }

    if (tid == 511) {
        active_count_ptr[0] = act_exc_offset + is_active;
        task_counter[0]     = 0;
        task_counter[1]     = 0;
        task_counter[2]     = 0;
        task_counter[3]     = 0;
    }

    // 4. Deterministic O(N) cooperative placement using shared memory integer atomic offsets
    __shared__ int s_heads[512];
    s_heads[tid] = exc_offset;
    __syncthreads();

    for (int i = tid; i < total_items; i += 512) {
        const int expert = ids[i];
        const int pos    = atomicAdd(&s_heads[expert], 1);
        grouped_tokens[pos]  = i / kTopK;
        grouped_paths[pos]   = i % kTopK;
        grouped_experts[pos] = expert;
    }
}

// Step 2: Grouped Expert Gate & Up Projection (Grid-Stride Active Experts GEMM)
// CTA: 256 threads (8 warps). 8 pairs per CTA (8 warps x 1 pair).
// Grid: (kIntermediate / 8, kGridY) = (80, 16).
__global__ __launch_bounds__(256, 4) void flash_next_moe_prefill_gate_up_kernel(
    const __nv_bfloat16* __restrict__ input,
    const std::int32_t* __restrict__ expert_offsets,
    const std::int32_t* __restrict__ expert_counts,
    const std::int32_t* __restrict__ active_experts,
    const std::int32_t* __restrict__ active_count_ptr,
    const std::int32_t* __restrict__ grouped_tokens,
    const std::int32_t* __restrict__ grouped_paths,
    const std::uint8_t* __restrict__ expert_codes,
    const std::uint8_t* __restrict__ expert_scales,
    const float* __restrict__ expert_divisors,
    std::uint64_t code_stride,
    std::uint64_t scale_stride,
    __nv_bfloat16* __restrict__ activations) {
    const int total_active = active_count_ptr[0];
    const int tid          = static_cast<int>(threadIdx.x);
    const int warp         = tid >> 5;
    const int lane         = tid & 31;
    const int local_pair   = warp;
    const int row_base     = static_cast<int>(blockIdx.x) * 8 + local_pair;
    if (row_base >= kIntermediate) { return; }

    constexpr int kBatch = 4;
    __shared__ uint4 s_input[kBatch][320];          // 20.48 KB
    __shared__ std::uint8_t s_gate_scales[16][160]; // 2.56 KB

    for (int act_idx = static_cast<int>(blockIdx.y); act_idx < total_active; act_idx += static_cast<int>(gridDim.y)) {
        const int expert      = active_experts[act_idx];
        const int token_count = expert_counts[expert];
        if (token_count <= 0) { continue; }

        const int start_idx = expert_offsets[expert];

        const auto* codes_base  = expert_codes + static_cast<std::uint64_t>(expert) * code_stride;
        const auto* scales_base = expert_scales + static_cast<std::uint64_t>(expert) * scale_stride;
        const float inv_divisor = 1.0F / expert_divisors[expert];

        // 1. Cooperative coalesced scale staging for 16 rows x 40 tiles = 640 tasks
        for (int task = tid; task < 640; task += 256) {
            const int tile     = task >> 4;
            const int r        = task & 15;
            const int is_up    = r >= 8 ? 1 : 0;
            const int pair_idx = static_cast<int>(blockIdx.x) * 8 + (r & 7);
            const int row      = is_up ? (pair_idx + kIntermediate) : pair_idx;

            const int m_tile       = row / 128;
            const int row_inner    = row % 128;
            const int row_mod32    = row_inner & 31;
            const int row_quartile = row_inner >> 5;
            const std::int64_t off = static_cast<std::int64_t>(m_tile * 40 + tile) * 512 +
                                     row_mod32 * 16 + row_quartile * 4;

            const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(scales_base + off);
            *reinterpret_cast<std::uint32_t*>(&s_gate_scales[r][tile * 4]) = word;
        }
        __syncthreads();

        const auto* g_row_ptr = codes_base + static_cast<std::int64_t>(row_base) * (kHidden / 2);
        const auto* u_row_ptr = codes_base + static_cast<std::int64_t>(row_base + kIntermediate) * (kHidden / 2);

        for (int token_base = 0; token_base < token_count; token_base += kBatch) {
            const int batch_tokens = min(kBatch, token_count - token_base);

            // 2. Cooperative single stage of ALL 2560 input channels for up to 4 tokens
            for (int i = tid; i < batch_tokens * 320; i += 256) {
                const int b_idx = i / 320;
                const int u4    = i % 320;
                const int token = grouped_tokens[start_idx + token_base + b_idx];
                const auto* src = input + static_cast<std::int64_t>(token) * kHidden + u4 * 8;
                s_input[b_idx][u4] = *reinterpret_cast<const uint4*>(src);
            }
            __syncthreads();

            float gate_acc[kBatch] = {};
            float up_acc[kBatch]   = {};

            #pragma unroll
            for (int phase = 0; phase < 5; ++phase) {
                const int group = phase * 32 + lane;
                const auto* g_packed = g_row_ptr + group * 8;
                const auto* u_packed = u_row_ptr + group * 8;
                const uint2 g_words  = *reinterpret_cast<const uint2*>(g_packed);
                const uint2 u_words  = *reinterpret_cast<const uint2*>(u_packed);

                const float g_coeff = ops::detail::decode_nvfp4_e4m3(
                    s_gate_scales[local_pair][group]) * inv_divisor;
                const float u_coeff = ops::detail::decode_nvfp4_e4m3(
                    s_gate_scales[8 + local_pair][group]) * inv_divisor;

                float2 g_raw[8];
                float2 u_raw[8];
                #pragma unroll
                for (int pair = 0; pair < 8; ++pair) {
                    const std::uint32_t g_w = (pair < 4) ? g_words.x : g_words.y;
                    const std::uint32_t u_w = (pair < 4) ? u_words.x : u_words.y;
                    const std::uint8_t gp   = static_cast<std::uint8_t>(g_w >> (8 * (pair & 3)));
                    const std::uint8_t up   = static_cast<std::uint8_t>(u_w >> (8 * (pair & 3)));
                    g_raw[pair] = ops::detail::decode_nvfp4_e2m1x2(gp);
                    u_raw[pair] = ops::detail::decode_nvfp4_e2m1x2(up);
                }

                #pragma unroll
                for (int b = 0; b < kBatch; ++b) {
                    if (b < batch_tokens) {
                        const auto x_v0 = s_input[b][group * 2];
                        const auto x_v1 = s_input[b][group * 2 + 1];
                        const std::uint32_t x_raw[8] = {
                            x_v0.x, x_v0.y, x_v0.z, x_v0.w,
                            x_v1.x, x_v1.y, x_v1.z, x_v1.w
                        };

                        float g_sum_raw = 0.0F;
                        float u_sum_raw = 0.0F;
                        #pragma unroll
                        for (int pair = 0; pair < 8; ++pair) {
                            const float2 xv = ops::bf16x2_bits_to_float2(x_raw[pair]);
                            g_sum_raw = fmaf(g_raw[pair].x, xv.x, g_sum_raw);
                            g_sum_raw = fmaf(g_raw[pair].y, xv.y, g_sum_raw);
                            u_sum_raw = fmaf(u_raw[pair].x, xv.x, u_sum_raw);
                            u_sum_raw = fmaf(u_raw[pair].y, xv.y, u_sum_raw);
                        }
                        gate_acc[b] = fmaf(g_sum_raw, g_coeff, gate_acc[b]);
                        up_acc[b]   = fmaf(u_sum_raw, u_coeff, up_acc[b]);
                    }
                }
            }

            #pragma unroll
            for (int b = 0; b < kBatch; ++b) {
                if (b < batch_tokens) {
                    const int token = grouped_tokens[start_idx + token_base + b];
                    const int path  = grouped_paths[start_idx + token_base + b];

                    const float gate_sum = ops::warp_reduce_sum(gate_acc[b]);
                    const float up_sum   = ops::warp_reduce_sum(up_acc[b]);
                    if (lane == 0) {
                        const float act = ops::silu(gate_sum) * up_sum;
                        activations[(static_cast<std::int64_t>(token) * kPaths + path) * kIntermediate + row_base] =
                            __float2bfloat16_rn(act);
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

// Step 3: Shared Expert Gate & Up Projection (Vectorized 128-bit loads)
// CTA: 256 threads (8 warps). Each warp processes 1 pair -> 8 pairs per CTA.
// Grid.x = 640 / 8 = 80, Grid.y = (tokens + 7) / 8.
__global__ void flash_next_moe_prefill_shared_gate_up_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ shared_gate,
    const __nv_bfloat16* __restrict__ shared_up,
    __nv_bfloat16* __restrict__ activations,
    int tokens) {
    const int warp         = static_cast<int>(threadIdx.x) >> 5;
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int row          = static_cast<int>(blockIdx.x) * 8 + warp;
    const int token_base   = static_cast<int>(blockIdx.y) * 8;
    const int batch_tokens = min(8, tokens - token_base);

    if (row >= kIntermediate || batch_tokens <= 0) { return; }

    const auto* gate_w = shared_gate + static_cast<std::int64_t>(row) * kHidden;
    const auto* up_w   = shared_up + static_cast<std::int64_t>(row) * kHidden;

    float gate_acc[8] = {};
    float up_acc[8]   = {};

    for (int col_base = 0; col_base < kHidden; col_base += 256) {
        const int col = col_base + lane * 8;
        const auto gw_v = *reinterpret_cast<const uint4*>(&gate_w[col]);
        const auto uw_v = *reinterpret_cast<const uint4*>(&up_w[col]);
        const std::uint32_t gw_raw[4] = {gw_v.x, gw_v.y, gw_v.z, gw_v.w};
        const std::uint32_t uw_raw[4] = {uw_v.x, uw_v.y, uw_v.z, uw_v.w};

        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            if (b < batch_tokens) {
                const int t_idx = token_base + b;
                const auto x_v  = *reinterpret_cast<const uint4*>(&input[static_cast<std::int64_t>(t_idx) * kHidden + col]);
                const std::uint32_t x_raw[4] = {x_v.x, x_v.y, x_v.z, x_v.w};

                #pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const float2 g_pair = ops::bf16x2_bits_to_float2(gw_raw[pair]);
                    const float2 u_pair = ops::bf16x2_bits_to_float2(uw_raw[pair]);
                    const float2 x_pair = ops::bf16x2_bits_to_float2(x_raw[pair]);

                    gate_acc[b] = fmaf(g_pair.x, x_pair.x, gate_acc[b]);
                    gate_acc[b] = fmaf(g_pair.y, x_pair.y, gate_acc[b]);
                    up_acc[b]   = fmaf(u_pair.x, x_pair.x, up_acc[b]);
                    up_acc[b]   = fmaf(u_pair.y, x_pair.y, up_acc[b]);
                }
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        if (b < batch_tokens) {
            const float gate_sum = ops::warp_reduce_sum(gate_acc[b]);
            const float up_sum   = ops::warp_reduce_sum(up_acc[b]);
            if (lane == 0) {
                const int t_idx = token_base + b;
                const float act = ops::silu(gate_sum) * up_sum;
                activations[(static_cast<std::int64_t>(t_idx) * kPaths + kTopK) * kIntermediate + row] =
                    __float2bfloat16_rn(act);
            }
        }
    }
}

// Step 4: Grouped Expert Down Projection (Grid-Stride Active Experts GEMM)
// CTA: 256 threads (8 warps). 16 rows per CTA (8 warps x 2 rows).
// Grid: (kHidden / 16, kGridY) = (160, 8).
__global__ __launch_bounds__(256, 4) void flash_next_moe_prefill_down_kernel(
    const __nv_bfloat16* __restrict__ activations,
    const std::int32_t* __restrict__ expert_offsets,
    const std::int32_t* __restrict__ expert_counts,
    const std::int32_t* __restrict__ active_experts,
    const std::int32_t* __restrict__ active_count_ptr,
    const std::int32_t* __restrict__ grouped_tokens,
    const std::int32_t* __restrict__ grouped_paths,
    const std::uint8_t* __restrict__ expert_codes,
    const std::uint8_t* __restrict__ expert_scales,
    const float* __restrict__ expert_divisors,
    std::uint64_t code_stride,
    std::uint64_t scale_stride,
    float* __restrict__ down_intermediate) {
    const int total_active = active_count_ptr[0];
    const int tid          = static_cast<int>(threadIdx.x);
    const int warp         = tid >> 5;
    const int lane         = tid & 31;
    const int local_row    = warp * 2;
    const int row_base     = static_cast<int>(blockIdx.x) * 16 + local_row;
    if (row_base >= kHidden) { return; }

    constexpr int kBatch = 4;
    __shared__ uint4 s_act[kBatch][80];            // 5.12 KB
    __shared__ std::uint8_t s_down_scales[16][40]; // 640 bytes

    for (int act_idx = static_cast<int>(blockIdx.y); act_idx < total_active; act_idx += static_cast<int>(gridDim.y)) {
        const int expert      = active_experts[act_idx];
        const int token_count = expert_counts[expert];
        if (token_count <= 0) { continue; }

        const int start_idx = expert_offsets[expert];

        const auto* codes_base  = expert_codes + static_cast<std::uint64_t>(expert) * code_stride;
        const auto* scales_base = expert_scales + static_cast<std::uint64_t>(expert) * scale_stride;
        const float inv_divisor = 1.0F / expert_divisors[expert];

        // 1. Cooperative coalesced scale staging for 16 rows x 10 tiles = 160 tasks
        for (int task = tid; task < 160; task += 256) {
            const int tile = task >> 4;
            const int r    = task & 15;
            const int row  = static_cast<int>(blockIdx.x) * 16 + r;

            const int m_tile       = row / 128;
            const int row_inner    = row % 128;
            const int row_mod32    = row_inner & 31;
            const int row_quartile = row_inner >> 5;
            const std::int64_t off = static_cast<std::int64_t>(m_tile * 10 + tile) * 512 +
                                     row_mod32 * 16 + row_quartile * 4;

            const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(scales_base + off);
            *reinterpret_cast<std::uint32_t*>(&s_down_scales[r][tile * 4]) = word;
        }
        __syncthreads();

        for (int token_base = 0; token_base < token_count; token_base += kBatch) {
            const int batch_tokens = min(kBatch, token_count - token_base);

            // 2. Cooperative stage of ALL 640 intermediate activations for up to 4 tokens
            for (int i = tid; i < batch_tokens * 80; i += 256) {
                const int b_idx = i / 80;
                const int u4    = i % 80;
                const int token = grouped_tokens[start_idx + token_base + b_idx];
                const int path  = grouped_paths[start_idx + token_base + b_idx];
                const auto* src = activations + (static_cast<std::int64_t>(token) * kPaths + path) * kIntermediate + u4 * 8;
                s_act[b_idx][u4] = *reinterpret_cast<const uint4*>(src);
            }
            __syncthreads();

            float down_acc[kBatch][2] = {};

            for (int group = lane; group < 40; group += 32) {
                float2 dw[2][8];
                #pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const int row = row_base + r;
                    if (row < kHidden) {
                        const auto* down_packed = codes_base + static_cast<std::int64_t>(row) * (kIntermediate / 2) + group * 8;
                        const uint2 d_words     = *reinterpret_cast<const uint2*>(down_packed);

                        #pragma unroll
                        for (int pair = 0; pair < 8; ++pair) {
                            const std::uint32_t d_w = (pair < 4) ? d_words.x : d_words.y;
                            const std::uint8_t dp   = static_cast<std::uint8_t>(d_w >> (8 * (pair & 3)));
                            dw[r][pair]             = ops::detail::decode_nvfp4_e2m1x2(dp);
                        }
                    }
                }

                #pragma unroll
                for (int b = 0; b < kBatch; ++b) {
                    if (b < batch_tokens) {
                        const auto h_v0 = s_act[b][group * 2];
                        const auto h_v1 = s_act[b][group * 2 + 1];
                        const std::uint32_t h_raw[8] = {
                            h_v0.x, h_v0.y, h_v0.z, h_v0.w,
                            h_v1.x, h_v1.y, h_v1.z, h_v1.w
                        };

                        #pragma unroll
                        for (int r = 0; r < 2; ++r) {
                            if (row_base + r < kHidden) {
                                const float down_coeff = ops::detail::decode_nvfp4_e4m3(
                                    s_down_scales[local_row + r][group]) * inv_divisor;
                                float d_sum_raw = 0.0F;
                                #pragma unroll
                                for (int pair = 0; pair < 8; ++pair) {
                                    const float2 hv = ops::bf16x2_bits_to_float2(h_raw[pair]);
                                    d_sum_raw = fmaf(dw[r][pair].x, hv.x, d_sum_raw);
                                    d_sum_raw = fmaf(dw[r][pair].y, hv.y, d_sum_raw);
                                }
                                down_acc[b][r] = fmaf(d_sum_raw, down_coeff, down_acc[b][r]);
                            }
                        }
                    }
                }
            }

            #pragma unroll
            for (int b = 0; b < kBatch; ++b) {
                if (b < batch_tokens) {
                    const int token = grouped_tokens[start_idx + token_base + b];
                    const int path  = grouped_paths[start_idx + token_base + b];

                    #pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        const int row = row_base + r;
                        if (row < kHidden) {
                            const float down_sum = ops::warp_reduce_sum(down_acc[b][r]);
                            if (lane == 0) {
                                down_intermediate[(static_cast<std::int64_t>(token) * kHidden + row) * kTopK + path] =
                                    down_sum;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

// Step 5: Prefill Shared Expert Down & Routed Path Reduction (Vectorized 128-bit loads)
// CTA: 256 threads (8 warps). Each warp processes 1 row -> 8 rows per CTA.
// Grid.x = 2560 / 8 = 320, Grid.y = (tokens + 7) / 8.
__global__ void flash_next_moe_prefill_down_reduce_kernel(
    const std::int32_t* __restrict__ ids,
    const float* __restrict__ alpha,
    const float* __restrict__ shared_scale,
    const __nv_bfloat16* __restrict__ activations,
    const float* __restrict__ down_intermediate,
    const __nv_bfloat16* __restrict__ shared_down,
    __nv_bfloat16* __restrict__ output,
    int tokens) {
    const int warp         = static_cast<int>(threadIdx.x) >> 5;
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int row          = static_cast<int>(blockIdx.x) * 8 + warp;
    const int token_base   = static_cast<int>(blockIdx.y) * 8;
    const int batch_tokens = min(8, tokens - token_base);

    if (row >= kHidden || batch_tokens <= 0) { return; }

    const auto* sd_row = shared_down + static_cast<std::int64_t>(row) * kIntermediate;
    float shared_val[8] = {};

    for (int col_base = 0; col_base < kIntermediate; col_base += 256) {
        const int col = col_base + lane * 8;
        if (col < kIntermediate) {
            const auto sd_v = *reinterpret_cast<const uint4*>(&sd_row[col]);
            const std::uint32_t sd_raw[4] = {sd_v.x, sd_v.y, sd_v.z, sd_v.w};

            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                if (b < batch_tokens) {
                    const int token      = token_base + b;
                    const auto* shared_x = activations + (static_cast<std::int64_t>(token) * kPaths + kTopK) * kIntermediate;
                    const auto sx_v      = *reinterpret_cast<const uint4*>(&shared_x[col]);
                    const std::uint32_t sx_raw[4] = {sx_v.x, sx_v.y, sx_v.z, sx_v.w};

                    #pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        const float2 s_pair = ops::bf16x2_bits_to_float2(sd_raw[pair]);
                        const float2 x_pair = ops::bf16x2_bits_to_float2(sx_raw[pair]);
                        shared_val[b] = fmaf(s_pair.x, x_pair.x, shared_val[b]);
                        shared_val[b] = fmaf(s_pair.y, x_pair.y, shared_val[b]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        if (b < batch_tokens) {
            const int token   = token_base + b;
            const float s_val = ops::warp_reduce_sum(shared_val[b]);

            // Parallel coalesced load of the 10 paths across lanes 0..9
            float p_val = 0.0F;
            if (lane < kTopK) {
                p_val = alpha[token * kTopK + lane] * down_intermediate[(static_cast<std::int64_t>(token) * kHidden + row) * kTopK + lane];
            }

            // All threads in the warp participate in the collective shuffle
            float routed = p_val;
            #pragma unroll
            for (int src = 1; src < kTopK; ++src) {
                const float v = __shfl_sync(0xffffffff, p_val, src);
                if (lane == 0) {
                    routed += v;
                }
            }

            if (lane == 0) {
                const float s_scale = shared_scale[token];
                const float total   = routed + s_val * s_scale;
                output[static_cast<std::int64_t>(token) * kHidden + row] = __float2bfloat16_rn(total);
            }
        }
    }
}

__global__ void flash_next_moe_bf16_gate_up_kernel(
    const __nv_bfloat16* __restrict__ input, const std::int32_t* __restrict__ ids,
    const __nv_bfloat16* __restrict__ expert_gate_up, std::uint64_t expert_stride,
    const __nv_bfloat16* __restrict__ shared_gate, const __nv_bfloat16* __restrict__ shared_up,
    __nv_bfloat16* __restrict__ activations) {
    const int token = static_cast<int>(blockIdx.y);
    const int path  = static_cast<int>(blockIdx.z);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int row   = static_cast<int>(blockIdx.x) * GateSchedule::kWarpsPerCta + warp;
    if (row >= kIntermediate) { return; }
    const auto* x = input + static_cast<std::int64_t>(token) * kHidden;
    float gate    = 0.0F;
    float up      = 0.0F;
    if (path < kTopK) {
        const int expert    = ids[token * kTopK + path];
        const auto* exp_ptr = expert_gate_up + static_cast<std::uint64_t>(expert) * expert_stride;
        const auto* gate_w  = exp_ptr + static_cast<std::int64_t>(row) * kHidden;
        const auto* up_w    = exp_ptr + static_cast<std::int64_t>(row + kIntermediate) * kHidden;
        dot_bf16_pair<kHidden>(x, gate_w, up_w, gate, up);
    } else {
        dot_bf16_pair<kHidden>(x, shared_gate + static_cast<std::int64_t>(row) * kHidden,
                               shared_up + static_cast<std::int64_t>(row) * kHidden, gate, up);
    }
    if (lane == 0) {
        activations[(static_cast<std::int64_t>(token) * kPaths + path) * kIntermediate + row] =
            __float2bfloat16_rn(ops::silu(gate) * up);
    }
}

__global__ void flash_next_moe_bf16_down_kernel(
    const std::int32_t* __restrict__ ids, const float* __restrict__ alpha,
    const float* __restrict__ shared_scale, const __nv_bfloat16* __restrict__ activations,
    const __nv_bfloat16* __restrict__ expert_down, std::uint64_t expert_stride,
    const __nv_bfloat16* __restrict__ shared_down, __nv_bfloat16* __restrict__ output) {
    const int token = static_cast<int>(blockIdx.y);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int row   = static_cast<int>(blockIdx.x) * kDownWarps + warp;
    if (row >= kHidden) { return; }

    float routed = 0.0F;
#pragma unroll
    for (int path = 0; path < kTopK; ++path) {
        const int expert = ids[token * kTopK + path];
        const auto* x =
            activations + (static_cast<std::int64_t>(token) * kPaths + path) * kIntermediate;
        const auto* down_w = expert_down + static_cast<std::uint64_t>(expert) * expert_stride +
                             static_cast<std::int64_t>(row) * kIntermediate;
        const float value = dot_bf16<kIntermediate>(x, down_w);
        if (lane == 0) { routed = fmaf(alpha[token * kTopK + path], value, routed); }
    }
    const auto* shared_x =
        activations + (static_cast<std::int64_t>(token) * kPaths + kTopK) * kIntermediate;
    const float shared_value = dot_bf16<kIntermediate>(
        shared_x, shared_down + static_cast<std::int64_t>(row) * kIntermediate);
    if (lane == 0) {
        output[static_cast<std::int64_t>(token) * kHidden + row] =
            __float2bfloat16_rn(fmaf(shared_scale[token], shared_value, routed));
    }
}

} // namespace

void flash_next_moe_kernels_launch(const Tensor& input, const MoeWeights& weights,
                                   const FlashNextMoeWorkspace& workspace, Tensor& output,
                                   cudaStream_t stream) {
    const int tokens = static_cast<int>(input.ne[1]);

    if (tokens < 128) {
        // Decode path (T < 128): UNTOUCHED!
        const dim3 gate_grid(kIntermediate / GateSchedule::kWarpsPerCta,
                             static_cast<unsigned>(tokens), kPaths);
        flash_next_moe_gate_up_kernel<<<gate_grid, GateSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(input.data),
            static_cast<const std::int32_t*>(workspace.ids.data),
            reinterpret_cast<const std::uint8_t*>(weights.expert_gate_up.codes),
            reinterpret_cast<const std::uint8_t*>(weights.expert_gate_up.scales),
            weights.expert_gate_up.weight_scale_divisors,
            weights.expert_gate_up.code_bytes_per_expert,
            weights.expert_gate_up.scale_bytes_per_expert,
            static_cast<const __nv_bfloat16*>(weights.shared_gate.qdata),
            static_cast<const __nv_bfloat16*>(weights.shared_up.qdata),
            static_cast<__nv_bfloat16*>(workspace.activations.data));
        CUDA_CHECK(cudaGetLastError());

        const dim3 down_grid(kHidden / kDownWarps, static_cast<unsigned>(tokens));
        flash_next_moe_down_kernel<<<down_grid, kDownWarps * 32, 0, stream>>>(
            static_cast<const std::int32_t*>(workspace.ids.data),
            static_cast<const float*>(workspace.alpha.data),
            static_cast<const float*>(workspace.shared_scale.data),
            static_cast<const __nv_bfloat16*>(workspace.activations.data),
            reinterpret_cast<const std::uint8_t*>(weights.expert_down.codes),
            reinterpret_cast<const std::uint8_t*>(weights.expert_down.scales),
            weights.expert_down.weight_scale_divisors,
            weights.expert_down.code_bytes_per_expert,
            weights.expert_down.scale_bytes_per_expert,
            static_cast<const __nv_bfloat16*>(weights.shared_down.qdata),
            static_cast<__nv_bfloat16*>(output.data));
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Prefill path (T >= 128): Group tokens by expert, load weights once per chunk
        // 1. Group tokens by expert and build active expert list (1 CTA scan)
        flash_next_moe_prefill_build_groups_kernel<<<1, 512, 0, stream>>>(
            static_cast<const std::int32_t*>(workspace.ids.data),
            static_cast<std::int32_t*>(workspace.expert_counts.data),
            static_cast<std::int32_t*>(workspace.expert_offsets.data),
            static_cast<std::int32_t*>(workspace.active_expert_ids.data),
            static_cast<std::int32_t*>(workspace.active_count.data),
            static_cast<std::int32_t*>(workspace.grouped_tokens.data),
            static_cast<std::int32_t*>(workspace.grouped_paths.data),
            static_cast<std::int32_t*>(workspace.grouped_experts.data),
            static_cast<std::int32_t*>(workspace.task_counter.data),
            tokens);
        CUDA_CHECK(cudaGetLastError());

        // 2. Grouped Expert Gate & Up
        const dim3 gate_grid(kIntermediate / 8, 16);
        flash_next_moe_prefill_gate_up_kernel<<<gate_grid, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(input.data),
            static_cast<const std::int32_t*>(workspace.expert_offsets.data),
            static_cast<const std::int32_t*>(workspace.expert_counts.data),
            static_cast<const std::int32_t*>(workspace.active_expert_ids.data),
            static_cast<const std::int32_t*>(workspace.active_count.data),
            static_cast<const std::int32_t*>(workspace.grouped_tokens.data),
            static_cast<const std::int32_t*>(workspace.grouped_paths.data),
            reinterpret_cast<const std::uint8_t*>(weights.expert_gate_up.codes),
            reinterpret_cast<const std::uint8_t*>(weights.expert_gate_up.scales),
            weights.expert_gate_up.weight_scale_divisors,
            weights.expert_gate_up.code_bytes_per_expert,
            weights.expert_gate_up.scale_bytes_per_expert,
            static_cast<__nv_bfloat16*>(workspace.activations.data));
        CUDA_CHECK(cudaGetLastError());

        // 3. Shared expert gate & up (8 pairs per CTA -> 80 blocks)
        const dim3 shared_grid(kIntermediate / 8, (static_cast<unsigned>(tokens) + 7) / 8);
        flash_next_moe_prefill_shared_gate_up_kernel<<<shared_grid, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(input.data),
            static_cast<const __nv_bfloat16*>(weights.shared_gate.qdata),
            static_cast<const __nv_bfloat16*>(weights.shared_up.qdata),
            static_cast<__nv_bfloat16*>(workspace.activations.data),
            tokens);
        CUDA_CHECK(cudaGetLastError());

        // 4. Grouped Down GEMM
        const dim3 down_grid(kHidden / 16, 8);
        flash_next_moe_prefill_down_kernel<<<down_grid, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(workspace.activations.data),
            static_cast<const std::int32_t*>(workspace.expert_offsets.data),
            static_cast<const std::int32_t*>(workspace.expert_counts.data),
            static_cast<const std::int32_t*>(workspace.active_expert_ids.data),
            static_cast<const std::int32_t*>(workspace.active_count.data),
            static_cast<const std::int32_t*>(workspace.grouped_tokens.data),
            static_cast<const std::int32_t*>(workspace.grouped_paths.data),
            reinterpret_cast<const std::uint8_t*>(weights.expert_down.codes),
            reinterpret_cast<const std::uint8_t*>(weights.expert_down.scales),
            weights.expert_down.weight_scale_divisors,
            weights.expert_down.code_bytes_per_expert,
            weights.expert_down.scale_bytes_per_expert,
            static_cast<float*>(workspace.down_intermediate.data));
        CUDA_CHECK(cudaGetLastError());

        // 5. Shared Down & Top-K weighted reduction
        const dim3 reduce_grid(kHidden / 8, (static_cast<unsigned>(tokens) + 7) / 8);
        flash_next_moe_prefill_down_reduce_kernel<<<reduce_grid, 256, 0, stream>>>(
            static_cast<const std::int32_t*>(workspace.ids.data),
            static_cast<const float*>(workspace.alpha.data),
            static_cast<const float*>(workspace.shared_scale.data),
            static_cast<const __nv_bfloat16*>(workspace.activations.data),
            static_cast<float*>(workspace.down_intermediate.data),
            static_cast<const __nv_bfloat16*>(weights.shared_down.qdata),
            static_cast<__nv_bfloat16*>(output.data),
            tokens);
        CUDA_CHECK(cudaGetLastError());
    }
}

void flash_next_moe_bf16_kernels_launch(const Tensor& input, const MoeBf16Weights& weights,
                                        const FlashNextMoeWorkspace& workspace, Tensor& output,
                                        cudaStream_t stream) {
    const int tokens = static_cast<int>(input.ne[1]);
    const std::uint64_t gate_up_stride =
        weights.expert_gate_up.bytes_per_expert / sizeof(std::uint16_t);
    const std::uint64_t down_stride =
        weights.expert_down.bytes_per_expert / sizeof(std::uint16_t);

    const dim3 gate_grid(kIntermediate / GateSchedule::kWarpsPerCta,
                         static_cast<unsigned>(tokens), kPaths);
    flash_next_moe_bf16_gate_up_kernel<<<gate_grid, GateSchedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data),
        static_cast<const std::int32_t*>(workspace.ids.data),
        reinterpret_cast<const __nv_bfloat16*>(weights.expert_gate_up.data),
        gate_up_stride,
        static_cast<const __nv_bfloat16*>(weights.shared_gate.qdata),
        static_cast<const __nv_bfloat16*>(weights.shared_up.qdata),
        static_cast<__nv_bfloat16*>(workspace.activations.data));
    CUDA_CHECK(cudaGetLastError());

    const dim3 down_grid(kHidden / kDownWarps, static_cast<unsigned>(tokens));
    flash_next_moe_bf16_down_kernel<<<down_grid, kDownWarps * 32, 0, stream>>>(
        static_cast<const std::int32_t*>(workspace.ids.data),
        static_cast<const float*>(workspace.alpha.data),
        static_cast<const float*>(workspace.shared_scale.data),
        static_cast<const __nv_bfloat16*>(workspace.activations.data),
        reinterpret_cast<const __nv_bfloat16*>(weights.expert_down.data),
        down_stride,
        static_cast<const __nv_bfloat16*>(weights.shared_down.qdata),
        static_cast<__nv_bfloat16*>(output.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
