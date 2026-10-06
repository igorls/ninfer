// Grouped routes (T > 8) of the NVFP4-bank SparseMoe. Tokens are grouped by selected expert so
// each active expert's weights are read once per call.
//
//   GroupedA16: SIMT W4A16 gate/up and down against the represented BF16 activations, FP32 down
//               partials, then one fixed-order reduction that also evaluates the shared down.
//   GroupedA4:  x and the expert products are quantized to NVFP4 (one dynamic UE4M3 scale per 16
//               values), gate/up and down run on block-scaled FP4 Tensor Cores; each routed down
//               result is scaled by its route weight and staged as BF16, then reduced in path
//               order onto the shared-expert base.
//
// Both routes evaluate the shared gate/up projections with BF16 Tensor Cores.

#include "ops/sparse_moe/nvfp4_bank/nvfp4_bank_moe.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/bf16/bf16_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_a4_quantize.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/sparse_moe/nvfp4_bank/expert_bank.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kExperts      = kNvfp4MoeExperts;
constexpr int kHidden       = kNvfp4MoeHidden;
constexpr int kIntermediate = kNvfp4MoeIntermediate;
constexpr int kTopK         = kNvfp4MoeTopK;
constexpr int kPaths        = kNvfp4MoePaths;
constexpr int kGateTiles    = kHidden / 64;       // 40 scale words per gate/up row
constexpr int kDownTiles    = kIntermediate / 64; // 10 scale words per down row

// ---------------------------------------------------------------------------------------------
// Grouping: histogram, exclusive offsets and the compact active-expert list, then placement.
// Placement order inside one expert group is irrelevant: every consumer computes each (token,
// path) independently and the final reduction reads paths in fixed order through token_to_pos.
// ---------------------------------------------------------------------------------------------
__global__ void nvfp4_moe_zero_counts_kernel(std::int32_t* __restrict__ counts) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < kExperts) { counts[tid] = 0; }
}

__global__ void nvfp4_moe_histogram_kernel(const std::int32_t* __restrict__ ids,
                                           std::int32_t* __restrict__ counts, int items) {
    const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < items) { atomicAdd(counts + ids[tid], 1); }
}

__device__ __forceinline__ int block_exclusive_scan_512(int value, int* warp_totals) {
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    int inclusive  = value;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xFFFF'FFFFU, inclusive, offset);
        if (lane >= offset) { inclusive += other; }
    }
    if (lane == 31) { warp_totals[warp] = inclusive; }
    __syncthreads();
    if (warp == 0) {
        int total = lane < 16 ? warp_totals[lane] : 0;
#pragma unroll
        for (int offset = 1; offset < 16; offset <<= 1) {
            const int other = __shfl_up_sync(0xFFFF'FFFFU, total, offset);
            if (lane >= offset) { total += other; }
        }
        if (lane < 16) { warp_totals[lane] = total; }
    }
    __syncthreads();
    const int base = warp > 0 ? warp_totals[warp - 1] : 0;
    // Every thread has read warp_totals before a later caller may overwrite it.
    __syncthreads();
    return base + inclusive - value;
}

__global__ void __launch_bounds__(kExperts)
    nvfp4_moe_offsets_kernel(const std::int32_t* __restrict__ counts,
                             std::int32_t* __restrict__ offsets,
                             std::int32_t* __restrict__ active_experts,
                             std::int32_t* __restrict__ active_count) {
    __shared__ int warp_totals[16];
    const int tid    = static_cast<int>(threadIdx.x);
    const int count  = counts[tid];
    const int offset = block_exclusive_scan_512(count, warp_totals);
    offsets[tid]     = offset;
    if (tid == kExperts - 1) { offsets[kExperts] = offset + count; }
    const int active = count > 0 ? 1 : 0;
    const int slot   = block_exclusive_scan_512(active, warp_totals);
    if (active != 0) { active_experts[slot] = tid; }
    if (tid == kExperts - 1) { *active_count = slot + active; }
}

__global__ void __launch_bounds__(kExperts)
    nvfp4_moe_place_kernel(const std::int32_t* __restrict__ ids,
                           const std::int32_t* __restrict__ offsets,
                           std::int32_t* __restrict__ grouped_tokens,
                           std::int32_t* __restrict__ grouped_paths,
                           std::int32_t* __restrict__ token_to_pos, int items) {
    __shared__ int heads[kExperts];
    const int tid = static_cast<int>(threadIdx.x);
    heads[tid]    = offsets[tid];
    __syncthreads();
    for (int item = tid; item < items; item += kExperts) {
        const int position       = atomicAdd(&heads[ids[item]], 1);
        grouped_tokens[position] = item / kTopK;
        grouped_paths[position]  = item % kTopK;
        token_to_pos[item]       = position;
    }
}

// ---------------------------------------------------------------------------------------------
// Shared expert gate/up with BF16 Tensor Core operands and private FP32 results. Apply SwiGLU
// before its BF16 product store, matching the decode route's nonlinear boundary.
// ---------------------------------------------------------------------------------------------
using SharedMmaSchedule =
    Bf16A16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;

__global__ void nvfp4_moe_shared_product_kernel(const float* __restrict__ gate,
                                                const float* __restrict__ up,
                                                __nv_bfloat16* __restrict__ product,
                                                __nv_bfloat16* __restrict__ activations,
                                                int tokens) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= tokens * kIntermediate) { return; }
    const int token = index / kIntermediate;
    const int row   = index - token * kIntermediate;
    const auto value =
        __float2bfloat16_rn(silu(gate[index]) * up[index]);
    if (product != nullptr) { product[index] = value; }
    activations[(static_cast<std::int64_t>(token) * kPaths + kTopK) * kIntermediate + row] = value;
}

// ---------------------------------------------------------------------------------------------
// GroupedA16 gate/up: eight intermediate rows (gate and up pair) per CTA, experts strided over
// grid.y, up to four tokens of one group staged in shared memory per pass.
// ---------------------------------------------------------------------------------------------
constexpr int kSimtTokens = 4;

__global__ void __launch_bounds__(256, 4) nvfp4_moe_a16_gate_up_kernel(
    const __nv_bfloat16* __restrict__ x, const std::int32_t* __restrict__ offsets,
    const std::int32_t* __restrict__ counts, const std::int32_t* __restrict__ active_experts,
    const std::int32_t* __restrict__ active_count, const std::int32_t* __restrict__ grouped_tokens,
    const std::int32_t* __restrict__ grouped_paths, const std::uint8_t* __restrict__ bank_codes,
    const std::uint8_t* __restrict__ bank_scales, const float* __restrict__ bank_divisors,
    std::uint64_t bank_code_stride, std::uint64_t bank_scale_stride,
    __nv_bfloat16* __restrict__ activations) {
    const Nvfp4ExpertBankDevice bank{bank_codes, bank_scales, bank_divisors, bank_code_stride,
                                     bank_scale_stride};
    __shared__ uint4 inputs[kSimtTokens][kHidden / 8];
    __shared__ std::uint8_t scale_bytes[16][kGateTiles * 4];
    const int total = *active_count;
    const int tid   = static_cast<int>(threadIdx.x);
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int pair  = static_cast<int>(blockIdx.x) * 8 + warp;

    for (int active = static_cast<int>(blockIdx.y); active < total;
         active += static_cast<int>(gridDim.y)) {
        const int expert_id            = active_experts[active];
        const int count                = counts[expert_id];
        const int start                = offsets[expert_id];
        const Nvfp4ExpertPlanes expert = nvfp4_expert(bank, expert_id);

        for (int task = tid; task < 16 * kGateTiles; task += 256) {
            const int tile  = task >> 4;
            const int local = task & 15;
            const int row =
                static_cast<int>(blockIdx.x) * 8 + (local & 7) + (local >= 8 ? kIntermediate : 0);
            *reinterpret_cast<std::uint32_t*>(&scale_bytes[local][tile * 4]) =
                *reinterpret_cast<const std::uint32_t*>(
                    expert.scales + nvfp4_expert_scale_word(row, tile, kGateTiles));
        }
        __syncthreads();

        const auto* gate_codes = expert.codes + static_cast<std::int64_t>(pair) * (kHidden / 2);
        const auto* up_codes =
            expert.codes + static_cast<std::int64_t>(pair + kIntermediate) * (kHidden / 2);

        for (int base = 0; base < count; base += kSimtTokens) {
            const int batch = min(kSimtTokens, count - base);
            for (int item = tid; item < batch * (kHidden / 8); item += 256) {
                const int b      = item / (kHidden / 8);
                const int chunk  = item - b * (kHidden / 8);
                const int token  = grouped_tokens[start + base + b];
                inputs[b][chunk] = *reinterpret_cast<const uint4*>(
                    x + static_cast<std::int64_t>(token) * kHidden + chunk * 8);
            }
            __syncthreads();

            float gate_acc[kSimtTokens] = {};
            float up_acc[kSimtTokens]   = {};
#pragma unroll
            for (int phase = 0; phase < kHidden / 512; ++phase) {
                const int group     = phase * 32 + lane;
                const uint2 g_words = *reinterpret_cast<const uint2*>(gate_codes + group * 8);
                const uint2 u_words = *reinterpret_cast<const uint2*>(up_codes + group * 8);
                const float g_coefficient =
                    decode_nvfp4_e4m3(scale_bytes[warp][group]) * expert.inverse_divisor;
                const float u_coefficient =
                    decode_nvfp4_e4m3(scale_bytes[8 + warp][group]) * expert.inverse_divisor;
                float2 g_values[8];
                float2 u_values[8];
#pragma unroll
                for (int p = 0; p < 8; ++p) {
                    const std::uint32_t gw = p < 4 ? g_words.x : g_words.y;
                    const std::uint32_t uw = p < 4 ? u_words.x : u_words.y;
                    g_values[p] =
                        decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(gw >> (8 * (p & 3))));
                    u_values[p] =
                        decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(uw >> (8 * (p & 3))));
                }
#pragma unroll
                for (int b = 0; b < kSimtTokens; ++b) {
                    if (b < batch) {
                        const uint4 v0              = inputs[b][group * 2];
                        const uint4 v1              = inputs[b][group * 2 + 1];
                        const std::uint32_t bits[8] = {v0.x, v0.y, v0.z, v0.w,
                                                       v1.x, v1.y, v1.z, v1.w};
                        float g_sum                 = 0.0F;
                        float u_sum                 = 0.0F;
#pragma unroll
                        for (int p = 0; p < 8; ++p) {
                            const float2 xv = bf16x2_bits_to_float2(bits[p]);
                            g_sum           = fmaf(g_values[p].x, xv.x, g_sum);
                            g_sum           = fmaf(g_values[p].y, xv.y, g_sum);
                            u_sum           = fmaf(u_values[p].x, xv.x, u_sum);
                            u_sum           = fmaf(u_values[p].y, xv.y, u_sum);
                        }
                        gate_acc[b] = fmaf(g_sum, g_coefficient, gate_acc[b]);
                        up_acc[b]   = fmaf(u_sum, u_coefficient, up_acc[b]);
                    }
                }
            }
#pragma unroll
            for (int b = 0; b < kSimtTokens; ++b) {
                if (b < batch) {
                    const float gate = warp_reduce_sum(gate_acc[b]);
                    const float up   = warp_reduce_sum(up_acc[b]);
                    if (lane == 0) {
                        const int position = start + base + b;
                        activations[(static_cast<std::int64_t>(grouped_tokens[position]) * kPaths +
                                     grouped_paths[position]) *
                                        kIntermediate +
                                    pair]  = __float2bfloat16_rn(silu(gate) * up);
                    }
                }
            }
            __syncthreads(); // inputs are restaged by the next pass
        }
        // scale_bytes are restaged for the next expert.
        __syncthreads();
    }
}

// GroupedA16 down: sixteen output rows per CTA (two per warp), FP32 partials [path, row, token].
__global__ void __launch_bounds__(256, 4) nvfp4_moe_a16_down_kernel(
    const __nv_bfloat16* __restrict__ activations, const std::int32_t* __restrict__ offsets,
    const std::int32_t* __restrict__ counts, const std::int32_t* __restrict__ active_experts,
    const std::int32_t* __restrict__ active_count, const std::int32_t* __restrict__ grouped_tokens,
    const std::int32_t* __restrict__ grouped_paths, const std::uint8_t* __restrict__ bank_codes,
    const std::uint8_t* __restrict__ bank_scales, const float* __restrict__ bank_divisors,
    std::uint64_t bank_code_stride, std::uint64_t bank_scale_stride, float* __restrict__ partials) {
    const Nvfp4ExpertBankDevice bank{bank_codes, bank_scales, bank_divisors, bank_code_stride,
                                     bank_scale_stride};
    __shared__ uint4 products[kSimtTokens][kIntermediate / 8];
    __shared__ std::uint8_t scale_bytes[16][kDownTiles * 4];
    const int total     = *active_count;
    const int tid       = static_cast<int>(threadIdx.x);
    const int warp      = tid >> 5;
    const int lane      = tid & 31;
    const int local_row = warp * 2;
    const int row_base  = static_cast<int>(blockIdx.x) * 16 + local_row;

    for (int active = static_cast<int>(blockIdx.y); active < total;
         active += static_cast<int>(gridDim.y)) {
        const int expert_id            = active_experts[active];
        const int count                = counts[expert_id];
        const int start                = offsets[expert_id];
        const Nvfp4ExpertPlanes expert = nvfp4_expert(bank, expert_id);

        for (int task = tid; task < 16 * kDownTiles; task += 256) {
            const int tile  = task >> 4;
            const int local = task & 15;
            *reinterpret_cast<std::uint32_t*>(&scale_bytes[local][tile * 4]) =
                *reinterpret_cast<const std::uint32_t*>(
                    expert.scales +
                    nvfp4_expert_scale_word(static_cast<int>(blockIdx.x) * 16 + local, tile,
                                            kDownTiles));
        }
        __syncthreads();

        for (int base = 0; base < count; base += kSimtTokens) {
            const int batch = min(kSimtTokens, count - base);
            for (int item = tid; item < batch * (kIntermediate / 8); item += 256) {
                const int b        = item / (kIntermediate / 8);
                const int chunk    = item - b * (kIntermediate / 8);
                const int position = start + base + b;
                products[b][chunk] = *reinterpret_cast<const uint4*>(
                    activations +
                    (static_cast<std::int64_t>(grouped_tokens[position]) * kPaths +
                     grouped_paths[position]) *
                        kIntermediate +
                    chunk * 8);
            }
            __syncthreads();

            float acc[kSimtTokens][2] = {};
            for (int group = lane; group < kIntermediate / 16; group += 32) {
                float2 w[2][8];
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const uint2 words = *reinterpret_cast<const uint2*>(
                        expert.codes +
                        static_cast<std::int64_t>(row_base + r) * (kIntermediate / 2) + group * 8);
#pragma unroll
                    for (int p = 0; p < 8; ++p) {
                        const std::uint32_t word = p < 4 ? words.x : words.y;
                        w[r][p] =
                            decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(word >> (8 * (p & 3))));
                    }
                }
#pragma unroll
                for (int b = 0; b < kSimtTokens; ++b) {
                    if (b < batch) {
                        const uint4 v0              = products[b][group * 2];
                        const uint4 v1              = products[b][group * 2 + 1];
                        const std::uint32_t bits[8] = {v0.x, v0.y, v0.z, v0.w,
                                                       v1.x, v1.y, v1.z, v1.w};
#pragma unroll
                        for (int r = 0; r < 2; ++r) {
                            const float coefficient =
                                decode_nvfp4_e4m3(scale_bytes[local_row + r][group]) *
                                expert.inverse_divisor;
                            float sum = 0.0F;
#pragma unroll
                            for (int p = 0; p < 8; ++p) {
                                const float2 hv = bf16x2_bits_to_float2(bits[p]);
                                sum             = fmaf(w[r][p].x, hv.x, sum);
                                sum             = fmaf(w[r][p].y, hv.y, sum);
                            }
                            acc[b][r] = fmaf(sum, coefficient, acc[b][r]);
                        }
                    }
                }
            }
#pragma unroll
            for (int b = 0; b < kSimtTokens; ++b) {
                if (b < batch) {
                    const int position = start + base + b;
                    const int token    = grouped_tokens[position];
                    const int path     = grouped_paths[position];
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        const float value = warp_reduce_sum(acc[b][r]);
                        if (lane == 0) {
                            partials[(static_cast<std::int64_t>(token) * kHidden + row_base + r) *
                                         kTopK +
                                     path] = value;
                        }
                    }
                }
            }
            __syncthreads(); // products are restaged by the next pass
        }
        __syncthreads(); // scale_bytes are restaged for the next expert
    }
}

// GroupedA16 epilogue: shared down (BF16 SIMT) plus the route-weighted routed partials in path
// order. Eight rows per CTA, eight tokens per pass.
__global__ void __launch_bounds__(256) nvfp4_moe_a16_reduce_kernel(
    const float* __restrict__ route_weights, const float* __restrict__ shared_scale,
    const __nv_bfloat16* __restrict__ activations, const float* __restrict__ partials,
    const __nv_bfloat16* __restrict__ shared_down, __nv_bfloat16* __restrict__ output, int tokens) {
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int row        = static_cast<int>(blockIdx.x) * 8 + warp;
    const int token_base = static_cast<int>(blockIdx.y) * 8;
    const int batch      = min(8, tokens - token_base);
    const auto* w_row    = shared_down + static_cast<std::int64_t>(row) * kIntermediate;

    float shared[8] = {};
    for (int column = lane * 8; column < kIntermediate; column += 256) {
        const uint4 w             = *reinterpret_cast<const uint4*>(w_row + column);
        const std::uint32_t wb[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            if (b < batch) {
                const uint4 a = *reinterpret_cast<const uint4*>(
                    activations +
                    (static_cast<std::int64_t>(token_base + b) * kPaths + kTopK) * kIntermediate +
                    column);
                const std::uint32_t ab[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float2 wv = bf16x2_bits_to_float2(wb[i]);
                    const float2 av = bf16x2_bits_to_float2(ab[i]);
                    shared[b]       = fmaf(wv.x, av.x, shared[b]);
                    shared[b]       = fmaf(wv.y, av.y, shared[b]);
                }
            }
        }
    }
#pragma unroll
    for (int b = 0; b < 8; ++b) {
        if (b < batch) {
            const int token        = token_base + b;
            const float shared_sum = warp_reduce_sum(shared[b]);
            float routed           = 0.0F;
            if (lane < kTopK) {
                routed =
                    route_weights[token * kTopK + lane] *
                    partials[(static_cast<std::int64_t>(token) * kHidden + row) * kTopK + lane];
            }
            routed = warp_reduce_sum<16>(routed); // lanes 0..9 hold the paths, 10..15 zero
            if (lane == 0) {
                output[static_cast<std::int64_t>(token) * kHidden + row] =
                    __float2bfloat16_rn(routed + shared_sum * shared_scale[token]);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// GroupedA4 gate/up: 16 intermediate pairs (32 weight rows) x 32 tokens per CTA, eight warps as
// 2 (rows) x 4 (tokens), two K=1280 stages of block-scaled FP4 MMA.
// ---------------------------------------------------------------------------------------------
constexpr int kA4GateCodes       = 640; // bytes per staged K half-row
constexpr int kA4GateScaleStride = 84;  // 20 scale words + padding per staged row
constexpr int kA4GateSharedBytes = 2 * 32 * (kA4GateCodes + kA4GateScaleStride);

__global__ void __launch_bounds__(256, 2) nvfp4_moe_a4_gate_up_kernel(
    const std::uint8_t* __restrict__ input_codes, const std::uint8_t* __restrict__ input_scales,
    const std::int32_t* __restrict__ offsets, const std::int32_t* __restrict__ counts,
    const std::int32_t* __restrict__ active_experts, const std::int32_t* __restrict__ active_count,
    const std::int32_t* __restrict__ grouped_tokens, const std::int32_t* __restrict__ grouped_paths,
    const std::uint8_t* __restrict__ bank_codes, const std::uint8_t* __restrict__ bank_scales,
    const float* __restrict__ bank_divisors, std::uint64_t bank_code_stride,
    std::uint64_t bank_scale_stride, __nv_bfloat16* __restrict__ activations) {
    const Nvfp4ExpertBankDevice bank{bank_codes, bank_scales, bank_divisors, bank_code_stride,
                                     bank_scale_stride};
    const int active_index = static_cast<int>(blockIdx.y);
    if (active_index >= *active_count) { return; }
    const int expert_id            = active_experts[active_index];
    const int count                = counts[expert_id];
    const int offset               = offsets[expert_id];
    const Nvfp4ExpertPlanes expert = nvfp4_expert(bank, expert_id);
    const int pair_base            = static_cast<int>(blockIdx.x) * 16;
    const int tid                  = static_cast<int>(threadIdx.x);
    const int warp                 = tid >> 5;
    const int lane                 = tid & 31;
    const int warp_m               = (warp & 3) >> 1;
    const int warp_n               = (warp >> 2) * 2 + (warp & 1);

    extern __shared__ __align__(16) std::uint8_t shared_bytes[];
    std::uint8_t* weight_codes  = shared_bytes;
    std::uint8_t* weight_scales = weight_codes + 32 * kA4GateCodes;
    std::uint8_t* act_codes     = weight_scales + 32 * kA4GateScaleStride;
    std::uint8_t* act_scales    = act_codes + 32 * kA4GateCodes;

    for (int token_chunk = 0; token_chunk < count; token_chunk += 32) {
        float accumulators[4] = {};
        for (int k_half = 0; k_half < 2; ++k_half) {
            for (int i = tid * 16; i < 32 * kA4GateCodes; i += 256 * 16) {
                const int row        = i / kA4GateCodes;
                const int column     = i - row * kA4GateCodes;
                const int local      = row & 15;
                const int pair       = pair_base + (row >> 4) * 8 + (local & 7);
                const int weight_row = pair + (local >= 8 ? kIntermediate : 0);
                cp_async<16, Cache::cg>(weight_codes + i,
                                        expert.codes +
                                            static_cast<std::int64_t>(weight_row) * (kHidden / 2) +
                                            k_half * kA4GateCodes + column);
                const int token_index = token_chunk + row;
                if (token_index < count) {
                    const int token = grouped_tokens[offset + token_index];
                    cp_async<16, Cache::ca>(act_codes + i,
                                            input_codes +
                                                static_cast<std::int64_t>(token) * (kHidden / 2) +
                                                k_half * kA4GateCodes + column);
                } else {
                    *reinterpret_cast<uint4*>(act_codes + i) = make_uint4(0, 0, 0, 0);
                }
            }
            for (int i = tid; i < 32 * 20; i += 256) {
                const int row        = i / 20;
                const int tile       = i - row * 20;
                const int local      = row & 15;
                const int pair       = pair_base + (row >> 4) * 8 + (local & 7);
                const int weight_row = pair + (local >= 8 ? kIntermediate : 0);
                cp_async<4, Cache::ca>(weight_scales + row * kA4GateScaleStride + tile * 4,
                                       expert.scales + nvfp4_expert_scale_word(weight_row,
                                                                               k_half * 20 + tile,
                                                                               kGateTiles));
                const int token_index = token_chunk + row;
                if (token_index < count) {
                    const int token = grouped_tokens[offset + token_index];
                    cp_async<4, Cache::ca>(act_scales + row * kA4GateScaleStride + tile * 4,
                                           input_scales +
                                               static_cast<std::int64_t>(token) * (kHidden / 16) +
                                               (k_half * 20 + tile) * 4);
                } else {
                    *reinterpret_cast<std::uint32_t*>(act_scales + row * kA4GateScaleStride +
                                                      tile * 4) = 0;
                }
            }
            cp_commit();
            cp_wait<0>();
            __syncthreads();
            if (token_chunk + warp_n * 8 < count) {
                const std::uint8_t* w_codes  = weight_codes + warp_m * 16 * kA4GateCodes;
                const std::uint8_t* w_scales = weight_scales + warp_m * 16 * kA4GateScaleStride;
                const std::uint8_t* a_codes  = act_codes + warp_n * 8 * kA4GateCodes;
                const std::uint8_t* a_scales = act_scales + warp_n * 8 * kA4GateScaleStride;
#pragma unroll 4
                for (int k = 0; k < 20; ++k) {
                    unsigned a[4];
                    unsigned b[2];
                    const int matrix_row = (lane & 7) + (((lane >> 3) & 1) << 3);
                    const int column     = (lane >> 4) * 16;
                    ldmatrix_x4(a[0], a[1], a[2], a[3],
                                smem_addr(w_codes + matrix_row * kA4GateCodes + k * 32 + column));
                    ldmatrix_x2(b[0], b[1],
                                smem_addr(a_codes + (lane & 7) * kA4GateCodes + k * 32 +
                                          ((lane >> 3) & 1) * 16));
                    const unsigned scale_a = *reinterpret_cast<const unsigned*>(
                        w_scales + (((lane & 1) << 3) | (lane >> 2)) * kA4GateScaleStride + k * 4);
                    const unsigned scale_b = *reinterpret_cast<const unsigned*>(
                        a_scales + (lane >> 2) * kA4GateScaleStride + k * 4);
                    mma_nvfp4_e4m3(accumulators[0], accumulators[1], accumulators[2],
                                   accumulators[3], a[0], a[1], a[2], a[3], b[0], b[1], scale_a,
                                   scale_b);
                }
            }
            __syncthreads(); // the next stage overwrites every staged plane
        }
        const int token_base = token_chunk + warp_n * 8;
        const int pair       = pair_base + warp_m * 8 + (lane >> 2);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int token_index = token_base + 2 * (lane & 3) + j;
            if (token_index < count) {
                const int position = offset + token_index;
                const float gate   = accumulators[j] * expert.inverse_divisor;
                const float up     = accumulators[j + 2] * expert.inverse_divisor;
                activations[(static_cast<std::int64_t>(grouped_tokens[position]) * kPaths +
                             grouped_paths[position]) *
                                kIntermediate +
                            pair]  = __float2bfloat16_rn(silu(gate) * up);
            }
        }
    }
}

// GroupedA4 down: 64 output rows x 16 grouped products per CTA (four warps as 2 x 2), block-scaled
// FP4 MMA over K=640; the result is scaled by 1/divisor and the route weight and staged as BF16 at
// the product's grouped position.
__global__ void __launch_bounds__(128, 4) nvfp4_moe_a4_down_kernel(
    const std::uint8_t* __restrict__ product_codes, const std::uint8_t* __restrict__ product_scales,
    const std::int32_t* __restrict__ offsets, const std::int32_t* __restrict__ counts,
    const std::int32_t* __restrict__ active_experts, const std::int32_t* __restrict__ active_count,
    const std::int32_t* __restrict__ grouped_tokens, const std::int32_t* __restrict__ grouped_paths,
    const float* __restrict__ route_weights, const std::uint8_t* __restrict__ bank_codes,
    const std::uint8_t* __restrict__ bank_scales, const float* __restrict__ bank_divisors,
    std::uint64_t bank_code_stride, std::uint64_t bank_scale_stride,
    __nv_bfloat16* __restrict__ weighted_products) {
    const Nvfp4ExpertBankDevice bank{bank_codes, bank_scales, bank_divisors, bank_code_stride,
                                     bank_scale_stride};
    constexpr int kCodes   = kIntermediate / 2;  // 320 bytes per row
    constexpr int kScales  = kIntermediate / 16; // 40 bytes per row
    const int active_index = static_cast<int>(blockIdx.y);
    if (active_index >= *active_count) { return; }
    const int expert_id            = active_experts[active_index];
    const int count                = counts[expert_id];
    const int offset               = offsets[expert_id];
    const Nvfp4ExpertPlanes expert = nvfp4_expert(bank, expert_id);
    const int row_base             = static_cast<int>(blockIdx.x) * 64;
    const int tid                  = static_cast<int>(threadIdx.x);
    const int warp                 = tid >> 5;
    const int warp_m               = warp >> 1;
    const int warp_n               = warp & 1;
    const int lane                 = tid & 31;

    __shared__ __align__(16) std::uint8_t w_codes[4][16 * kCodes];
    __shared__ __align__(16) std::uint8_t w_scales[4][16 * kScales];
    __shared__ __align__(16) std::uint8_t a_codes[2][8 * kCodes];
    __shared__ __align__(16) std::uint8_t a_scales[2][8 * kScales];

#pragma unroll
    for (int i = tid * 16; i < 64 * kCodes; i += 128 * 16) {
        const int row    = i / kCodes;
        const int column = i - row * kCodes;
        *reinterpret_cast<uint4*>(w_codes[row >> 4] + (row & 15) * kCodes + column) =
            *reinterpret_cast<const uint4*>(
                expert.codes + static_cast<std::int64_t>(row_base + row) * kCodes + column);
    }
#pragma unroll
    for (int i = tid; i < 64 * kDownTiles; i += 128) {
        const int row  = i / kDownTiles;
        const int tile = i - row * kDownTiles;
        *reinterpret_cast<std::uint32_t*>(w_scales[row >> 4] + (row & 15) * kScales + tile * 4) =
            *reinterpret_cast<const std::uint32_t*>(
                expert.scales + nvfp4_expert_scale_word(row_base + row, tile, kDownTiles));
    }
    __syncthreads();

    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;

    for (int chunk = 0; chunk < count; chunk += 16) {
#pragma unroll
        for (int i = tid * 16; i < 16 * kCodes; i += 128 * 16) {
            const int index   = i / kCodes;
            const int column  = i - index * kCodes;
            auto* destination = a_codes[index >> 3] + (index & 7) * kCodes + column;
            if (chunk + index < count) {
                const int position = offset + chunk + index;
                const std::int64_t product =
                    static_cast<std::int64_t>(grouped_tokens[position]) * kPaths +
                    grouped_paths[position];
                *reinterpret_cast<uint4*>(destination) =
                    *reinterpret_cast<const uint4*>(product_codes + product * kCodes + column);
            } else {
                *reinterpret_cast<uint4*>(destination) = make_uint4(0, 0, 0, 0);
            }
        }
#pragma unroll
        for (int i = tid * 4; i < 16 * kScales; i += 128 * 4) {
            const int index   = i / kScales;
            const int column  = i - index * kScales;
            auto* destination = a_scales[index >> 3] + (index & 7) * kScales + column;
            if (chunk + index < count) {
                const int position = offset + chunk + index;
                const std::int64_t product =
                    static_cast<std::int64_t>(grouped_tokens[position]) * kPaths +
                    grouped_paths[position];
                *reinterpret_cast<std::uint32_t*>(destination) =
                    *reinterpret_cast<const std::uint32_t*>(product_scales + product * kScales +
                                                            column);
            } else {
                *reinterpret_cast<std::uint32_t*>(destination) = 0;
            }
        }
        __syncthreads();

        const int local_base = chunk + warp_n * 8;
        const int batch      = max(0, min(8, count - local_base));
        if (batch > 0) {
#pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                const int group       = warp_m * 2 + sub;
                float accumulators[4] = {};
#pragma unroll 2
                for (int k = 0; k < kIntermediate / 64; ++k) {
                    unsigned a[4];
                    unsigned b[2];
                    ldmatrix_x4(
                        a[0], a[1], a[2], a[3],
                        smem_addr(w_codes[group] + a_row_offset * kCodes + k * 32 + a_column_byte));
                    ldmatrix_x2(b[0], b[1],
                                smem_addr(a_codes[warp_n] + b_row_offset * kCodes + k * 32 +
                                          b_column_byte));
                    const std::uint32_t sfa = *reinterpret_cast<const std::uint32_t*>(
                        w_scales[group] + sfa_row * kScales + k * 4);
                    const std::uint32_t sfb = *reinterpret_cast<const std::uint32_t*>(
                        a_scales[warp_n] + sfb_row * kScales + k * 4);
                    mma_nvfp4_e4m3(accumulators[0], accumulators[1], accumulators[2],
                                   accumulators[3], a[0], a[1], a[2], a[3], b[0], b[1], sfa, sfb);
                }
                const int row0 = row_base + group * 16 + (lane >> 2);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int local = 2 * (lane & 3) + j;
                    if (local < batch) {
                        const int position = offset + local_base + local;
                        const float scale  = route_weights[grouped_tokens[position] * kTopK +
                                                          grouped_paths[position]] *
                                            expert.inverse_divisor;
                        auto* staged =
                            weighted_products + static_cast<std::int64_t>(position) * kHidden;
                        staged[row0]     = __float2bfloat16_rn(accumulators[j] * scale);
                        staged[row0 + 8] = __float2bfloat16_rn(accumulators[j + 2] * scale);
                    }
                }
            }
        }
        __syncthreads(); // the next chunk restages the product planes
    }
}

// GroupedA4 shared down output: BF16(shared_scale[t] * value) is the base of the reduction.
struct ScaledSharedOutput {
    __nv_bfloat16* data;
    const float* scale;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        data[static_cast<std::int64_t>(token) * kHidden + row] =
            __float2bfloat16_rn(value * scale[token]);
    }
};

// GroupedA4 reduction: base plus the ten staged products in path order.
__global__ void __launch_bounds__(128)
    nvfp4_moe_a4_reduce_kernel(const std::int32_t* __restrict__ token_to_pos,
                               const __nv_bfloat16* __restrict__ weighted_products,
                               __nv_bfloat16* __restrict__ output, int tokens) {
    const int row_base   = static_cast<int>(blockIdx.x) * 64;
    const int token_base = static_cast<int>(blockIdx.y) * 16;
    const int tid        = static_cast<int>(threadIdx.x);
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int local = tid * 8 + item;
        const int row   = row_base + (local >> 4);
        const int token = token_base + (local & 15);
        if (token < tokens) {
            auto* out = output + static_cast<std::int64_t>(token) * kHidden + row;
            float sum = __bfloat162float(*out);
#pragma unroll
            for (int path = 0; path < kTopK; ++path) {
                const int position = token_to_pos[token * kTopK + path];
                sum += __bfloat162float(
                    weighted_products[static_cast<std::int64_t>(position) * kHidden + row]);
            }
            *out = __float2bfloat16_rn(sum);
        }
    }
}

void launch_grouping(const Nvfp4MoeWorkspace& w, int tokens, cudaStream_t stream) {
    const int items = tokens * kTopK;
    auto* counts    = static_cast<std::int32_t*>(w.expert_counts.data);
    nvfp4_moe_zero_counts_kernel<<<(kExperts + 255) / 256, 256, 0, stream>>>(counts);
    CUDA_CHECK(cudaGetLastError());
    nvfp4_moe_histogram_kernel<<<(items + 255) / 256, 256, 0, stream>>>(
        static_cast<const std::int32_t*>(w.ids.data), counts, items);
    CUDA_CHECK(cudaGetLastError());
    nvfp4_moe_offsets_kernel<<<1, kExperts, 0, stream>>>(
        counts, static_cast<std::int32_t*>(w.expert_offsets.data),
        static_cast<std::int32_t*>(w.active_experts.data),
        static_cast<std::int32_t*>(w.active_count.data));
    CUDA_CHECK(cudaGetLastError());
    nvfp4_moe_place_kernel<<<1, kExperts, 0, stream>>>(
        static_cast<const std::int32_t*>(w.ids.data),
        static_cast<const std::int32_t*>(w.expert_offsets.data),
        static_cast<std::int32_t*>(w.grouped_tokens.data),
        static_cast<std::int32_t*>(w.grouped_paths.data),
        static_cast<std::int32_t*>(w.token_to_pos.data), items);
    CUDA_CHECK(cudaGetLastError());
}

struct SharedFloatOutput {
    float* data;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        data[static_cast<std::int64_t>(token) * kIntermediate + row] = value;
    }
};

void launch_shared_gate_up(const Tensor& x, const SparseMoeNvfp4BankWeights& weights,
                           const Nvfp4MoeWorkspace& w, int tokens, cudaStream_t stream) {
    const auto* input = static_cast<const __nv_bfloat16*>(x.data);
    auto* gate        = static_cast<float*>(w.shared_gemm.data);
    auto* up          = gate + static_cast<std::int64_t>(kIntermediate) * tokens;
    launch_bf16_a16_mma<SharedMmaSchedule>(
        Bf16A16Operands{input, static_cast<const __nv_bfloat16*>(weights.shared_gate.qdata),
                        kIntermediate, kHidden, tokens},
        SharedFloatOutput{gate}, LinearIdentityEpilogue{}, stream);
    launch_bf16_a16_mma<SharedMmaSchedule>(
        Bf16A16Operands{input, static_cast<const __nv_bfloat16*>(weights.shared_up.qdata),
                        kIntermediate, kHidden, tokens},
        SharedFloatOutput{up}, LinearIdentityEpilogue{}, stream);
    const int elements = tokens * kIntermediate;
    nvfp4_moe_shared_product_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
        gate, up, static_cast<__nv_bfloat16*>(w.shared_product.data),
        static_cast<__nv_bfloat16*>(w.activations.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_moe_grouped(const Tensor& x, const SparseMoeNvfp4BankWeights& weights,
                       const Nvfp4MoeWorkspace& w, Nvfp4MoeRoute route, Tensor& destination,
                       cudaStream_t stream) {
    const int tokens = x.ne[1];
    launch_grouping(w, tokens, stream);
    launch_shared_gate_up(x, weights, w, tokens, stream);

    const auto* offsets        = static_cast<const std::int32_t*>(w.expert_offsets.data);
    const auto* counts         = static_cast<const std::int32_t*>(w.expert_counts.data);
    const auto* active_experts = static_cast<const std::int32_t*>(w.active_experts.data);
    const auto* active_count   = static_cast<const std::int32_t*>(w.active_count.data);
    const auto* grouped_tokens = static_cast<const std::int32_t*>(w.grouped_tokens.data);
    const auto* grouped_paths  = static_cast<const std::int32_t*>(w.grouped_paths.data);
    auto* activations          = static_cast<__nv_bfloat16*>(w.activations.data);
    auto* output               = static_cast<__nv_bfloat16*>(destination.data);
    const auto gate_up_bank    = nvfp4_expert_bank_device(weights.gate_up);
    const auto down_bank       = nvfp4_expert_bank_device(weights.down);
    const auto* shared_down    = static_cast<const __nv_bfloat16*>(weights.shared_down.qdata);

    if (route == Nvfp4MoeRoute::GroupedA16) {
        nvfp4_moe_a16_gate_up_kernel<<<dim3(kIntermediate / 8, 16), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), offsets, counts, active_experts,
            active_count, grouped_tokens, grouped_paths, gate_up_bank.codes, gate_up_bank.scales,
            gate_up_bank.divisors, gate_up_bank.code_stride, gate_up_bank.scale_stride,
            activations);
        CUDA_CHECK(cudaGetLastError());
        nvfp4_moe_a16_down_kernel<<<dim3(kHidden / 16, 8), 256, 0, stream>>>(
            activations, offsets, counts, active_experts, active_count, grouped_tokens,
            grouped_paths, down_bank.codes, down_bank.scales, down_bank.divisors,
            down_bank.code_stride, down_bank.scale_stride,
            static_cast<float*>(w.down_partials.data));
        CUDA_CHECK(cudaGetLastError());
        nvfp4_moe_a16_reduce_kernel<<<dim3(kHidden / 8, static_cast<unsigned>((tokens + 7) / 8)),
                                      256, 0, stream>>>(
            static_cast<const float*>(w.weights.data),
            static_cast<const float*>(w.shared_scale.data), activations,
            static_cast<const float*>(w.down_partials.data), shared_down, output, tokens);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    using InputGeometry            = Nvfp4ActivationGeometry<kHidden>;
    using ProductGeometry          = Nvfp4ActivationGeometry<kIntermediate>;
    constexpr int kQuantizeThreads = 256;
    const int input_tasks          = tokens * InputGeometry::kGroupsPerRow;
    nvfp4_a4_quantize_kernel<InputGeometry, kQuantizeThreads, Nvfp4ScaleLayout::RowMajor>
        <<<(input_tasks + kQuantizeThreads - 1) / kQuantizeThreads, kQuantizeThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<std::uint8_t*>(w.input_codes.data),
            static_cast<std::uint8_t*>(w.input_scales.data), tokens, tokens, 1.0F);
    CUDA_CHECK(cudaGetLastError());

    nvfp4_moe_a4_gate_up_kernel<<<dim3(kIntermediate / 16, kExperts), 256, kA4GateSharedBytes,
                                  stream>>>(
        static_cast<const std::uint8_t*>(w.input_codes.data),
        static_cast<const std::uint8_t*>(w.input_scales.data), offsets, counts, active_experts,
        active_count, grouped_tokens, grouped_paths, gate_up_bank.codes, gate_up_bank.scales,
        gate_up_bank.divisors, gate_up_bank.code_stride, gate_up_bank.scale_stride, activations);
    CUDA_CHECK(cudaGetLastError());

    const int product_rows  = tokens * kPaths;
    const int product_tasks = product_rows * ProductGeometry::kGroupsPerRow;
    nvfp4_a4_quantize_kernel<ProductGeometry, kQuantizeThreads, Nvfp4ScaleLayout::RowMajor>
        <<<(product_tasks + kQuantizeThreads - 1) / kQuantizeThreads, kQuantizeThreads, 0,
           stream>>>(activations, static_cast<std::uint8_t*>(w.product_codes.data),
                     static_cast<std::uint8_t*>(w.product_scales.data), product_rows, product_rows,
                     1.0F);
    CUDA_CHECK(cudaGetLastError());

    const auto* shared_product = static_cast<const __nv_bfloat16*>(w.shared_product.data);
    launch_bf16_a16_mma<SharedMmaSchedule>(
        Bf16A16Operands{shared_product, shared_down, kHidden, kIntermediate, tokens},
        ScaledSharedOutput{output, static_cast<const float*>(w.shared_scale.data)},
        LinearIdentityEpilogue{}, stream);

    nvfp4_moe_a4_down_kernel<<<dim3(kHidden / 64, kExperts), 128, 0, stream>>>(
        static_cast<const std::uint8_t*>(w.product_codes.data),
        static_cast<const std::uint8_t*>(w.product_scales.data), offsets, counts, active_experts,
        active_count, grouped_tokens, grouped_paths, static_cast<const float*>(w.weights.data),
        down_bank.codes, down_bank.scales, down_bank.divisors, down_bank.code_stride,
        down_bank.scale_stride, static_cast<__nv_bfloat16*>(w.weighted_products.data));
    CUDA_CHECK(cudaGetLastError());

    nvfp4_moe_a4_reduce_kernel<<<dim3(kHidden / 64, static_cast<unsigned>((tokens + 15) / 16)), 128,
                                 0, stream>>>(
        static_cast<const std::int32_t*>(w.token_to_pos.data),
        static_cast<const __nv_bfloat16*>(w.weighted_products.data), output, tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
