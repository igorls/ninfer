#include "targets/qwen3_8_flash_next/impl/moe_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "targets/qwen3_8_flash_next/impl/moe_shared_kernels.h"
#include "targets/qwen3_8_flash_next/impl/stage_ledger.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
// The staging switch below reads the environment; these were relied on
// transitively before and are now named explicitly.
#include <cstdlib>
#include <cstring>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

using Activation2560Geometry = ops::detail::Nvfp4ActivationGeometry<2'560>;
using Activation640Geometry  = ops::detail::Nvfp4ActivationGeometry<640>;

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
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const auto* x_u4      = reinterpret_cast<const uint4*>(x);
    const auto* first_u4  = reinterpret_cast<const uint4*>(first);
    const auto* second_u4 = reinterpret_cast<const uint4*>(second);
    float first_sum  = 0.0F;
    float second_sum = 0.0F;
    constexpr int kIters = K / (32 * 8); // 2560 / 256 = 10
    #pragma unroll
    for (int iter = 0; iter < kIters; ++iter) {
        const int idx = iter * 32 + lane;
        const uint4 xv = x_u4[idx];
        const uint4 fv = first_u4[idx];
        const uint4 sv = second_u4[idx];

        const float2 x0 = ops::bf16x2_bits_to_float2(xv.x);
        const float2 x1 = ops::bf16x2_bits_to_float2(xv.y);
        const float2 x2 = ops::bf16x2_bits_to_float2(xv.z);
        const float2 x3 = ops::bf16x2_bits_to_float2(xv.w);

        const float2 f0 = ops::bf16x2_bits_to_float2(fv.x);
        const float2 f1 = ops::bf16x2_bits_to_float2(fv.y);
        const float2 f2 = ops::bf16x2_bits_to_float2(fv.z);
        const float2 f3 = ops::bf16x2_bits_to_float2(fv.w);

        const float2 s0 = ops::bf16x2_bits_to_float2(sv.x);
        const float2 s1 = ops::bf16x2_bits_to_float2(sv.y);
        const float2 s2 = ops::bf16x2_bits_to_float2(sv.z);
        const float2 s3 = ops::bf16x2_bits_to_float2(sv.w);

        first_sum = fmaf(f0.x, x0.x, first_sum);
        first_sum = fmaf(f0.y, x0.y, first_sum);
        first_sum = fmaf(f1.x, x1.x, first_sum);
        first_sum = fmaf(f1.y, x1.y, first_sum);
        first_sum = fmaf(f2.x, x2.x, first_sum);
        first_sum = fmaf(f2.y, x2.y, first_sum);
        first_sum = fmaf(f3.x, x3.x, first_sum);
        first_sum = fmaf(f3.y, x3.y, first_sum);

        second_sum = fmaf(s0.x, x0.x, second_sum);
        second_sum = fmaf(s0.y, x0.y, second_sum);
        second_sum = fmaf(s1.x, x1.x, second_sum);
        second_sum = fmaf(s1.y, x1.y, second_sum);
        second_sum = fmaf(s2.x, x2.x, second_sum);
        second_sum = fmaf(s2.y, x2.y, second_sum);
        second_sum = fmaf(s3.x, x3.x, second_sum);
        second_sum = fmaf(s3.y, x3.y, second_sum);
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


__global__ void flash_next_moe_gate_up_kernel(
    const __nv_bfloat16* __restrict__ input, const std::int32_t* __restrict__ ids,
    const std::uint8_t* __restrict__ expert_codes, const std::uint8_t* __restrict__ expert_scales,
    const float* __restrict__ expert_divisors, std::uint64_t code_stride,
    std::uint64_t scale_stride, const __nv_bfloat16* __restrict__ shared_gate,
    const __nv_bfloat16* __restrict__ shared_up, __nv_bfloat16* __restrict__ activations) {
    const int token = static_cast<int>(blockIdx.y);
    const int path  = static_cast<int>(blockIdx.z);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int row   = static_cast<int>(blockIdx.x) * 8 + warp;
    if (row >= kIntermediate) { return; }

    const auto* x = input + static_cast<std::int64_t>(token) * kHidden;
    float gate = 0.0F;
    float up   = 0.0F;

    if (path < kTopK) {
        const int expert    = ids[token * kTopK + path];
        const auto* codes   = expert_codes + static_cast<std::uint64_t>(expert) * code_stride;
        const auto* scales  = expert_scales + static_cast<std::uint64_t>(expert) * scale_stride;
        const float inverse = 1.0F / expert_divisors[expert];

        const int row_gate = row;
        const int row_up   = row + kIntermediate;

        const int m_tile_gate       = row_gate / 128;
        const int row_inner_gate    = row_gate % 128;
        const int row_mod32_gate    = row_inner_gate & 31;
        const int row_quartile_gate = row_inner_gate >> 5;
        const std::int64_t scale_base_gate =
            static_cast<std::int64_t>(m_tile_gate * 160) * 512 +
            row_mod32_gate * 16 + row_quartile_gate * 4;

        const int m_tile_up       = row_up / 128;
        const int row_inner_up    = row_up % 128;
        const int row_mod32_up    = row_inner_up & 31;
        const int row_quartile_up = row_inner_up >> 5;
        const std::int64_t scale_base_up =
            static_cast<std::int64_t>(m_tile_up * 160) * 512 +
            row_mod32_up * 16 + row_quartile_up * 4;

        const auto* codes_gate = codes + static_cast<std::int64_t>(row_gate) * 1280;
        const auto* codes_up   = codes + static_cast<std::int64_t>(row_up) * 1280;

        float gate_acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        float up_acc[4]   = {0.0F, 0.0F, 0.0F, 0.0F};

        #pragma unroll
        for (int phase = 0; phase < 5; ++phase) {
            // 1. Cooperative Scale Load via Warp Shuffle (Lanes 0..7 load gate scales, Lanes 8..15 load up scales)
            std::uint32_t my_scale_word = 0;
            if (lane < 8) {
                const std::int64_t off = scale_base_gate + static_cast<std::int64_t>(phase * 8 + lane) * 512;
                my_scale_word = *reinterpret_cast<const std::uint32_t*>(scales + off);
            } else if (lane < 16) {
                const std::int64_t off = scale_base_up + static_cast<std::int64_t>(phase * 8 + (lane - 8)) * 512;
                my_scale_word = *reinterpret_cast<const std::uint32_t*>(scales + off);
            }

            const std::uint32_t gate_word = __shfl_sync(0xFFFFFFFFU, my_scale_word, lane >> 2);
            const std::uint32_t up_word   = __shfl_sync(0xFFFFFFFFU, my_scale_word, 8 + (lane >> 2));

            const std::uint8_t scale_byte_gate = static_cast<std::uint8_t>(gate_word >> ((lane & 3) * 8));
            const std::uint8_t scale_byte_up   = static_cast<std::uint8_t>(up_word >> ((lane & 3) * 8));

            const float coef_gate = ops::detail::decode_nvfp4_e4m3(scale_byte_gate) * inverse;
            const float coef_up   = ops::detail::decode_nvfp4_e4m3(scale_byte_up) * inverse;

            // 2. Vectorized 128-bit Activation Load (1 load of 16 bytes = 8 BF16s per thread)
            const auto* x_vec_ptr = reinterpret_cast<const uint4*>(x + phase * 512);
            const uint4 act_raw = x_vec_ptr[lane];
            const float2 act[4] = {
                ops::bf16x2_bits_to_float2(act_raw.x),
                ops::bf16x2_bits_to_float2(act_raw.y),
                ops::bf16x2_bits_to_float2(act_raw.z),
                ops::bf16x2_bits_to_float2(act_raw.w)
            };

            // 3. Vectorized 64-bit Code Loads (8 bytes for gate, 8 bytes for up)
            const auto* code_gate_ptr = reinterpret_cast<const uint2*>(codes_gate + phase * 256);
            const auto* code_up_ptr   = reinterpret_cast<const uint2*>(codes_up + phase * 256);
            const uint2 cg = code_gate_ptr[lane];
            const uint2 cu = code_up_ptr[lane];

            const float2 dw_gate[8] = {
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(cg.x)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(cg.x >> 8)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(cg.x >> 16)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(cg.x >> 24)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(cg.y)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(cg.y >> 8)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(cg.y >> 16)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(cg.y >> 24))
            };

            const float2 dw_up[8] = {
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(cu.x)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(cu.x >> 8)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(cu.x >> 16)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(cu.x >> 24)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(cu.y)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(cu.y >> 8)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(cu.y >> 16)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(cu.y >> 24))
            };

            // 4. Compute 16 FMAs for Gate and 16 FMAs for Up preserving the 4 accumulator chains
            gate_acc[0] = fmaf(dw_gate[0].x * coef_gate, act[0].x, gate_acc[0]);
            gate_acc[1] = fmaf(dw_gate[0].y * coef_gate, act[0].y, gate_acc[1]);
            gate_acc[2] = fmaf(dw_gate[1].x * coef_gate, act[0].x, gate_acc[2]);
            gate_acc[3] = fmaf(dw_gate[1].y * coef_gate, act[0].y, gate_acc[3]);

            gate_acc[0] = fmaf(dw_gate[2].x * coef_gate, act[1].x, gate_acc[0]);
            gate_acc[1] = fmaf(dw_gate[2].y * coef_gate, act[1].y, gate_acc[1]);
            gate_acc[2] = fmaf(dw_gate[3].x * coef_gate, act[1].x, gate_acc[2]);
            gate_acc[3] = fmaf(dw_gate[3].y * coef_gate, act[1].y, gate_acc[3]);

            gate_acc[0] = fmaf(dw_gate[4].x * coef_gate, act[2].x, gate_acc[0]);
            gate_acc[1] = fmaf(dw_gate[4].y * coef_gate, act[2].y, gate_acc[1]);
            gate_acc[2] = fmaf(dw_gate[5].x * coef_gate, act[2].x, gate_acc[2]);
            gate_acc[3] = fmaf(dw_gate[5].y * coef_gate, act[2].y, gate_acc[3]);

            gate_acc[0] = fmaf(dw_gate[6].x * coef_gate, act[3].x, gate_acc[0]);
            gate_acc[1] = fmaf(dw_gate[6].y * coef_gate, act[3].y, gate_acc[1]);
            gate_acc[2] = fmaf(dw_gate[7].x * coef_gate, act[3].x, gate_acc[2]);
            gate_acc[3] = fmaf(dw_gate[7].y * coef_gate, act[3].y, gate_acc[3]);

            up_acc[0] = fmaf(dw_up[0].x * coef_up, act[0].x, up_acc[0]);
            up_acc[1] = fmaf(dw_up[0].y * coef_up, act[0].y, up_acc[1]);
            up_acc[2] = fmaf(dw_up[1].x * coef_up, act[0].x, up_acc[2]);
            up_acc[3] = fmaf(dw_up[1].y * coef_up, act[0].y, up_acc[3]);

            up_acc[0] = fmaf(dw_up[2].x * coef_up, act[1].x, up_acc[0]);
            up_acc[1] = fmaf(dw_up[2].y * coef_up, act[1].y, up_acc[1]);
            up_acc[2] = fmaf(dw_up[3].x * coef_up, act[1].x, up_acc[2]);
            up_acc[3] = fmaf(dw_up[3].y * coef_up, act[1].y, up_acc[3]);

            up_acc[0] = fmaf(dw_up[4].x * coef_up, act[2].x, up_acc[0]);
            up_acc[1] = fmaf(dw_up[4].y * coef_up, act[2].y, up_acc[1]);
            up_acc[2] = fmaf(dw_up[5].x * coef_up, act[2].x, up_acc[2]);
            up_acc[3] = fmaf(dw_up[5].y * coef_up, act[2].y, up_acc[3]);

            up_acc[0] = fmaf(dw_up[6].x * coef_up, act[3].x, up_acc[0]);
            up_acc[1] = fmaf(dw_up[6].y * coef_up, act[3].y, up_acc[1]);
            up_acc[2] = fmaf(dw_up[7].x * coef_up, act[3].x, up_acc[2]);
            up_acc[3] = fmaf(dw_up[7].y * coef_up, act[3].y, up_acc[3]);
        }

        #pragma unroll
        for (int chain = 0; chain < 4; ++chain) {
            gate += gate_acc[chain];
            up   += up_acc[chain];
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
    const int tid   = static_cast<int>(threadIdx.x);
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int row   = static_cast<int>(blockIdx.x) * kDownWarps + warp;

    if (row >= kHidden) { return; }

    const int m_tile       = row / 128;
    const int row_inner    = row % 128;
    const int row_mod32    = row_inner & 31;
    const int row_quartile = row_inner >> 5;
    const std::int64_t scale_row_base = static_cast<std::int64_t>(m_tile * 10) * 512 +
                                        row_mod32 * 16 + row_quartile * 4;

    const auto* token_activations =
        activations + static_cast<std::int64_t>(token) * kPaths * kIntermediate;

    float routed = 0.0F;

#pragma unroll
    for (int path = 0; path < kTopK; ++path) {
        const int expert = ids[token * kTopK + path];
        const auto* exp_codes =
            expert_codes + static_cast<std::uint64_t>(expert) * code_stride;
        const auto* exp_scales =
            expert_scales + static_cast<std::uint64_t>(expert) * scale_stride;
        const float inverse = 1.0F / expert_divisors[expert];
        const auto* act_path = token_activations + static_cast<std::int64_t>(path) * kIntermediate;

        // Lanes 0..9 load the 10 scale words (tiles 0..9) for this row
        std::uint32_t my_tile = 0;
        if (lane < 10) {
            const std::int64_t off = scale_row_base + static_cast<std::int64_t>(lane) * 512;
            my_tile = *reinterpret_cast<const std::uint32_t*>(exp_scales + off);
        }

        // Broadcast tile words across the warp with all 32 threads active (zero divergence)
        const std::uint32_t tile_g0 = __shfl_sync(0xFFFFFFFFU, my_tile, lane >> 2);
        const std::uint32_t tile_g1 = __shfl_sync(0xFFFFFFFFU, my_tile, 8 + (lane >> 2));

        const std::uint8_t scale_g0 = static_cast<std::uint8_t>(tile_g0 >> ((lane & 3) * 8));
        const float coef_g0 = ops::detail::decode_nvfp4_e4m3(scale_g0) * inverse;

        const std::uint8_t scale_g1 = static_cast<std::uint8_t>(tile_g1 >> ((lane & 3) * 8));
        const float coef_g1 = ops::detail::decode_nvfp4_e4m3(scale_g1) * inverse;

        float sum0 = 0.0F;
        float sum1 = 0.0F;

        // Group 0..31: all 32 lanes active
        {
            const int group = lane;
            const auto* packed =
                exp_codes + static_cast<std::int64_t>(row) * (kIntermediate / 2) + group * 8;
            const uint2 code_words = *reinterpret_cast<const uint2*>(packed);

            const auto* act_ptr = reinterpret_cast<const uint4*>(act_path + group * 16);
            const uint4 act_v0 = act_ptr[0];
            const uint4 act_v1 = act_ptr[1];

            const float2 dw[8] = {
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.x)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.x >> 8)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.x >> 16)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.x >> 24)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.y)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.y >> 8)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.y >> 16)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.y >> 24))
            };

            const float2 hv[8] = {
                ops::bf16x2_bits_to_float2(act_v0.x),
                ops::bf16x2_bits_to_float2(act_v0.y),
                ops::bf16x2_bits_to_float2(act_v0.z),
                ops::bf16x2_bits_to_float2(act_v0.w),
                ops::bf16x2_bits_to_float2(act_v1.x),
                ops::bf16x2_bits_to_float2(act_v1.y),
                ops::bf16x2_bits_to_float2(act_v1.z),
                ops::bf16x2_bits_to_float2(act_v1.w)
            };

#pragma unroll
            for (int pair = 0; pair < 8; ++pair) {
                sum0 = fmaf(dw[pair].x * coef_g0, hv[pair].x, sum0);
                sum1 = fmaf(dw[pair].y * coef_g0, hv[pair].y, sum1);
            }
        }

        // Group 32..39: lanes 0..7 active
        if (lane < 8) {
            const int group = lane + 32;
            const auto* packed =
                exp_codes + static_cast<std::int64_t>(row) * (kIntermediate / 2) + group * 8;
            const uint2 code_words = *reinterpret_cast<const uint2*>(packed);

            const auto* act_ptr = reinterpret_cast<const uint4*>(act_path + group * 16);
            const uint4 act_v0 = act_ptr[0];
            const uint4 act_v1 = act_ptr[1];

            const float2 dw[8] = {
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.x)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.x >> 8)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.x >> 16)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.x >> 24)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.y)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.y >> 8)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.y >> 16)),
                ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.y >> 24))
            };

            const float2 hv[8] = {
                ops::bf16x2_bits_to_float2(act_v0.x),
                ops::bf16x2_bits_to_float2(act_v0.y),
                ops::bf16x2_bits_to_float2(act_v0.z),
                ops::bf16x2_bits_to_float2(act_v0.w),
                ops::bf16x2_bits_to_float2(act_v1.x),
                ops::bf16x2_bits_to_float2(act_v1.y),
                ops::bf16x2_bits_to_float2(act_v1.z),
                ops::bf16x2_bits_to_float2(act_v1.w)
            };

#pragma unroll
            for (int pair = 0; pair < 8; ++pair) {
                sum0 = fmaf(dw[pair].x * coef_g1, hv[pair].x, sum0);
                sum1 = fmaf(dw[pair].y * coef_g1, hv[pair].y, sum1);
            }
        }

        const float value = ops::warp_reduce_sum(sum0 + sum1);
        if (lane == 0) { routed = fmaf(alpha[token * kTopK + path], value, routed); }
    }

    // 3. Vectorized 128-bit loads for shared expert dot product directly from L1/L2
    float shared_sum = 0.0F;
    const auto* shared_row = reinterpret_cast<const uint4*>(
        shared_down + static_cast<std::int64_t>(row) * kIntermediate);
    const auto* shared_act = reinterpret_cast<const uint4*>(
        token_activations + static_cast<std::int64_t>(kTopK) * kIntermediate);

#pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
        const uint4 w = shared_row[phase * 32 + lane];
        const uint4 a = shared_act[phase * 32 + lane];
        const float2 wv0 = ops::bf16x2_bits_to_float2(w.x);
        const float2 av0 = ops::bf16x2_bits_to_float2(a.x);
        const float2 wv1 = ops::bf16x2_bits_to_float2(w.y);
        const float2 av1 = ops::bf16x2_bits_to_float2(a.y);
        const float2 wv2 = ops::bf16x2_bits_to_float2(w.z);
        const float2 av2 = ops::bf16x2_bits_to_float2(a.z);
        const float2 wv3 = ops::bf16x2_bits_to_float2(w.w);
        const float2 av3 = ops::bf16x2_bits_to_float2(a.w);
        shared_sum = fmaf(wv0.x, av0.x, shared_sum);
        shared_sum = fmaf(wv0.y, av0.y, shared_sum);
        shared_sum = fmaf(wv1.x, av1.x, shared_sum);
        shared_sum = fmaf(wv1.y, av1.y, shared_sum);
        shared_sum = fmaf(wv2.x, av2.x, shared_sum);
        shared_sum = fmaf(wv2.y, av2.y, shared_sum);
        shared_sum = fmaf(wv3.x, av3.x, shared_sum);
        shared_sum = fmaf(wv3.y, av3.y, shared_sum);
    }
    if (lane < 16) {
        const uint4 w = shared_row[64 + lane];
        const uint4 a = shared_act[64 + lane];
        const float2 wv0 = ops::bf16x2_bits_to_float2(w.x);
        const float2 av0 = ops::bf16x2_bits_to_float2(a.x);
        const float2 wv1 = ops::bf16x2_bits_to_float2(w.y);
        const float2 av1 = ops::bf16x2_bits_to_float2(a.y);
        const float2 wv2 = ops::bf16x2_bits_to_float2(w.z);
        const float2 av2 = ops::bf16x2_bits_to_float2(a.z);
        const float2 wv3 = ops::bf16x2_bits_to_float2(w.w);
        const float2 av3 = ops::bf16x2_bits_to_float2(a.w);
        shared_sum = fmaf(wv0.x, av0.x, shared_sum);
        shared_sum = fmaf(wv0.y, av0.y, shared_sum);
        shared_sum = fmaf(wv1.x, av1.x, shared_sum);
        shared_sum = fmaf(wv1.y, av1.y, shared_sum);
        shared_sum = fmaf(wv2.x, av2.x, shared_sum);
        shared_sum = fmaf(wv2.y, av2.y, shared_sum);
        shared_sum = fmaf(wv3.x, av3.x, shared_sum);
        shared_sum = fmaf(wv3.y, av3.y, shared_sum);
    }

    const float shared_value = ops::warp_reduce_sum(shared_sum);

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
    std::int32_t* __restrict__ token_to_pos,
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
        if (token_to_pos != nullptr) {
            token_to_pos[i]  = pos;
        }
    }
}



// Step 2a: Grouped Expert Gate & Up Projection (SIMT W4A16 for small token counts T < 512)
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

// Step 2b: Grouped Expert Gate & Up Projection (Native NVFP4 Tensor Core MMA)
// Tile: 32 rows (16 Gate + 16 Up intermediate pairs) x 16 tokens per CTA (2 warps along M x 2 warps along N)
// CTA: 128 threads (4 warps). Warps (0,1) compute pairs 0..7; Warps (2,3) compute pairs 8..15.
// Shared memory: 40960 + 5120 (weights) + 20480 + 2560 (activations) = 69120 bytes (67.5 KB)
// Grid: (kIntermediate / 16, 512)
template <bool kAsyncStaging = false>
__global__ __launch_bounds__(128, 4) void flash_next_moe_prefill_gate_up_mma_kernel(
    const std::uint8_t* __restrict__ act_codes,     // [1280, tokens]
    const std::uint8_t* __restrict__ act_scales,    // [160, tokens]
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
    const int active_idx   = static_cast<int>(blockIdx.y);
    if (active_idx >= total_active) { return; }

    const int expert = active_experts[active_idx];
    const int count  = expert_counts[expert];
    if (count <= 0) { return; }

    const int pair_base = static_cast<int>(blockIdx.x) * 16;
    if (pair_base >= kIntermediate) { return; }

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int warp_m = warp >> 1; // 0 or 1 (pairs 0..7 or pairs 8..15)
    const int warp_n = warp & 1;  // 0 or 1 (tokens 0..7 or tokens 8..15)
    const int lane   = tid & 31;

    extern __shared__ alignas(16) std::uint8_t s_dyn_mem[];
    auto* s_w_codes    = s_dyn_mem;                          // 32 * 1280 = 40960 bytes
    auto* s_w_scales   = s_dyn_mem + 40960;                  // 32 * 164 = 5248 bytes
    auto* s_a_codes_0  = s_dyn_mem + 40960 + 5248;           // 16 * 1280 = 20480 bytes
    auto* s_a_scales_0 = s_dyn_mem + 40960 + 5248 + 20480;   // 16 * 164 = 2624 bytes
    auto* s_a_codes_1  = s_dyn_mem + 40960 + 5248 + 20480 + 2624; // 16 * 1280 = 20480 bytes
    auto* s_a_scales_1 = s_dyn_mem + 40960 + 5248 + 20480 + 2624 + 20480; // 16 * 164 = 2624 bytes

    const auto* exp_codes  = expert_codes + static_cast<std::uint64_t>(expert) * code_stride;
    const auto* exp_scales = expert_scales + static_cast<std::uint64_t>(expert) * scale_stride;
    const int offset       = expert_offsets[expert];
    const float alpha      = 1.0F / expert_divisors[expert];

    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;

    if constexpr (kAsyncStaging) {
        // Lever A: Cooperatively stage 32 rows of weights via direct hardware cp.async
        #pragma unroll
        for (int i = tid * 16; i < 32 * 1280; i += 128 * 16) {
            const int row = i / 1280;
            const int col = i - row * 1280;
            const int m_grp = row >> 4;   // 0 or 1
            const int r_in  = row & 15;   // 0..15
            const int p_off = m_grp * 8 + (r_in & 7);
            const int global_row = (r_in < 8) ? (pair_base + p_off) : (pair_base + p_off + kIntermediate);
            ops::cp_async<16, ops::Cache::cg>(s_w_codes + i,
                                              exp_codes + static_cast<std::int64_t>(global_row) * 1280 + col);
        }

        #pragma unroll
        for (int i = tid; i < 32 * 40; i += 128) {
            const int row  = i / 40;
            const int tile = i - row * 40;
            const int m_grp = row >> 4;
            const int r_in  = row & 15;
            const int p_off = m_grp * 8 + (r_in & 7);
            const int global_row = (r_in < 8) ? (pair_base + p_off) : (pair_base + p_off + kIntermediate);
            const int m_tile     = global_row / 128;
            const int row_inner  = global_row % 128;
            const int row_mod32  = row_inner & 31;
            const int row_quart  = row_inner >> 5;
            const std::int64_t off = static_cast<std::int64_t>(m_tile * 40 + tile) * 512 +
                                     row_mod32 * 16 + row_quart * 4;
            ops::cp_async<4, ops::Cache::ca>(s_w_scales + row * 164 + tile * 4,
                                             exp_scales + off);
        }

        // Lever B: Co-issue chunk-0 activations into Buffer 0 concurrently with weights
        #pragma unroll
        for (int i = tid * 16; i < 16 * 1280; i += 128 * 16) {
            const int tok_idx = i / 1280;
            const int col     = i - tok_idx * 1280;
            if (tok_idx < count) {
                const int pos = offset + tok_idx;
                const int tok = grouped_tokens[pos];
                ops::cp_async<16, ops::Cache::ca>(s_a_codes_0 + i,
                                                  act_codes + static_cast<std::int64_t>(tok) * 1280 + col);
            } else {
                *reinterpret_cast<uint4*>(s_a_codes_0 + i) = make_uint4(0, 0, 0, 0);
            }
        }

        #pragma unroll
        for (int i = tid; i < 16 * 40; i += 128) {
            const int tok_idx = i / 40;
            const int tile    = i - tok_idx * 40;
            if (tok_idx < count) {
                const int pos = offset + tok_idx;
                const int tok = grouped_tokens[pos];
                ops::cp_async<4, ops::Cache::ca>(s_a_scales_0 + tok_idx * 164 + tile * 4,
                                                 act_scales + static_cast<std::int64_t>(tok) * 160 + tile * 4);
            } else {
                *reinterpret_cast<std::uint32_t*>(s_a_scales_0 + tok_idx * 164 + tile * 4) = 0;
            }
        }

        ops::cp_commit();
        ops::cp_wait<0>();
        __syncthreads();

        // Lever C: Double-buffered pipelined execution across token chunks
        for (int t_chunk = 0; t_chunk < count; t_chunk += 16) {
            const int buf_idx   = (t_chunk >> 4) & 1;
            auto* curr_a_codes  = (buf_idx == 0) ? s_a_codes_0 : s_a_codes_1;
            auto* curr_a_scales = (buf_idx == 0) ? s_a_scales_0 : s_a_scales_1;
            auto* next_a_codes  = (buf_idx == 0) ? s_a_codes_1 : s_a_codes_0;
            auto* next_a_scales = (buf_idx == 0) ? s_a_scales_1 : s_a_scales_0;

            const int next_chunk = t_chunk + 16;
            if (next_chunk < count) {
                #pragma unroll
                for (int i = tid * 16; i < 16 * 1280; i += 128 * 16) {
                    const int tok_idx = i / 1280;
                    const int col     = i - tok_idx * 1280;
                    if (next_chunk + tok_idx < count) {
                        const int pos = offset + next_chunk + tok_idx;
                        const int tok = grouped_tokens[pos];
                        ops::cp_async<16, ops::Cache::ca>(next_a_codes + i,
                                                          act_codes + static_cast<std::int64_t>(tok) * 1280 + col);
                    } else {
                        *reinterpret_cast<uint4*>(next_a_codes + i) = make_uint4(0, 0, 0, 0);
                    }
                }

                #pragma unroll
                for (int i = tid; i < 16 * 40; i += 128) {
                    const int tok_idx = i / 40;
                    const int tile    = i - tok_idx * 40;
                    if (next_chunk + tok_idx < count) {
                        const int pos = offset + next_chunk + tok_idx;
                        const int tok = grouped_tokens[pos];
                        ops::cp_async<4, ops::Cache::ca>(next_a_scales + tok_idx * 164 + tile * 4,
                                                         act_scales + static_cast<std::int64_t>(tok) * 160 + tile * 4);
                    } else {
                        *reinterpret_cast<std::uint32_t*>(next_a_scales + tok_idx * 164 + tile * 4) = 0;
                    }
                }
                ops::cp_commit();
            }

            const int t_base    = t_chunk + warp_n * 8;
            const int cur_batch = max(0, min(8, count - t_base));

            if (cur_batch > 0) {
                float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const auto* warp_w_codes  = s_w_codes + warp_m * (16 * 1280);
                const auto* warp_w_scales = s_w_scales + warp_m * (16 * 164);
                const auto* warp_a_codes  = curr_a_codes + warp_n * (8 * 1280);
                const auto* warp_a_scales = curr_a_scales + warp_n * (8 * 164);

                #pragma unroll 4
                for (int k64 = 0; k64 < 40; ++k64) {
                    unsigned a[4];
                    unsigned b[2];

                    const int a_row = a_row_offset;
                    const auto* a_addr = warp_w_codes + a_row * 1280 + k64 * 32 + a_column_byte;
                    ops::ldmatrix_x4(a[0], a[1], a[2], a[3], ops::smem_addr(a_addr));

                    const int b_row = b_row_offset;
                    const auto* b_addr = warp_a_codes + b_row * 1280 + k64 * 32 + b_column_byte;
                    ops::ldmatrix_x2(b[0], b[1], ops::smem_addr(b_addr));

                    const uint32_t sfa = *reinterpret_cast<const uint32_t*>(warp_w_scales + sfa_row * 164 + k64 * 4);
                    const uint32_t sfb = *reinterpret_cast<const uint32_t*>(warp_a_scales + sfb_row * 164 + k64 * 4);

                    ops::mma_nvfp4_e4m3(accumulators[0], accumulators[1], accumulators[2], accumulators[3],
                                       a[0], a[1], a[2], a[3], b[0], b[1], sfa, sfb);
                }

                const int tok0 = 2 * (lane & 3);
                const int tok1 = tok0 + 1;
                const int local_pair = lane >> 2;
                const int pair = pair_base + warp_m * 8 + local_pair;

                const float gate0 = accumulators[0] * alpha;
                const float gate1 = accumulators[1] * alpha;
                const float up0   = accumulators[2] * alpha;
                const float up1   = accumulators[3] * alpha;

                if (tok0 < cur_batch) {
                    const int pos = offset + t_base + tok0;
                    const int tok = grouped_tokens[pos];
                    const int path = grouped_paths[pos];
                    activations[(static_cast<std::int64_t>(tok) * kPaths + path) * kIntermediate + pair] =
                        __float2bfloat16_rn(ops::silu(gate0) * up0);
                }
                if (tok1 < cur_batch) {
                    const int pos = offset + t_base + tok1;
                    const int tok = grouped_tokens[pos];
                    const int path = grouped_paths[pos];
                    activations[(static_cast<std::int64_t>(tok) * kPaths + path) * kIntermediate + pair] =
                        __float2bfloat16_rn(ops::silu(gate1) * up1);
                }
            }

            if (next_chunk < count) {
                ops::cp_wait<0>();
                __syncthreads();
            }
        }
    } else {
        auto* s_a_codes  = s_a_codes_0;
        auto* s_a_scales = s_a_scales_0;

        // Cooperatively load 32 rows of weights (40960 bytes) across all 128 threads (20 iterations)
        #pragma unroll
        for (int i = tid * 16; i < 32 * 1280; i += 128 * 16) {
            const int row = i / 1280;
            const int col = i - row * 1280;
            const int m_grp = row >> 4;   // 0 or 1
            const int r_in  = row & 15;   // 0..15
            const int p_off = m_grp * 8 + (r_in & 7);
            const int global_row = (r_in < 8) ? (pair_base + p_off) : (pair_base + p_off + kIntermediate);
            *reinterpret_cast<uint4*>(s_w_codes + i) =
                *reinterpret_cast<const uint4*>(exp_codes + static_cast<std::int64_t>(global_row) * 1280 + col);
        }

        // Cooperatively unpack 1280 scale words (5120 bytes) across all 128 threads (10 iterations)
        // Stored with 164-byte row stride (41 words, coprime with 32 banks) to eliminate bank conflicts.
        #pragma unroll
        for (int i = tid; i < 32 * 40; i += 128) {
            const int row  = i / 40;
            const int tile = i - row * 40;
            const int m_grp = row >> 4;
            const int r_in  = row & 15;
            const int p_off = m_grp * 8 + (r_in & 7);
            const int global_row = (r_in < 8) ? (pair_base + p_off) : (pair_base + p_off + kIntermediate);
            const int m_tile     = global_row / 128;
            const int row_inner  = global_row % 128;
            const int row_mod32  = row_inner & 31;
            const int row_quart  = row_inner >> 5;
            const std::int64_t off = static_cast<std::int64_t>(m_tile * 40 + tile) * 512 +
                                     row_mod32 * 16 + row_quart * 4;
            *reinterpret_cast<std::uint32_t*>(s_w_scales + row * 164 + tile * 4) =
                *reinterpret_cast<const std::uint32_t*>(exp_scales + off);
        }
        __syncthreads();

        for (int t_chunk = 0; t_chunk < count; t_chunk += 16) {
            // Cooperatively load 16 tokens of activations across all 128 threads
            #pragma unroll
            for (int i = tid * 16; i < 16 * 1280; i += 128 * 16) {
                const int tok_idx = i / 1280;
                const int col     = i - tok_idx * 1280;
                if (t_chunk + tok_idx < count) {
                    const int pos = offset + t_chunk + tok_idx;
                    const int tok = grouped_tokens[pos];
                    *reinterpret_cast<uint4*>(s_a_codes + i) =
                        *reinterpret_cast<const uint4*>(act_codes + static_cast<std::int64_t>(tok) * 1280 + col);
                } else {
                    *reinterpret_cast<uint4*>(s_a_codes + i) = make_uint4(0, 0, 0, 0);
                }
            }

            #pragma unroll
            for (int i = tid; i < 16 * 40; i += 128) {
                const int tok_idx = i / 40;
                const int tile    = i - tok_idx * 40;
                if (t_chunk + tok_idx < count) {
                    const int pos = offset + t_chunk + tok_idx;
                    const int tok = grouped_tokens[pos];
                    *reinterpret_cast<std::uint32_t*>(s_a_scales + tok_idx * 164 + tile * 4) =
                        *reinterpret_cast<const std::uint32_t*>(act_scales + static_cast<std::int64_t>(tok) * 160 + tile * 4);
                } else {
                    *reinterpret_cast<std::uint32_t*>(s_a_scales + tok_idx * 164 + tile * 4) = 0;
                }
            }
            __syncthreads();

            const int t_base    = t_chunk + warp_n * 8;
            const int cur_batch = max(0, min(8, count - t_base));

            if (cur_batch > 0) {
                float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const auto* warp_w_codes  = s_w_codes + warp_m * (16 * 1280);
                const auto* warp_w_scales = s_w_scales + warp_m * (16 * 164);
                const auto* warp_a_codes  = s_a_codes + warp_n * (8 * 1280);
                const auto* warp_a_scales = s_a_scales + warp_n * (8 * 164);

                #pragma unroll 4
                for (int k64 = 0; k64 < 40; ++k64) {
                    unsigned a[4];
                    unsigned b[2];

                    const int a_row = a_row_offset;
                    const auto* a_addr = warp_w_codes + a_row * 1280 + k64 * 32 + a_column_byte;
                    ops::ldmatrix_x4(a[0], a[1], a[2], a[3], ops::smem_addr(a_addr));

                    const int b_row = b_row_offset;
                    const auto* b_addr = warp_a_codes + b_row * 1280 + k64 * 32 + b_column_byte;
                    ops::ldmatrix_x2(b[0], b[1], ops::smem_addr(b_addr));

                    const uint32_t sfa = *reinterpret_cast<const uint32_t*>(warp_w_scales + sfa_row * 164 + k64 * 4);
                    const uint32_t sfb = *reinterpret_cast<const uint32_t*>(warp_a_scales + sfb_row * 164 + k64 * 4);

                    ops::mma_nvfp4_e4m3(accumulators[0], accumulators[1], accumulators[2], accumulators[3],
                                       a[0], a[1], a[2], a[3], b[0], b[1], sfa, sfb);
                }

                const int tok0 = 2 * (lane & 3);
                const int tok1 = tok0 + 1;
                const int local_pair = lane >> 2;
                const int pair = pair_base + warp_m * 8 + local_pair;

                const float gate0 = accumulators[0] * alpha;
                const float gate1 = accumulators[1] * alpha;
                const float up0   = accumulators[2] * alpha;
                const float up1   = accumulators[3] * alpha;

                if (tok0 < cur_batch) {
                    const int pos = offset + t_base + tok0;
                    const int tok = grouped_tokens[pos];
                    const int path = grouped_paths[pos];
                    activations[(static_cast<std::int64_t>(tok) * kPaths + path) * kIntermediate + pair] =
                        __float2bfloat16_rn(ops::silu(gate0) * up0);
                }
                if (tok1 < cur_batch) {
                    const int pos = offset + t_base + tok1;
                    const int tok = grouped_tokens[pos];
                    const int path = grouped_paths[pos];
                    activations[(static_cast<std::int64_t>(tok) * kPaths + path) * kIntermediate + pair] =
                        __float2bfloat16_rn(ops::silu(gate1) * up1);
                }
            }
            __syncthreads();
        }
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
    const int pair         = static_cast<int>(blockIdx.x) * 8 + warp;
    const int token_base   = static_cast<int>(blockIdx.y) * 8;
    const int batch_tokens = min(8, tokens - token_base);

    if (pair >= kIntermediate || batch_tokens <= 0) { return; }

    const auto* g_row = shared_gate + static_cast<std::int64_t>(pair) * kHidden;
    const auto* u_row = shared_up + static_cast<std::int64_t>(pair) * kHidden;

    float gate_sum[8] = {};
    float up_sum[8]   = {};

    for (int col_base = 0; col_base < kHidden; col_base += 256) {
        const int col = col_base + lane * 8;
        if (col < kHidden) {
            const auto g_v = *reinterpret_cast<const uint4*>(&g_row[col]);
            const auto u_v = *reinterpret_cast<const uint4*>(&u_row[col]);
            const std::uint32_t g_raw[4] = {g_v.x, g_v.y, g_v.z, g_v.w};
            const std::uint32_t u_raw[4] = {u_v.x, u_v.y, u_v.z, u_v.w};

            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                if (b < batch_tokens) {
                    const int token  = token_base + b;
                    const auto* in_x = input + static_cast<std::int64_t>(token) * kHidden;
                    const auto x_v   = *reinterpret_cast<const uint4*>(&in_x[col]);
                    const std::uint32_t x_raw[4] = {x_v.x, x_v.y, x_v.z, x_v.w};

                    #pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const float2 g_pair = ops::bf16x2_bits_to_float2(g_raw[p]);
                        const float2 u_pair = ops::bf16x2_bits_to_float2(u_raw[p]);
                        const float2 x_pair = ops::bf16x2_bits_to_float2(x_raw[p]);
                        gate_sum[b] = fmaf(g_pair.x, x_pair.x, gate_sum[b]);
                        gate_sum[b] = fmaf(g_pair.y, x_pair.y, gate_sum[b]);
                        up_sum[b]   = fmaf(u_pair.x, x_pair.x, up_sum[b]);
                        up_sum[b]   = fmaf(u_pair.y, x_pair.y, up_sum[b]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        if (b < batch_tokens) {
            const int token    = token_base + b;
            const float g_tot  = ops::warp_reduce_sum(gate_sum[b]);
            const float u_tot  = ops::warp_reduce_sum(up_sum[b]);
            if (lane == 0) {
                activations[(static_cast<std::int64_t>(token) * kPaths + kTopK) * kIntermediate + pair] =
                    __float2bfloat16_rn(ops::silu(g_tot) * u_tot);
            }
        }
    }
}

// Step 4a: Grouped Expert Down Projection (SIMT W4A16 for small token counts T < 512)
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

// Step 4b: Grouped Expert Down GEMM (Native NVFP4 Tensor Core MMA with Fused Routing Alpha Epilogue)
// Tile: 64 output rows x 16 tokens per CTA (2 warps along M x 2 warps along N, 2 sub-tiles along M)
// CTA: 128 threads (4 warps).
// Shared memory: 20480 + 2560 (weights) + 5120 + 640 (activations) = 28800 bytes (28.125 KB <= 48 KB hardware limit)
// Grid: (kHidden / 64, 512)
__global__ __launch_bounds__(128, 4) void flash_next_moe_prefill_down_mma_kernel(
    const std::uint8_t* __restrict__ act_codes,     // [320, 11 * tokens]
    const std::uint8_t* __restrict__ act_scales,    // [40, 11 * tokens]
    const std::int32_t* __restrict__ expert_offsets,
    const std::int32_t* __restrict__ expert_counts,
    const std::int32_t* __restrict__ active_experts,
    const std::int32_t* __restrict__ active_count_ptr,
    const std::int32_t* __restrict__ grouped_tokens,
    const std::int32_t* __restrict__ grouped_paths,
    const float* __restrict__ alpha_weights,        // [10, tokens]
    const std::uint8_t* __restrict__ expert_codes,
    const std::uint8_t* __restrict__ expert_scales,
    const float* __restrict__ expert_divisors,
    std::uint64_t code_stride,
    std::uint64_t scale_stride,
    __nv_bfloat16* __restrict__ staged_down) {      // [10 * tokens, 2560]

    const int total_active = active_count_ptr[0];
    const int active_idx   = static_cast<int>(blockIdx.y);
    if (active_idx >= total_active) { return; }

    const int expert = active_experts[active_idx];
    const int count  = expert_counts[expert];
    if (count <= 0) { return; }

    const int row_base = static_cast<int>(blockIdx.x) * 64;
    if (row_base >= kHidden) { return; }

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int warp_m = warp >> 1; // 0 or 1 (rows 0..31 or rows 32..63)
    const int warp_n = warp & 1;  // 0 or 1 (tokens 0..7 or tokens 8..15)
    const int lane   = tid & 31;

    __shared__ alignas(16) std::uint8_t s_w_codes[4][16 * 320];
    __shared__ alignas(16) std::uint8_t s_w_scales[4][16 * 40];
    __shared__ alignas(16) std::uint8_t s_a_codes[2][8 * 320];
    __shared__ alignas(16) std::uint8_t s_a_scales[2][8 * 40];

    const auto* exp_codes  = expert_codes + static_cast<std::uint64_t>(expert) * code_stride;
    const auto* exp_scales = expert_scales + static_cast<std::uint64_t>(expert) * scale_stride;
    const int offset       = expert_offsets[expert];
    const float inv_div    = 1.0F / expert_divisors[expert];

    // Cooperatively load 64 rows of Down weights (20480 bytes) across all 128 threads (10 iterations)
    #pragma unroll
    for (int i = tid * 16; i < 64 * 320; i += 128 * 16) {
        const int row = i / 320;
        const int col = i - row * 320;
        const int global_row = row_base + row;
        const int m_grp = row >> 4;
        const int r_in  = row & 15;
        *reinterpret_cast<uint4*>(s_w_codes[m_grp] + r_in * 320 + col) =
            *reinterpret_cast<const uint4*>(exp_codes + static_cast<std::int64_t>(global_row) * 320 + col);
    }

    // Cooperatively unpack 640 scale words (2560 bytes) across all 128 threads (5 iterations)
    #pragma unroll
    for (int i = tid; i < 64 * 10; i += 128) {
        const int row  = i / 10;
        const int tile = i - row * 10;
        const int global_row = row_base + row;
        const int m_tile     = global_row / 128;
        const int row_inner  = global_row % 128;
        const int row_mod32  = row_inner & 31;
        const int row_quart  = row_inner >> 5;
        const std::int64_t off = static_cast<std::int64_t>(m_tile * 10 + tile) * 512 +
                                 row_mod32 * 16 + row_quart * 4;
        const int m_grp = row >> 4;
        const int r_in  = row & 15;
        *reinterpret_cast<std::uint32_t*>(s_w_scales[m_grp] + r_in * 40 + tile * 4) =
            *reinterpret_cast<const std::uint32_t*>(exp_scales + off);
    }
    __syncthreads();

    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;

    for (int t_chunk = 0; t_chunk < count; t_chunk += 16) {
        // Cooperatively load 16 tokens of intermediate activations across all 128 threads
        #pragma unroll
        for (int i = tid * 16; i < 16 * 320; i += 128 * 16) {
            const int tok_idx = i / 320;
            const int col     = i - tok_idx * 320;
            if (t_chunk + tok_idx < count) {
                const int pos     = offset + t_chunk + tok_idx;
                const int tok     = grouped_tokens[pos];
                const int path    = grouped_paths[pos];
                const int flat_id = tok * kPaths + path;
                *reinterpret_cast<uint4*>(s_a_codes[tok_idx / 8] + (tok_idx % 8) * 320 + col) =
                    *reinterpret_cast<const uint4*>(act_codes + static_cast<std::int64_t>(flat_id) * 320 + col);
            } else {
                *reinterpret_cast<uint4*>(s_a_codes[tok_idx / 8] + (tok_idx % 8) * 320 + col) = make_uint4(0, 0, 0, 0);
            }
        }

        #pragma unroll
        for (int i = tid * 4; i < 16 * 40; i += 128 * 4) {
            const int tok_idx = i / 40;
            const int col     = i - tok_idx * 40;
            if (t_chunk + tok_idx < count) {
                const int pos     = offset + t_chunk + tok_idx;
                const int tok     = grouped_tokens[pos];
                const int path    = grouped_paths[pos];
                const int flat_id = tok * kPaths + path;
                *reinterpret_cast<std::uint32_t*>(s_a_scales[tok_idx / 8] + (tok_idx % 8) * 40 + col) =
                    *reinterpret_cast<const std::uint32_t*>(act_scales + static_cast<std::int64_t>(flat_id) * 40 + col);
            } else {
                *reinterpret_cast<std::uint32_t*>(s_a_scales[tok_idx / 8] + (tok_idx % 8) * 40 + col) = 0;
            }
        }
        __syncthreads();

        const int t_base    = t_chunk + warp_n * 8;
        const int cur_batch = max(0, min(8, count - t_base));

        if (cur_batch > 0) {
            #pragma unroll
            for (int sub_m = 0; sub_m < 2; ++sub_m) {
                const int grp_idx = warp_m * 2 + sub_m;
                float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                #pragma unroll 2
                for (int k64 = 0; k64 < 10; ++k64) {
                    unsigned a[4];
                    unsigned b[2];

                    const int a_row = a_row_offset;
                    const auto* a_addr = s_w_codes[grp_idx] + a_row * 320 + k64 * 32 + a_column_byte;
                    ops::ldmatrix_x4(a[0], a[1], a[2], a[3], ops::smem_addr(a_addr));

                    const int b_row = b_row_offset;
                    const auto* b_addr = s_a_codes[warp_n] + b_row * 320 + k64 * 32 + b_column_byte;
                    ops::ldmatrix_x2(b[0], b[1], ops::smem_addr(b_addr));

                    const uint32_t sfa = *reinterpret_cast<const uint32_t*>(s_w_scales[grp_idx] + sfa_row * 40 + k64 * 4);
                    const uint32_t sfb = *reinterpret_cast<const uint32_t*>(s_a_scales[warp_n] + sfb_row * 40 + k64 * 4);

                    ops::mma_nvfp4_e4m3(accumulators[0], accumulators[1], accumulators[2], accumulators[3],
                                       a[0], a[1], a[2], a[3], b[0], b[1], sfa, sfb);
                }

                const int tok0 = 2 * (lane & 3);
                const int tok1 = tok0 + 1;
                const int row0 = row_base + grp_idx * 16 + (lane >> 2);
                const int row1 = row0 + 8;

                if (tok0 < cur_batch) {
                    const int pos  = offset + t_base + tok0;
                    const int tok  = grouped_tokens[pos];
                    const int path = grouped_paths[pos];
                    const float w  = alpha_weights[tok * kTopK + path] * inv_div;
                    staged_down[static_cast<std::int64_t>(pos) * kHidden + row0] = __float2bfloat16_rn(accumulators[0] * w);
                    staged_down[static_cast<std::int64_t>(pos) * kHidden + row1] = __float2bfloat16_rn(accumulators[2] * w);
                }
                if (tok1 < cur_batch) {
                    const int pos  = offset + t_base + tok1;
                    const int tok  = grouped_tokens[pos];
                    const int path = grouped_paths[pos];
                    const float w  = alpha_weights[tok * kTopK + path] * inv_div;
                    staged_down[static_cast<std::int64_t>(pos) * kHidden + row0] = __float2bfloat16_rn(accumulators[1] * w);
                    staged_down[static_cast<std::int64_t>(pos) * kHidden + row1] = __float2bfloat16_rn(accumulators[3] * w);
                }
            }
        }
        __syncthreads();
    }
}

// Step 5: Prefill Shared Expert Down Kernel (BF16 SIMT FMA, not a tensor-core MMA)
// Computes SharedDown(2560 x 640) * activations_shared(640 x tokens) and scales by shared_scale[token],
// initializing the layer output (BF16 [2560, tokens]).
// Grid: (kHidden / 64, (tokens + 15) / 16)
// Threads: 128 (4 warps).
__global__ __launch_bounds__(128, 4) void flash_next_moe_prefill_shared_down_kernel(
    const __nv_bfloat16* __restrict__ shared_down,
    const __nv_bfloat16* __restrict__ activations,
    const float* __restrict__ shared_scale,
    __nv_bfloat16* __restrict__ output,
    int tokens) {

    const int row_base   = static_cast<int>(blockIdx.x) * 64;
    const int token_base = static_cast<int>(blockIdx.y) * 16;
    if (row_base >= kHidden || token_base >= tokens) { return; }

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int warp_m = warp >> 1; // 0 or 1 (rows 0..31 or rows 32..63)
    const int warp_n = warp & 1;  // 0 or 1 (tokens 0..7 or tokens 8..15)
    const int lane   = tid & 31;

    __shared__ alignas(16) __nv_bfloat16 s_w[64 * 128]; // 16 KB
    __shared__ alignas(16) __nv_bfloat16 s_x[16 * 128]; // 4 KB

    float accum[2][4] = {}; // [sub_m][4 accumulators]
    const int t_offset = token_base + warp_n * 8;

    for (int k_stage = 0; k_stage < 5; ++k_stage) {
        const int k_base = k_stage * 128;

        // Load 64 rows x 128 cols weights (8192 elements = 16384 bytes)
        #pragma unroll
        for (int i = tid * 8; i < 64 * 128; i += 128 * 8) {
            const int r = i / 128;
            const int c = i % 128;
            *reinterpret_cast<uint4*>(&s_w[r * 128 + c]) =
                *reinterpret_cast<const uint4*>(&shared_down[static_cast<std::int64_t>(row_base + r) * kIntermediate + k_base + c]);
        }

        // Load 16 tokens x 128 cols activations (2048 elements = 4096 bytes)
        #pragma unroll
        for (int i = tid * 8; i < 16 * 128; i += 128 * 8) {
            const int tok_idx = i / 128;
            const int c       = i % 128;
            const int tok     = token_base + tok_idx;
            if (tok < tokens) {
                const auto* src = activations + (static_cast<std::int64_t>(tok) * kPaths + kTopK) * kIntermediate + k_base + c;
                *reinterpret_cast<uint4*>(&s_x[tok_idx * 128 + c]) =
                    *reinterpret_cast<const uint4*>(src);
            } else {
                *reinterpret_cast<uint4*>(&s_x[tok_idx * 128 + c]) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // Compute 32 rows x 8 tokens for this warp
        #pragma unroll
        for (int sub_m = 0; sub_m < 2; ++sub_m) {
            const int r_local0 = warp_m * 32 + sub_m * 16 + (lane >> 2);
            const int r_local1 = r_local0 + 8;
            const int tok0_local = warp_n * 8 + 2 * (lane & 3);
            const int tok1_local = tok0_local + 1;

            #pragma unroll
            for (int k_idx = 0; k_idx < 128; k_idx += 2) {
                const float2 w0 = ops::bf16x2_bits_to_float2(*reinterpret_cast<const uint32_t*>(&s_w[r_local0 * 128 + k_idx]));
                const float2 w1 = ops::bf16x2_bits_to_float2(*reinterpret_cast<const uint32_t*>(&s_w[r_local1 * 128 + k_idx]));
                const float2 x0 = ops::bf16x2_bits_to_float2(*reinterpret_cast<const uint32_t*>(&s_x[tok0_local * 128 + k_idx]));
                const float2 x1 = ops::bf16x2_bits_to_float2(*reinterpret_cast<const uint32_t*>(&s_x[tok1_local * 128 + k_idx]));

                accum[sub_m][0] = fmaf(w0.x, x0.x, fmaf(w0.y, x0.y, accum[sub_m][0]));
                accum[sub_m][1] = fmaf(w0.x, x1.x, fmaf(w0.y, x1.y, accum[sub_m][1]));
                accum[sub_m][2] = fmaf(w1.x, x0.x, fmaf(w1.y, x0.y, accum[sub_m][2]));
                accum[sub_m][3] = fmaf(w1.x, x1.x, fmaf(w1.y, x1.y, accum[sub_m][3]));
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int sub_m = 0; sub_m < 2; ++sub_m) {
        const int row0 = row_base + warp_m * 32 + sub_m * 16 + (lane >> 2);
        const int row1 = row0 + 8;
        const int tok0 = t_offset + 2 * (lane & 3);
        const int tok1 = tok0 + 1;

        if (tok0 < tokens) {
            const float s_scale0 = shared_scale[tok0];
            output[static_cast<std::int64_t>(tok0) * kHidden + row0] = __float2bfloat16_rn(accum[sub_m][0] * s_scale0);
            output[static_cast<std::int64_t>(tok0) * kHidden + row1] = __float2bfloat16_rn(accum[sub_m][2] * s_scale0);
        }
        if (tok1 < tokens) {
            const float s_scale1 = shared_scale[tok1];
            output[static_cast<std::int64_t>(tok1) * kHidden + row0] = __float2bfloat16_rn(accum[sub_m][1] * s_scale1);
            output[static_cast<std::int64_t>(tok1) * kHidden + row1] = __float2bfloat16_rn(accum[sub_m][3] * s_scale1);
        }
    }
}

// Step 6: Fused Fixed-Order FP32 Warp-Tile Reduction Kernel
// Sums SharedDown base + 10 routed paths in strictly fixed sequential order p = 0..9
// Grid: (kHidden / 64, (tokens + 15) / 16)
// Threads: 128 (4 warps).
__global__ __launch_bounds__(128, 4) void flash_next_moe_prefill_down_reduce_fused_kernel(
    const std::int32_t* __restrict__ token_to_pos,
    const __nv_bfloat16* __restrict__ staged_down,
    __nv_bfloat16* __restrict__ output,
    int tokens) {

    const int row_base   = static_cast<int>(blockIdx.x) * 64;
    const int token_base = static_cast<int>(blockIdx.y) * 16;
    if (row_base >= kHidden || token_base >= tokens) { return; }

    const int tid = static_cast<int>(threadIdx.x);

    #pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int local_idx = tid * 8 + item;
        const int r_off     = local_idx >> 4; // 0..63
        const int t_off     = local_idx & 15; // 0..15
        const int tok       = token_base + t_off;
        const int row       = row_base + r_off;

        if (tok < tokens) {
            const auto* base_ptr = output + static_cast<std::int64_t>(tok) * kHidden + row;
            float sum = __bfloat162float(*base_ptr);

            #pragma unroll
            for (int p = 0; p < 10; ++p) {
                const int pos = token_to_pos[tok * 10 + p];
                const float v = __bfloat162float(staged_down[static_cast<std::int64_t>(pos) * kHidden + row]);
                sum += v;
            }

            output[static_cast<std::int64_t>(tok) * kHidden + row] = __float2bfloat16_rn(sum);
        }
    }
}

// Step 5 (SIMT fallback): Prefill Shared Expert Down & Routed Path Reduction (Vectorized 128-bit loads for T < 512)
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

// Hybrid dispatch threshold:
// At T < 512 (e.g. T=128), average active tokens per expert is small (~2.7), where SIMT W4A16
// avoids activation quantization overhead and achieves lower latency.
// At T >= 512 (e.g. T=512, 2048), tokens per expert is high (>= 10), where Native NVFP4 MMA
// achieves 1.26x+ higher throughput.
constexpr int kMmaPrefillThreshold = 512;

void flash_next_moe_kernels_launch(const Tensor& input, const MoeWeights& weights,
                                   const FlashNextMoeWorkspace& workspace, Tensor& output,
                                   cudaStream_t stream) {
    const int tokens = static_cast<int>(input.ne[1]);

    if (tokens <= 8) {
        // Decode path (tokens <= 8): UNTOUCHED fused kernels
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
        // Prefill path (tokens > 8): Group tokens by expert, load weights once per chunk
        // 1. Group tokens by expert and build active expert list
        if (flash_next_moe_shared_mma_enabled()) {
            flash_next_moe_prefill_build_groups_mma(
                workspace.ids, workspace.expert_counts, workspace.expert_offsets,
                workspace.active_expert_ids, workspace.active_count, workspace.grouped_tokens,
                workspace.grouped_paths, workspace.grouped_experts, workspace.token_to_pos,
                workspace.task_counter, tokens, stream);
        } else {
            flash_next_moe_prefill_build_groups_kernel<<<1, 512, 0, stream>>>(
                static_cast<const std::int32_t*>(workspace.ids.data),
                static_cast<std::int32_t*>(workspace.expert_counts.data),
                static_cast<std::int32_t*>(workspace.expert_offsets.data),
                static_cast<std::int32_t*>(workspace.active_expert_ids.data),
                static_cast<std::int32_t*>(workspace.active_count.data),
                static_cast<std::int32_t*>(workspace.grouped_tokens.data),
                static_cast<std::int32_t*>(workspace.grouped_paths.data),
                static_cast<std::int32_t*>(workspace.grouped_experts.data),
                static_cast<std::int32_t*>(workspace.token_to_pos.data),
                static_cast<std::int32_t*>(workspace.task_counter.data),
                tokens);
            CUDA_CHECK(cudaGetLastError());
        }
        stage_ledger_record(stream, FlashNextStageId::MoE_Grouping);

        if (tokens >= kMmaPrefillThreshold) {
            // Large tokens: Native NVFP4 Tensor Core MMA route
            // 2. Shared expert gate & up. Completely disjoint from routed MMA.
            if (flash_next_moe_shared_mma_enabled()) {
                flash_next_moe_prefill_shared_gate_up_mma(
                    input, weights.shared_gate, weights.shared_up, workspace.activations,
                    workspace.shared_gemm, tokens, stream);
            } else {
                const dim3 shared_grid(kIntermediate / 8, (static_cast<unsigned>(tokens) + 7) / 8);
                flash_next_moe_prefill_shared_gate_up_kernel<<<shared_grid, 256, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(input.data),
                    static_cast<const __nv_bfloat16*>(weights.shared_gate.qdata),
                    static_cast<const __nv_bfloat16*>(weights.shared_up.qdata),
                    static_cast<__nv_bfloat16*>(workspace.activations.data),
                    tokens);
                CUDA_CHECK(cudaGetLastError());
            }
            stage_ledger_record(stream, FlashNextStageId::MoE_SharedGateUp);

            // 3. Quantize input activations X -> NVFP4
            constexpr int kActThreads = 256;
            const int act_tasks = tokens * Activation2560Geometry::kGroupsPerRow;
            ops::detail::nvfp4_w4a4_quantize_kernel<Activation2560Geometry>
                <<<(act_tasks + kActThreads - 1) / kActThreads, kActThreads, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(input.data),
                    static_cast<std::uint8_t*>(workspace.act_codes.data),
                    static_cast<std::uint8_t*>(workspace.act_scales.data),
                    tokens, 1.0F);
            CUDA_CHECK(cudaGetLastError());
            stage_ledger_record(stream, FlashNextStageId::MoE_QuantInput);

            // 4. Grouped Expert Gate & Up (Native NVFP4 Tensor Core MMA)
            static const bool s_mma_smem_init = []() {
                cudaFuncSetAttribute(flash_next_moe_prefill_gate_up_mma_kernel<false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, 69312);
                cudaFuncSetAttribute(flash_next_moe_prefill_gate_up_mma_kernel<true>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, 92416);
                return true;
            }();
            (void)s_mma_smem_init;

            const char* staging_env = std::getenv("NINFER_FLASH_NEXT_MOE_STAGING");
            const bool staging_new = staging_env && (std::strcmp(staging_env, "new") == 0 ||
                                                     std::strcmp(staging_env, "1") == 0);

            const dim3 gate_grid(kIntermediate / 16, 512);
            if (staging_new) {
                flash_next_moe_prefill_gate_up_mma_kernel<true><<<gate_grid, 128, 92416, stream>>>(
                    static_cast<const std::uint8_t*>(workspace.act_codes.data),
                    static_cast<const std::uint8_t*>(workspace.act_scales.data),
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
            } else {
                flash_next_moe_prefill_gate_up_mma_kernel<false><<<gate_grid, 128, 69312, stream>>>(
                    static_cast<const std::uint8_t*>(workspace.act_codes.data),
                    static_cast<const std::uint8_t*>(workspace.act_scales.data),
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
            }
            CUDA_CHECK(cudaGetLastError());
            stage_ledger_record(stream, FlashNextStageId::MoE_RoutedGateUp);

            // 5. Quantize intermediate activations -> NVFP4
            const int down_act_rows = tokens * kPaths;
            const int down_act_tasks = down_act_rows * Activation640Geometry::kGroupsPerRow;
            ops::detail::nvfp4_w4a4_quantize_kernel<Activation640Geometry>
                <<<(down_act_tasks + kActThreads - 1) / kActThreads, kActThreads, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(workspace.activations.data),
                    static_cast<std::uint8_t*>(workspace.down_act_codes.data),
                    static_cast<std::uint8_t*>(workspace.down_act_scales.data),
                    down_act_rows, 1.0F);
            CUDA_CHECK(cudaGetLastError());
            stage_ledger_record(stream, FlashNextStageId::MoE_QuantDown);

            // 6. Shared Down: SharedDown(2560 x 640) * shared_act * shared_scale -> output base
            if (flash_next_moe_shared_mma_enabled()) {
                flash_next_moe_prefill_shared_down_mma(weights.shared_down, workspace.shared_gemm,
                                                       workspace.shared_scale, output, tokens,
                                                       stream);
            } else {
                const dim3 shared_down_grid(kHidden / 64, (static_cast<unsigned>(tokens) + 15) / 16);
                flash_next_moe_prefill_shared_down_kernel<<<shared_down_grid, 128, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(weights.shared_down.qdata),
                    static_cast<const __nv_bfloat16*>(workspace.activations.data),
                    static_cast<const float*>(workspace.shared_scale.data),
                    static_cast<__nv_bfloat16*>(output.data),
                    tokens);
                CUDA_CHECK(cudaGetLastError());
            }
            stage_ledger_record(stream, FlashNextStageId::MoE_SharedDown);

            // 7. Grouped Down GEMM (Native NVFP4 Tensor Core MMA with Fused Routing Alpha Epilogue)
            const dim3 down_grid(kHidden / 64, 512);
            flash_next_moe_prefill_down_mma_kernel<<<down_grid, 128, 0, stream>>>(
                static_cast<const std::uint8_t*>(workspace.down_act_codes.data),
                static_cast<const std::uint8_t*>(workspace.down_act_scales.data),
                static_cast<const std::int32_t*>(workspace.expert_offsets.data),
                static_cast<const std::int32_t*>(workspace.expert_counts.data),
                static_cast<const std::int32_t*>(workspace.active_expert_ids.data),
                static_cast<const std::int32_t*>(workspace.active_count.data),
                static_cast<const std::int32_t*>(workspace.grouped_tokens.data),
                static_cast<const std::int32_t*>(workspace.grouped_paths.data),
                static_cast<const float*>(workspace.alpha.data),
                reinterpret_cast<const std::uint8_t*>(weights.expert_down.codes),
                reinterpret_cast<const std::uint8_t*>(weights.expert_down.scales),
                weights.expert_down.weight_scale_divisors,
                weights.expert_down.code_bytes_per_expert,
                weights.expert_down.scale_bytes_per_expert,
                static_cast<__nv_bfloat16*>(workspace.staged_down.data));
            CUDA_CHECK(cudaGetLastError());
            stage_ledger_record(stream, FlashNextStageId::MoE_RoutedDown);

            // 8. Fused Fixed-Order FP32 Warp-Tile Reduction Kernel
            const dim3 reduce_grid(kHidden / 64, (static_cast<unsigned>(tokens) + 15) / 16);
            flash_next_moe_prefill_down_reduce_fused_kernel<<<reduce_grid, 128, 0, stream>>>(
                static_cast<const std::int32_t*>(workspace.token_to_pos.data),
                static_cast<const __nv_bfloat16*>(workspace.staged_down.data),
                static_cast<__nv_bfloat16*>(output.data),
                tokens);
            CUDA_CHECK(cudaGetLastError());
            stage_ledger_record(stream, FlashNextStageId::MoE_Reduce);
        } else {
            // Small tokens (8 < tokens < 512): SIMT W4A16 route (avoids quant overhead)
            // 2. Grouped Expert Gate & Up (SIMT W4A16)
            constexpr int kGridY = 16;
            const dim3 gate_grid(kIntermediate / 8, kGridY);
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
            stage_ledger_record(stream, FlashNextStageId::MoE_RoutedGateUp);

            // 3. Shared expert gate & up
            if (flash_next_moe_shared_mma_enabled()) {
                flash_next_moe_prefill_shared_gate_up_mma(
                    input, weights.shared_gate, weights.shared_up, workspace.activations,
                    workspace.shared_gemm, tokens, stream);
            } else {
                const dim3 shared_grid(kIntermediate / 8, (static_cast<unsigned>(tokens) + 7) / 8);
                flash_next_moe_prefill_shared_gate_up_kernel<<<shared_grid, 256, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(input.data),
                    static_cast<const __nv_bfloat16*>(weights.shared_gate.qdata),
                    static_cast<const __nv_bfloat16*>(weights.shared_up.qdata),
                    static_cast<__nv_bfloat16*>(workspace.activations.data),
                    tokens);
                CUDA_CHECK(cudaGetLastError());
            }
            stage_ledger_record(stream, FlashNextStageId::MoE_SharedGateUp);

            // 4. Grouped Down GEMM (SIMT W4A16)
            constexpr int kDownGridY = 8;
            const dim3 down_grid(kHidden / 16, kDownGridY);
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
            stage_ledger_record(stream, FlashNextStageId::MoE_RoutedDown);

            // 5. Shared Down & Top-K weighted reduction (SIMT)
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
            stage_ledger_record(stream, FlashNextStageId::MoE_Reduce);
        }
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
