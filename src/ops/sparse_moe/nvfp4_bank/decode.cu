// Short route (T <= 48) of the NVFP4-bank SparseMoe. Gate/up: one warp per (intermediate row,
// token, path) evaluates both the gate and up rows against the represented BF16 activation (A16),
// with the shared expert at blockIdx.z == 0 (tiling up to 6 tokens per CTA so the 6.55 MB BF16
// shared gate/up weights are read once across speculative verify batches) and the ten routed
// experts at blockIdx.z == 1..10 (consuming the top-10 ids selected by nvfp4_moe_route). Down: at
// T = 1 one CTA per output row gives each of the eleven paths its own warp so every weight of the
// row is in flight at once; from T = 2, where T x 320 CTAs fill the device, one warp per output
// row walks the paths over shared-memory staged products and expert planes.

#include "ops/sparse_moe/nvfp4_bank/nvfp4_bank_moe.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_a16_gemv.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/sparse_moe/nvfp4_bank/expert_bank.cuh"
#include "ops/sparse_moe/nvfp4_bank/select.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden          = kNvfp4MoeHidden;
constexpr int kIntermediate    = kNvfp4MoeIntermediate;
constexpr int kTopK            = kNvfp4MoeTopK;
constexpr int kPaths           = kNvfp4MoePaths;
constexpr int kSharedTokenTile = 6;
constexpr int kStepsPerLane    = kHidden / (32 * 8);

using GateGeometry = Nvfp4Geometry<2 * kIntermediate, kHidden>;
// Keep the per-token reduction order across the complete speculative verification extent.
using GateScheduleT1 =
    Nvfp4A16GemvSchedule<8, 2, 8, 4, Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1>;

constexpr int kRowWarps        = 8;
constexpr int kPathMaxTokens   = 1; // down: one CTA per output row up to this T
constexpr int kPathThreads     = kPaths * 32;
constexpr int kDownTiles       = kIntermediate / 64; // scale words per down row
constexpr int kDownGroups      = kIntermediate / 16; // K16 groups per down row
constexpr int kPathBlocksPerSm = 4;

template <class GateSchedule, bool Preselected>
__global__ void __launch_bounds__(GateSchedule::kThreads, 4) nvfp4_moe_decode_gate_up_kernel(
    const __nv_bfloat16* __restrict__ x, const float* __restrict__ scores,
    std::int32_t* __restrict__ ids, float* __restrict__ route_weights,
    float* __restrict__ shared_scale, const std::uint8_t* __restrict__ bank_codes,
    const std::uint8_t* __restrict__ bank_scales, const float* __restrict__ bank_divisors,
    std::uint64_t bank_code_stride, std::uint64_t bank_scale_stride,
    const __nv_bfloat16* __restrict__ shared_gate, const __nv_bfloat16* __restrict__ shared_up,
    __nv_bfloat16* __restrict__ activations, int tokens) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row  = static_cast<int>(blockIdx.x) * GateSchedule::kWarpsPerCta + warp;

    if (blockIdx.z == 0) {
        constexpr int kTile = Preselected ? kSharedTokenTile : (kNvfp4MoePreselectMinTokens - 1);
        const int token_base = static_cast<int>(blockIdx.y) * kTile;
        if (token_base >= tokens) { return; }
        const int count   = min(kTile, tokens - token_base);
        const auto* g_row = reinterpret_cast<const uint4*>(
            shared_gate + static_cast<std::int64_t>(row) * kHidden);
        const auto* u_row = reinterpret_cast<const uint4*>(
            shared_up + static_cast<std::int64_t>(row) * kHidden);
        if constexpr (!Preselected) {
            if (count == 1) {
                const auto* x_vec = reinterpret_cast<const uint4*>(x);
                float gate_sum0 = 0.0F, gate_sum1 = 0.0F;
                float up_sum0 = 0.0F, up_sum1 = 0.0F;
#pragma unroll
                for (int j = 0; j < kStepsPerLane; ++j) {
                    const uint4 gw   = __ldg(g_row + j * 32 + lane);
                    const uint4 uw   = __ldg(u_row + j * 32 + lane);
                    const uint4 xv   = __ldg(x_vec + j * 32 + lane);
                    const float2 g01 = bf16x2_bits_to_float2(gw.x);
                    const float2 g23 = bf16x2_bits_to_float2(gw.y);
                    const float2 g45 = bf16x2_bits_to_float2(gw.z);
                    const float2 g67 = bf16x2_bits_to_float2(gw.w);
                    const float2 u01 = bf16x2_bits_to_float2(uw.x);
                    const float2 u23 = bf16x2_bits_to_float2(uw.y);
                    const float2 u45 = bf16x2_bits_to_float2(uw.z);
                    const float2 u67 = bf16x2_bits_to_float2(uw.w);
                    const float2 x01 = bf16x2_bits_to_float2(xv.x);
                    const float2 x23 = bf16x2_bits_to_float2(xv.y);
                    const float2 x45 = bf16x2_bits_to_float2(xv.z);
                    const float2 x67 = bf16x2_bits_to_float2(xv.w);
                    gate_sum0        = fmaf(g01.x, x01.x, gate_sum0);
                    gate_sum1        = fmaf(g01.y, x01.y, gate_sum1);
                    gate_sum0        = fmaf(g23.x, x23.x, gate_sum0);
                    gate_sum1        = fmaf(g23.y, x23.y, gate_sum1);
                    gate_sum0        = fmaf(g45.x, x45.x, gate_sum0);
                    gate_sum1        = fmaf(g45.y, x45.y, gate_sum1);
                    gate_sum0        = fmaf(g67.x, x67.x, gate_sum0);
                    gate_sum1        = fmaf(g67.y, x67.y, gate_sum1);
                    up_sum0          = fmaf(u01.x, x01.x, up_sum0);
                    up_sum1          = fmaf(u01.y, x01.y, up_sum1);
                    up_sum0          = fmaf(u23.x, x23.x, up_sum0);
                    up_sum1          = fmaf(u23.y, x23.y, up_sum1);
                    up_sum0          = fmaf(u45.x, x45.x, up_sum0);
                    up_sum1          = fmaf(u45.y, x45.y, up_sum1);
                    up_sum0          = fmaf(u67.x, x67.x, up_sum0);
                    up_sum1          = fmaf(u67.y, x67.y, up_sum1);
                }
                const float gate = warp_reduce_sum(gate_sum0 + gate_sum1);
                const float up   = warp_reduce_sum(up_sum0 + up_sum1);
                if (lane == 0) {
                    activations[static_cast<std::int64_t>(kTopK) * kIntermediate + row] =
                        __float2bfloat16_rn(silu(gate) * up);
                }
                if (blockIdx.x == 0 && warp == 0) {
                    nvfp4_moe_select_token(scores, ids, route_weights, shared_scale, lane);
                }
                return;
            }
        }
        float gate_acc[kTile][2] = {};
        float up_acc[kTile][2]   = {};
#pragma unroll
        for (int j = 0; j < kStepsPerLane; ++j) {
            const uint4 gw   = __ldg(g_row + j * 32 + lane);
            const uint4 uw   = __ldg(u_row + j * 32 + lane);
            const float2 g01 = bf16x2_bits_to_float2(gw.x);
            const float2 g23 = bf16x2_bits_to_float2(gw.y);
            const float2 g45 = bf16x2_bits_to_float2(gw.z);
            const float2 g67 = bf16x2_bits_to_float2(gw.w);
            const float2 u01 = bf16x2_bits_to_float2(uw.x);
            const float2 u23 = bf16x2_bits_to_float2(uw.y);
            const float2 u45 = bf16x2_bits_to_float2(uw.z);
            const float2 u67 = bf16x2_bits_to_float2(uw.w);
#pragma unroll
            for (int b = 0; b < kTile; ++b) {
                if (b < count) {
                    const auto* x_vec = reinterpret_cast<const uint4*>(
                        x + static_cast<std::int64_t>(token_base + b) * kHidden);
                    const uint4 xv   = __ldg(x_vec + j * 32 + lane);
                    const float2 x01 = bf16x2_bits_to_float2(xv.x);
                    const float2 x23 = bf16x2_bits_to_float2(xv.y);
                    const float2 x45 = bf16x2_bits_to_float2(xv.z);
                    const float2 x67 = bf16x2_bits_to_float2(xv.w);
                    gate_acc[b][0]   = fmaf(g01.x, x01.x, gate_acc[b][0]);
                    gate_acc[b][1]   = fmaf(g01.y, x01.y, gate_acc[b][1]);
                    gate_acc[b][0]   = fmaf(g23.x, x23.x, gate_acc[b][0]);
                    gate_acc[b][1]   = fmaf(g23.y, x23.y, gate_acc[b][1]);
                    gate_acc[b][0]   = fmaf(g45.x, x45.x, gate_acc[b][0]);
                    gate_acc[b][1]   = fmaf(g45.y, x45.y, gate_acc[b][1]);
                    gate_acc[b][0]   = fmaf(g67.x, x67.x, gate_acc[b][0]);
                    gate_acc[b][1]   = fmaf(g67.y, x67.y, gate_acc[b][1]);
                    up_acc[b][0]     = fmaf(u01.x, x01.x, up_acc[b][0]);
                    up_acc[b][1]     = fmaf(u01.y, x01.y, up_acc[b][1]);
                    up_acc[b][0]     = fmaf(u23.x, x23.x, up_acc[b][0]);
                    up_acc[b][1]     = fmaf(u23.y, x23.y, up_acc[b][1]);
                    up_acc[b][0]     = fmaf(u45.x, x45.x, up_acc[b][0]);
                    up_acc[b][1]     = fmaf(u45.y, x45.y, up_acc[b][1]);
                    up_acc[b][0]     = fmaf(u67.x, x67.x, up_acc[b][0]);
                    up_acc[b][1]     = fmaf(u67.y, x67.y, up_acc[b][1]);
                }
            }
        }
#pragma unroll
        for (int b = 0; b < kTile; ++b) {
            if (b < count) {
                const float gate = warp_reduce_sum(gate_acc[b][0] + gate_acc[b][1]);
                const float up   = warp_reduce_sum(up_acc[b][0] + up_acc[b][1]);
                if (lane == 0) {
                    activations[(static_cast<std::int64_t>(token_base + b) * kPaths + kTopK) *
                                    kIntermediate +
                                row] = __float2bfloat16_rn(silu(gate) * up);
                }
            }
        }
        if constexpr (!Preselected) {
            if (blockIdx.x == 0 && warp < count) {
                const int sel_token = token_base + warp;
                nvfp4_moe_select_token(
                    scores + static_cast<std::int64_t>(sel_token) * kNvfp4MoeScoreRows,
                    ids + sel_token * kTopK, route_weights + sel_token * kTopK,
                    shared_scale + sel_token, lane);
            }
        }
        return;
    }

    const Nvfp4ExpertBankDevice bank{bank_codes, bank_scales, bank_divisors, bank_code_stride,
                                     bank_scale_stride};
    __shared__ Nvfp4A16GemvSharedStorage<GateGeometry, GateSchedule> shared;
    __shared__ alignas(16) __nv_bfloat16 smem_x[kHidden];
    __shared__ Nvfp4ExpertPlanes smem_expert;

    const int token = static_cast<int>(blockIdx.y);
    const int path  = static_cast<int>(blockIdx.z) - 1;
    const int tid   = static_cast<int>(threadIdx.x);

    const auto* x_vec =
        reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(token) * kHidden);
    auto* s_vec = reinterpret_cast<uint4*>(smem_x);
    for (int idx = tid; idx < kHidden / 8; idx += GateSchedule::kThreads) {
        s_vec[idx] = __ldg(x_vec + idx);
    }
    if constexpr (Preselected) {
        if (tid == 0) { smem_expert = nvfp4_expert(bank, ids[token * kTopK + path]); }
    } else {
        if (warp == 0) {
            const auto* scores_t = scores + static_cast<std::int64_t>(token) * kNvfp4MoeScoreRows;
            const int expert_id  = nvfp4_moe_select_rank(scores_t, path, lane);
            if (lane == 0) { smem_expert = nvfp4_expert(bank, expert_id); }
        }
    }
    __syncthreads();

    const Nvfp4ExpertPlanes expert                    = smem_expert;
    const int parent_rows[GateSchedule::kRowsPerWarp] = {row, row + kIntermediate};
    float accumulators[GateSchedule::kRowsPerWarp][GateSchedule::kAccumulatorChains] = {};
    compute_nvfp4_rows<GateGeometry, GateSchedule>(
        smem_x, expert.codes, expert.scales, shared, expert.inverse_divisor, parent_rows,
        warp * GateSchedule::kRowsPerWarp, lane, accumulators);
    float gate = 0.0F;
    float up   = 0.0F;
#pragma unroll
    for (int chain = 0; chain < GateSchedule::kAccumulatorChains; ++chain) {
        gate += accumulators[0][chain];
        up += accumulators[1][chain];
    }
    gate = warp_reduce_sum(gate);
    up   = warp_reduce_sum(up);
    if (lane == 0) {
        activations[(static_cast<std::int64_t>(token) * kPaths + path) * kIntermediate + row] =
            __float2bfloat16_rn(silu(gate) * up);
    }
}

// One K16 group of one NVFP4 down row against its preloaded 16 BF16 activations, as two FMA chains.
__device__ __forceinline__ void accumulate_down_group_loaded(uint2 words, uint4 a0, uint4 a1,
                                                             float coefficient, float& sum0,
                                                             float& sum1) {
    const std::uint32_t code_words[2]      = {words.x, words.y};
    const std::uint32_t activation_bits[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const float2 w = decode_nvfp4_e2m1x2(
            static_cast<std::uint8_t>(code_words[pair >> 2] >> (8 * (pair & 3))));
        const float2 h = bf16x2_bits_to_float2(activation_bits[pair]);
        sum0           = fmaf(w.x * coefficient, h.x, sum0);
        sum1           = fmaf(w.y * coefficient, h.y, sum1);
    }
}

// Warp dot product of NVFP4 down row `row` of one expert with a 640-wide BF16 product. Lane l owns
// group l and, for l < 8, group l + 32; the ten scale words of the row are loaded by lanes 0..9
// and broadcast after issuing the code and product loads. The result is valid in lane 0.
__device__ __forceinline__ float routed_down_value(const Nvfp4ExpertPlanes& expert,
                                                   const __nv_bfloat16* __restrict__ product,
                                                   int row, int lane) {
    const auto* row_codes = expert.codes + static_cast<std::int64_t>(row) * (kIntermediate / 2);
    const auto* prod_vec  = reinterpret_cast<const uint4*>(product);

    const uint2 words0 = __ldg(reinterpret_cast<const uint2*>(row_codes + lane * 8));
    const uint4 a00    = prod_vec[lane * 2 + 0];
    const uint4 a01    = prod_vec[lane * 2 + 1];
    uint2 words1       = {0U, 0U};
    uint4 a10          = {0U, 0U, 0U, 0U};
    uint4 a11          = {0U, 0U, 0U, 0U};
    if (lane < kDownGroups - 32) {
        words1 = __ldg(reinterpret_cast<const uint2*>(row_codes + (lane + 32) * 8));
        a10    = prod_vec[(lane + 32) * 2 + 0];
        a11    = prod_vec[(lane + 32) * 2 + 1];
    }
    std::uint32_t tile_word = 0;
    if (lane < kDownTiles) {
        tile_word = __ldg(reinterpret_cast<const std::uint32_t*>(
            expert.scales + nvfp4_expert_scale_word(row, lane, kDownTiles)));
    }

    const std::uint32_t word0 = __shfl_sync(0xFFFF'FFFFU, tile_word, lane >> 2);
    const std::uint32_t word1 = __shfl_sync(0xFFFF'FFFFU, tile_word, 8 + (lane >> 2));
    const float coefficient0 =
        decode_nvfp4_e4m3(static_cast<std::uint8_t>(word0 >> ((lane & 3) * 8))) *
        expert.inverse_divisor;
    const float coefficient1 =
        decode_nvfp4_e4m3(static_cast<std::uint8_t>(word1 >> ((lane & 3) * 8))) *
        expert.inverse_divisor;
    float sum0 = 0.0F;
    float sum1 = 0.0F;
    accumulate_down_group_loaded(words0, a00, a01, coefficient0, sum0, sum1);
    if (lane < kDownGroups - 32) {
        accumulate_down_group_loaded(words1, a10, a11, coefficient1, sum0, sum1);
    }
    return warp_reduce_sum(sum0 + sum1);
}

__device__ __forceinline__ void accumulate_bf16x8(uint4 w, uint4 a, float& sum) {
    const std::uint32_t wb[4] = {w.x, w.y, w.z, w.w};
    const std::uint32_t ab[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 wv = bf16x2_bits_to_float2(wb[i]);
        const float2 av = bf16x2_bits_to_float2(ab[i]);
        sum             = fmaf(wv.x, av.x, sum);
        sum             = fmaf(wv.y, av.y, sum);
    }
}

// Warp BF16 dot product of shared_down row `row` with the shared product (80 x 16 bytes).
__device__ __forceinline__ float shared_down_value(const __nv_bfloat16* __restrict__ shared_down,
                                                   const __nv_bfloat16* __restrict__ product,
                                                   int row, int lane) {
    const auto* w  = reinterpret_cast<const uint4*>(shared_down +
                                                   static_cast<std::int64_t>(row) * kIntermediate);
    const auto* a  = reinterpret_cast<const uint4*>(product);
    const uint4 w0 = __ldg(w + lane);
    const uint4 a0 = a[lane];
    const uint4 w1 = __ldg(w + 32 + lane);
    const uint4 a1 = a[32 + lane];
    uint4 w2       = {0U, 0U, 0U, 0U};
    uint4 a2       = {0U, 0U, 0U, 0U};
    if (lane < 16) {
        w2 = __ldg(w + 64 + lane);
        a2 = a[64 + lane];
    }
    float sum = 0.0F;
    accumulate_bf16x8(w0, a0, sum);
    accumulate_bf16x8(w1, a1, sum);
    if (lane < 16) { accumulate_bf16x8(w2, a2, sum); }
    return warp_reduce_sum(sum);
}

template <bool StageProducts>
__global__ void __launch_bounds__(kRowWarps * 32, 4) nvfp4_moe_decode_down_row_kernel(
    const std::int32_t* __restrict__ ids, const float* __restrict__ route_weights,
    const float* __restrict__ shared_scale, const __nv_bfloat16* __restrict__ activations,
    const std::uint8_t* __restrict__ bank_codes, const std::uint8_t* __restrict__ bank_scales,
    const float* __restrict__ bank_divisors, std::uint64_t bank_code_stride,
    std::uint64_t bank_scale_stride, const __nv_bfloat16* __restrict__ shared_down,
    __nv_bfloat16* __restrict__ output) {
    const Nvfp4ExpertBankDevice bank{bank_codes, bank_scales, bank_divisors, bank_code_stride,
                                     bank_scale_stride};
    __shared__ alignas(16)
        __nv_bfloat16 smem_products[StageProducts ? (kPaths * kIntermediate) : 8];
    __shared__ Nvfp4ExpertPlanes smem_experts[kTopK];
    __shared__ float smem_weights[kTopK];
    __shared__ float smem_shared_scale;

    const int token      = static_cast<int>(blockIdx.y);
    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int row        = static_cast<int>(blockIdx.x) * kRowWarps + warp;
    const auto* products = activations + static_cast<std::int64_t>(token) * kPaths * kIntermediate;

    if constexpr (StageProducts) {
        const auto* prod_src = reinterpret_cast<const uint4*>(products);
        auto* prod_dst       = reinterpret_cast<uint4*>(smem_products);
        for (int idx = tid; idx < (kPaths * kIntermediate) / 8; idx += kRowWarps * 32) {
            prod_dst[idx] = __ldg(prod_src + idx);
        }
    }
    if (tid < kTopK) {
        smem_experts[tid] = nvfp4_expert(bank, ids[token * kTopK + tid]);
        smem_weights[tid] = route_weights[token * kTopK + tid];
    } else if (tid == kTopK) {
        smem_shared_scale = shared_scale[token];
    }
    __syncthreads();

    const __nv_bfloat16* active_products = StageProducts ? smem_products : products;
    float routed                         = 0.0F;
#pragma unroll
    for (int path = 0; path < kTopK; ++path) {
        const float value = routed_down_value(
            smem_experts[path], active_products + static_cast<std::int64_t>(path) * kIntermediate,
            row, lane);
        if (lane == 0) { routed = fmaf(smem_weights[path], value, routed); }
    }
    const float shared = shared_down_value(
        shared_down, active_products + static_cast<std::int64_t>(kTopK) * kIntermediate, row, lane);
    if (lane == 0) {
        output[static_cast<std::int64_t>(token) * kHidden + row] =
            __float2bfloat16_rn(fmaf(smem_shared_scale, shared, routed));
    }
}

__global__ void __launch_bounds__(kPathThreads, kPathBlocksPerSm) nvfp4_moe_decode_down_path_kernel(
    const std::int32_t* __restrict__ ids, const float* __restrict__ route_weights,
    const float* __restrict__ shared_scale, const __nv_bfloat16* __restrict__ activations,
    const std::uint8_t* __restrict__ bank_codes, const std::uint8_t* __restrict__ bank_scales,
    const float* __restrict__ bank_divisors, std::uint64_t bank_code_stride,
    std::uint64_t bank_scale_stride, const __nv_bfloat16* __restrict__ shared_down,
    __nv_bfloat16* __restrict__ output) {
    const Nvfp4ExpertBankDevice bank{bank_codes, bank_scales, bank_divisors, bank_code_stride,
                                     bank_scale_stride};
    __shared__ float path_value[kPaths];
    __shared__ float path_weight[kPaths];
    const int token      = static_cast<int>(blockIdx.y);
    const int row        = static_cast<int>(blockIdx.x);
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const auto* products = activations + static_cast<std::int64_t>(token) * kPaths * kIntermediate;
    float value          = 0.0F;
    if (warp < kTopK) {
        const Nvfp4ExpertPlanes expert = nvfp4_expert(bank, ids[token * kTopK + warp]);
        value                          = routed_down_value(
            expert, products + static_cast<std::int64_t>(warp) * kIntermediate, row, lane);
    } else {
        value = shared_down_value(
            shared_down, products + static_cast<std::int64_t>(kTopK) * kIntermediate, row, lane);
        if (lane < kTopK) { path_weight[lane] = route_weights[token * kTopK + lane]; }
        if (lane == kTopK) { path_weight[kTopK] = shared_scale[token]; }
    }
    if (lane == 0) { path_value[warp] = value; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float routed = 0.0F;
#pragma unroll
        for (int path = 0; path < kTopK; ++path) {
            routed = fmaf(path_weight[path], path_value[path], routed);
        }
        output[static_cast<std::int64_t>(token) * kHidden + row] =
            __float2bfloat16_rn(fmaf(path_weight[kTopK], path_value[kTopK], routed));
    }
}

} // namespace

void nvfp4_moe_decode(const Tensor& x, const SparseMoeNvfp4BankWeights& weights,
                      const Nvfp4MoeWorkspace& workspace, Tensor& destination,
                      cudaStream_t stream) {
    const int tokens         = x.ne[1];
    const auto* scores       = static_cast<const float*>(workspace.scores.data);
    auto* ids                = static_cast<std::int32_t*>(workspace.ids.data);
    auto* route_weight       = static_cast<float*>(workspace.weights.data);
    auto* shared_scale       = static_cast<float*>(workspace.shared_scale.data);
    auto* activations        = static_cast<__nv_bfloat16*>(workspace.activations.data);

    const auto gate_up        = nvfp4_expert_bank_device(weights.gate_up);
    const auto launch_gate_up = [&]<class GateSchedule, bool Preselected>() {
        const dim3 grid(kIntermediate / GateSchedule::kWarpsPerCta, static_cast<unsigned>(tokens),
                        kPaths);
        nvfp4_moe_decode_gate_up_kernel<GateSchedule, Preselected>
            <<<grid, GateSchedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data), scores, ids, route_weight, shared_scale,
                gate_up.codes, gate_up.scales, gate_up.divisors, gate_up.code_stride,
                gate_up.scale_stride,
                static_cast<const __nv_bfloat16*>(weights.shared_gate.qdata),
                static_cast<const __nv_bfloat16*>(weights.shared_up.qdata), activations, tokens);
        CUDA_CHECK(cudaGetLastError());
    };
    if (tokens >= kNvfp4MoePreselectMinTokens) {
        launch_gate_up.template operator()<GateScheduleT1, true>();
    } else {
        launch_gate_up.template operator()<GateScheduleT1, false>();
    }

    const auto down       = nvfp4_expert_bank_device(weights.down);
    const auto* shared_dn = static_cast<const __nv_bfloat16*>(weights.shared_down.qdata);
    auto* output          = static_cast<__nv_bfloat16*>(destination.data);
    if (tokens <= kPathMaxTokens) {
        nvfp4_moe_decode_down_path_kernel<<<dim3(kHidden, static_cast<unsigned>(tokens)),
                                            kPathThreads, 0, stream>>>(
            ids, route_weight, shared_scale, activations, down.codes, down.scales, down.divisors,
            down.code_stride, down.scale_stride, shared_dn, output);
    } else if (tokens >= kNvfp4MoePreselectMinTokens) {
        nvfp4_moe_decode_down_row_kernel<true>
            <<<dim3(kHidden / kRowWarps, static_cast<unsigned>(tokens)), kRowWarps * 32, 0,
               stream>>>(ids, route_weight, shared_scale, activations, down.codes, down.scales,
                         down.divisors, down.code_stride, down.scale_stride, shared_dn, output);
    } else {
        nvfp4_moe_decode_down_row_kernel<false>
            <<<dim3(kHidden / kRowWarps, static_cast<unsigned>(tokens)), kRowWarps * 32, 0,
               stream>>>(ids, route_weight, shared_scale, activations, down.codes, down.scales,
                         down.divisors, down.code_stride, down.scale_stride, shared_dn, output);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
