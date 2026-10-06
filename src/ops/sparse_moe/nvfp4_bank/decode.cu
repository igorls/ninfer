// Short route (T <= 48) of the NVFP4-bank SparseMoe. Gate/up: one warp per (intermediate row,
// token, path) evaluates both the gate and up rows against the represented BF16 activation (A16),
// with the shared expert as path 10. Each routed CTA first selects its own path's expert from the
// token's router scores, which the router kernel wrote before this kernel began; the first
// shared-path CTA of each token writes the token's complete selection for the down kernel. The
// selection has no cross-CTA arrival counter, so the call carries no counter reset. Down: at
// T = 1 one CTA per output row gives each of the eleven paths its own warp so every weight of the
// row is in flight at once; from T = 2, where T x 320 CTAs fill the device, one warp per output
// row walks the paths.

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

constexpr int kHidden       = kNvfp4MoeHidden;
constexpr int kIntermediate = kNvfp4MoeIntermediate;
constexpr int kTopK         = kNvfp4MoeTopK;
constexpr int kPaths        = kNvfp4MoePaths;

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

__device__ __forceinline__ void dot_bf16_pair(const __nv_bfloat16* __restrict__ x,
                                              const __nv_bfloat16* __restrict__ first,
                                              const __nv_bfloat16* __restrict__ second,
                                              float& first_result, float& second_result) {
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    float first_sum  = 0.0F;
    float second_sum = 0.0F;
    for (int column = lane; column < kHidden; column += 32) {
        const float value = __bfloat162float(x[column]);
        first_sum         = fmaf(__bfloat162float(first[column]), value, first_sum);
        second_sum        = fmaf(__bfloat162float(second[column]), value, second_sum);
    }
    first_result  = warp_reduce_sum(first_sum);
    second_result = warp_reduce_sum(second_sum);
}

template <class GateSchedule>
__global__ void nvfp4_moe_decode_gate_up_kernel(
    const __nv_bfloat16* __restrict__ x, const float* __restrict__ scores,
    std::int32_t* __restrict__ ids, float* __restrict__ route_weights,
    float* __restrict__ shared_scale, const std::uint8_t* __restrict__ bank_codes,
    const std::uint8_t* __restrict__ bank_scales, const float* __restrict__ bank_divisors,
    std::uint64_t bank_code_stride, std::uint64_t bank_scale_stride,
    const __nv_bfloat16* __restrict__ shared_gate, const __nv_bfloat16* __restrict__ shared_up,
    __nv_bfloat16* __restrict__ activations) {
    const Nvfp4ExpertBankDevice bank{bank_codes, bank_scales, bank_divisors, bank_code_stride,
                                     bank_scale_stride};
    __shared__ Nvfp4A16GemvSharedStorage<GateGeometry, GateSchedule> shared;
    __shared__ int selected_expert;
    const int token      = static_cast<int>(blockIdx.y);
    const int path       = static_cast<int>(blockIdx.z);
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int row        = static_cast<int>(blockIdx.x) * GateSchedule::kWarpsPerCta + warp;
    const auto* x_t      = x + static_cast<std::int64_t>(token) * kHidden;
    const auto* scores_t = scores + static_cast<std::int64_t>(token) * kNvfp4MoeScoreRows;
    float gate           = 0.0F;
    float up             = 0.0F;
    if (path < kTopK) {
        // path is CTA-uniform, so every thread of a routed CTA reaches this barrier.
        if (warp == 0) {
            const int expert_id = nvfp4_moe_select_rank(scores_t, path, lane);
            if (lane == 0) { selected_expert = expert_id; }
        }
        __syncthreads();
        const Nvfp4ExpertPlanes expert = nvfp4_expert(bank, selected_expert);
        const int parent_rows[GateSchedule::kRowsPerWarp] = {row, row + kIntermediate};
        float accumulators[GateSchedule::kRowsPerWarp][GateSchedule::kAccumulatorChains] = {};
        compute_nvfp4_rows<GateGeometry, GateSchedule>(
            x_t, expert.codes, expert.scales, shared, expert.inverse_divisor, parent_rows,
            warp * GateSchedule::kRowsPerWarp, lane, accumulators);
#pragma unroll
        for (int chain = 0; chain < GateSchedule::kAccumulatorChains; ++chain) {
            gate += accumulators[0][chain];
            up += accumulators[1][chain];
        }
        gate = warp_reduce_sum(gate);
        up   = warp_reduce_sum(up);
    } else {
        dot_bf16_pair(x_t, shared_gate + static_cast<std::int64_t>(row) * kHidden,
                      shared_up + static_cast<std::int64_t>(row) * kHidden, gate, up);
        // The shared path needs no selection; its first CTA publishes the token's selection for
        // the down kernel, which runs after this kernel completes.
        if (blockIdx.x == 0 && warp == 0) {
            nvfp4_moe_select_token(scores_t, ids + token * kTopK, route_weights + token * kTopK,
                                   shared_scale + token, lane);
        }
    }
    if (lane == 0) {
        activations[(static_cast<std::int64_t>(token) * kPaths + path) * kIntermediate + row] =
            __float2bfloat16_rn(silu(gate) * up);
    }
}

// One K16 group of one NVFP4 down row against its 16 BF16 activations, as two FMA chains.
__device__ __forceinline__ void accumulate_down_group(const std::uint8_t* __restrict__ codes,
                                                      const __nv_bfloat16* __restrict__ product,
                                                      int row, int group, float coefficient,
                                                      float& sum0, float& sum1) {
    const uint2 words = *reinterpret_cast<const uint2*>(
        codes + static_cast<std::int64_t>(row) * (kIntermediate / 2) + group * 8);
    const auto* activation                 = reinterpret_cast<const uint4*>(product + group * 16);
    const uint4 a0                         = activation[0];
    const uint4 a1                         = activation[1];
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
// and broadcast. The result is valid in lane 0.
__device__ __forceinline__ float routed_down_value(const Nvfp4ExpertPlanes& expert,
                                                   const __nv_bfloat16* __restrict__ product,
                                                   int row, int lane) {
    std::uint32_t tile_word = 0;
    if (lane < kDownTiles) {
        tile_word = *reinterpret_cast<const std::uint32_t*>(
            expert.scales + nvfp4_expert_scale_word(row, lane, kDownTiles));
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
    accumulate_down_group(expert.codes, product, row, lane, coefficient0, sum0, sum1);
    if (lane < kDownGroups - 32) {
        accumulate_down_group(expert.codes, product, row, lane + 32, coefficient1, sum0, sum1);
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
    const auto* w = reinterpret_cast<const uint4*>(shared_down +
                                                   static_cast<std::int64_t>(row) * kIntermediate);
    const auto* a = reinterpret_cast<const uint4*>(product);
    float sum     = 0.0F;
    accumulate_bf16x8(w[lane], a[lane], sum);
    accumulate_bf16x8(w[32 + lane], a[32 + lane], sum);
    if (lane < 16) { accumulate_bf16x8(w[64 + lane], a[64 + lane], sum); }
    return warp_reduce_sum(sum);
}

__global__ void nvfp4_moe_decode_down_row_kernel(
    const std::int32_t* __restrict__ ids, const float* __restrict__ route_weights,
    const float* __restrict__ shared_scale, const __nv_bfloat16* __restrict__ activations,
    const std::uint8_t* __restrict__ bank_codes, const std::uint8_t* __restrict__ bank_scales,
    const float* __restrict__ bank_divisors, std::uint64_t bank_code_stride,
    std::uint64_t bank_scale_stride, const __nv_bfloat16* __restrict__ shared_down,
    __nv_bfloat16* __restrict__ output) {
    const Nvfp4ExpertBankDevice bank{bank_codes, bank_scales, bank_divisors, bank_code_stride,
                                     bank_scale_stride};
    const int token      = static_cast<int>(blockIdx.y);
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int row        = static_cast<int>(blockIdx.x) * kRowWarps + warp;
    const auto* products = activations + static_cast<std::int64_t>(token) * kPaths * kIntermediate;
    float routed         = 0.0F;
#pragma unroll
    for (int path = 0; path < kTopK; ++path) {
        const Nvfp4ExpertPlanes expert = nvfp4_expert(bank, ids[token * kTopK + path]);
        const float value              = routed_down_value(
            expert, products + static_cast<std::int64_t>(path) * kIntermediate, row, lane);
        if (lane == 0) { routed = fmaf(route_weights[token * kTopK + path], value, routed); }
    }
    const float shared = shared_down_value(
        shared_down, products + static_cast<std::int64_t>(kTopK) * kIntermediate, row, lane);
    if (lane == 0) {
        output[static_cast<std::int64_t>(token) * kHidden + row] =
            __float2bfloat16_rn(fmaf(shared_scale[token], shared, routed));
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
    const auto launch_gate_up = [&]<class GateSchedule>() {
        const dim3 grid(kIntermediate / GateSchedule::kWarpsPerCta, static_cast<unsigned>(tokens),
                        kPaths);
        nvfp4_moe_decode_gate_up_kernel<GateSchedule><<<grid, GateSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), scores, ids, route_weight, shared_scale,
            gate_up.codes, gate_up.scales, gate_up.divisors, gate_up.code_stride,
            gate_up.scale_stride,
            static_cast<const __nv_bfloat16*>(weights.shared_gate.qdata),
            static_cast<const __nv_bfloat16*>(weights.shared_up.qdata), activations);
        CUDA_CHECK(cudaGetLastError());
    };
    launch_gate_up.template operator()<GateScheduleT1>();

    const auto down       = nvfp4_expert_bank_device(weights.down);
    const auto* shared_dn = static_cast<const __nv_bfloat16*>(weights.shared_down.qdata);
    auto* output          = static_cast<__nv_bfloat16*>(destination.data);
    if (tokens <= kPathMaxTokens) {
        nvfp4_moe_decode_down_path_kernel<<<dim3(kHidden, static_cast<unsigned>(tokens)),
                                            kPathThreads, 0, stream>>>(
            ids, route_weight, shared_scale, activations, down.codes, down.scales, down.divisors,
            down.code_stride, down.scale_stride, shared_dn, output);
    } else {
        nvfp4_moe_decode_down_row_kernel<<<dim3(kHidden / kRowWarps, static_cast<unsigned>(tokens)),
                                           kRowWarps * 32, 0, stream>>>(
            ids, route_weight, shared_scale, activations, down.codes, down.scales, down.divisors,
            down.code_stride, down.scale_stride, shared_dn, output);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
