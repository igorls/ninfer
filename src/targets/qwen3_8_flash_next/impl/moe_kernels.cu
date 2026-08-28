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

} // namespace

void flash_next_moe_kernels_launch(const Tensor& input, const MoeWeights& weights,
                                   const FlashNextMoeWorkspace& workspace, Tensor& output,
                                   cudaStream_t stream) {
    const dim3 gate_grid(kIntermediate / GateSchedule::kWarpsPerCta,
                         static_cast<unsigned>(input.ne[1]), kPaths);
    flash_next_moe_gate_up_kernel<<<gate_grid, GateSchedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data),
        static_cast<const std::int32_t*>(workspace.ids.data),
        reinterpret_cast<const std::uint8_t*>(weights.expert_gate_up.codes),
        reinterpret_cast<const std::uint8_t*>(weights.expert_gate_up.scales),
        weights.expert_gate_up.weight_scale_divisors, weights.expert_gate_up.code_bytes_per_expert,
        weights.expert_gate_up.scale_bytes_per_expert,
        static_cast<const __nv_bfloat16*>(weights.shared_gate.qdata),
        static_cast<const __nv_bfloat16*>(weights.shared_up.qdata),
        static_cast<__nv_bfloat16*>(workspace.activations.data));
    CUDA_CHECK(cudaGetLastError());

    const dim3 down_grid(kHidden / kDownWarps, static_cast<unsigned>(input.ne[1]));
    flash_next_moe_down_kernel<<<down_grid, kDownWarps * 32, 0, stream>>>(
        static_cast<const std::int32_t*>(workspace.ids.data),
        static_cast<const float*>(workspace.alpha.data),
        static_cast<const float*>(workspace.shared_scale.data),
        static_cast<const __nv_bfloat16*>(workspace.activations.data),
        reinterpret_cast<const std::uint8_t*>(weights.expert_down.codes),
        reinterpret_cast<const std::uint8_t*>(weights.expert_down.scales),
        weights.expert_down.weight_scale_divisors, weights.expert_down.code_bytes_per_expert,
        weights.expert_down.scale_bytes_per_expert,
        static_cast<const __nv_bfloat16*>(weights.shared_down.qdata),
        static_cast<__nv_bfloat16*>(output.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
