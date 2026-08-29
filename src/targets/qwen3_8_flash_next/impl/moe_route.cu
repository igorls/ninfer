#include "targets/qwen3_8_flash_next/impl/moe_route.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

constexpr int kExperts = 512;
constexpr int kTopK    = 10;
constexpr int kHidden  = 2'560;

struct RankedValue {
    float value;
    int id;
    int origin;
};

__device__ __forceinline__ bool better(const RankedValue& left, const RankedValue& right) {
    return left.value > right.value || (left.value == right.value && left.id < right.id);
}

__device__ __forceinline__ RankedValue warp_best(RankedValue value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        RankedValue other;
        other.value  = __shfl_down_sync(0xFFFF'FFFFU, value.value, offset);
        other.id     = __shfl_down_sync(0xFFFF'FFFFU, value.id, offset);
        other.origin = __shfl_down_sync(0xFFFF'FFFFU, value.origin, offset);
        if (better(other, value)) { value = other; }
    }
    value.value  = __shfl_sync(0xFFFF'FFFFU, value.value, 0);
    value.id     = __shfl_sync(0xFFFF'FFFFU, value.id, 0);
    value.origin = __shfl_sync(0xFFFF'FFFFU, value.origin, 0);
    return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFF'FFFFU, value, offset);
    }
    return __shfl_sync(0xFFFF'FFFFU, value, 0);
}

template <int Tokens>
__global__ void route_kernel(const float* __restrict__ scores, std::int32_t* __restrict__ ids,
                             float* __restrict__ alpha, float* __restrict__ shared_scale) {
    static_assert(Tokens >= 1 && Tokens <= 8);
    __shared__ float selected_logits[Tokens][kTopK];
    const int token           = static_cast<int>(threadIdx.x) >> 5;
    const int lane            = static_cast<int>(threadIdx.x) & 31;
    const float* token_scores = scores + static_cast<std::int64_t>(token) * (kExperts + 1);

    RankedValue local[16];
#pragma unroll
    for (int item = 0; item < 16; ++item) {
        const int id = lane + item * 32;
        local[item]  = {token_scores[id], id, lane};
    }
#pragma unroll
    for (int item = 1; item < 16; ++item) {
        const RankedValue value = local[item];
        int position            = item;
        while (position > 0 && better(value, local[position - 1])) {
            local[position] = local[position - 1];
            --position;
        }
        local[position] = value;
    }

    int cursor = 0;
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        RankedValue candidate =
            cursor < 16 ? local[cursor] : RankedValue{-CUDART_INF_F, INT_MAX, lane};
        const RankedValue winner = warp_best(candidate);
        if (lane == 0) {
            ids[token * kTopK + rank]    = winner.id;
            selected_logits[token][rank] = winner.value;
        }
        if (lane == winner.origin) { ++cursor; }
        __syncwarp();
    }
    const float top_0 = selected_logits[token][0];
    const float weight =
        lane < kTopK ? expf(selected_logits[token][lane] - top_0) : 0.0F;
    const float denominator = warp_sum(weight);
    if (lane < kTopK) {
        alpha[token * kTopK + lane] = weight / denominator;
    }
    if (lane == 0) {
        const float gate    = token_scores[kExperts];
        shared_scale[token] = 1.0F / (1.0F + expf(-gate));
    }
}

__device__ __forceinline__ float warp_sum_lane0(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFF'FFFFU, value, offset);
    }
    return value;
}

template <int Tokens>
__global__ void route_projection_kernel(const __nv_bfloat16* __restrict__ input,
                                        const __nv_bfloat16* __restrict__ router,
                                        const __nv_bfloat16* __restrict__ shared_gate,
                                        float* __restrict__ scores) {
    static_assert(Tokens >= 1 && Tokens <= 8);
    __shared__ float partial[8][Tokens];
    const int row  = static_cast<int>(blockIdx.x);
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const __nv_bfloat16* weight =
        row < kExperts ? router + static_cast<std::int64_t>(row) * kHidden : shared_gate;
    float accumulators[Tokens] = {};
    for (int column = tid; column < kHidden; column += static_cast<int>(blockDim.x)) {
        const float w = __bfloat162float(weight[column]);
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            accumulators[token] = fmaf(
                w, __bfloat162float(input[static_cast<std::int64_t>(token) * kHidden + column]),
                accumulators[token]);
        }
    }
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const float sum = warp_sum_lane0(accumulators[token]);
        if (lane == 0) { partial[warp][token] = sum; }
    }
    __syncthreads();
    if (warp == 0 && lane < Tokens) {
        float sum = 0.0F;
#pragma unroll
        for (int source_warp = 0; source_warp < 8; ++source_warp) {
            sum += partial[source_warp][lane];
        }
        scores[static_cast<std::int64_t>(lane) * (kExperts + 1) + row] = sum;
    }
}

template <class Launch>
void dispatch_tokens(std::int32_t tokens, Launch&& launch) {
    switch (tokens) {
    case 1:
        launch.template operator()<1>();
        return;
    case 2:
        launch.template operator()<2>();
        return;
    case 3:
        launch.template operator()<3>();
        return;
    case 4:
        launch.template operator()<4>();
        return;
    case 5:
        launch.template operator()<5>();
        return;
    case 6:
        launch.template operator()<6>();
        return;
    case 7:
        launch.template operator()<7>();
        return;
    case 8:
        launch.template operator()<8>();
        return;
    default:
        throw std::invalid_argument("Flash-Next routing supports startup-fixed T=1..8");
    }
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool exact_bf16_weight(const Weight& weight, std::int32_t rows, std::int32_t columns) {
    return weight.qtype == QType::BF16_CTRL && weight.layout == QuantLayout::Contiguous &&
           weight.n == rows && weight.k == columns && weight.ndim == 2 && weight.shape[0] == rows &&
           weight.shape[1] == columns && weight.padded_shape[0] == rows &&
           weight.padded_shape[1] == columns && weight.qdata == weight.payload &&
           weight.payload_bytes >= static_cast<std::uint64_t>(rows) * columns * 2 &&
           aligned_to(weight.qdata, 16);
}

} // namespace

void flash_next_route_scores(const Tensor& scores, Tensor& ids, Tensor& alpha, Tensor& shared_scale,
                             cudaStream_t stream) {
    const std::int32_t tokens = scores.ne[1];
    if (scores.dtype != DType::FP32 || ids.dtype != DType::I32 || alpha.dtype != DType::FP32 ||
        shared_scale.dtype != DType::FP32 || scores.ne[0] != kExperts + 1 || scores.ne[2] != 1 ||
        scores.ne[3] != 1 || ids.ne[0] != kTopK || ids.ne[1] != tokens || ids.ne[2] != 1 ||
        ids.ne[3] != 1 || alpha.ne[0] != kTopK || alpha.ne[1] != tokens || alpha.ne[2] != 1 ||
        alpha.ne[3] != 1 || shared_scale.ne[0] != tokens || shared_scale.ne[1] != 1 ||
        shared_scale.ne[2] != 1 || shared_scale.ne[3] != 1 || !scores.is_contiguous() ||
        !ids.is_contiguous() || !alpha.is_contiguous() || !shared_scale.is_contiguous() ||
        !aligned_to(scores.data, 16) || !aligned_to(ids.data, 16) || !aligned_to(alpha.data, 16) ||
        !aligned_to(shared_scale.data, 16) || stream == nullptr) {
        throw std::invalid_argument(
            "Flash-Next route requires aligned exact [513,T] top-10 tensors");
    }
    dispatch_tokens(tokens, [&]<int Tokens>() {
        route_kernel<Tokens><<<1, Tokens * 32, 0, stream>>>(
            static_cast<const float*>(scores.data), static_cast<std::int32_t*>(ids.data),
            static_cast<float*>(alpha.data), static_cast<float*>(shared_scale.data));
        CUDA_CHECK(cudaGetLastError());
    });
}

void flash_next_route(const Tensor& input, const Weight& router, const Weight& shared_gate,
                      Tensor& score_workspace, Tensor& ids, Tensor& alpha, Tensor& shared_scale,
                      cudaStream_t stream) {
    const std::int32_t tokens = input.ne[1];
    if (input.dtype != DType::BF16 || input.ne[0] != kHidden || input.ne[2] != 1 ||
        input.ne[3] != 1 || tokens < 1 || tokens > 8 || !input.is_contiguous() ||
        !aligned_to(input.data, 16) || !exact_bf16_weight(router, kExperts, kHidden) ||
        !exact_bf16_weight(shared_gate, 1, kHidden) || score_workspace.dtype != DType::FP32 ||
        score_workspace.ne[0] != kExperts + 1 || score_workspace.ne[1] != tokens ||
        score_workspace.ne[2] != 1 || score_workspace.ne[3] != 1 ||
        !score_workspace.is_contiguous() || !aligned_to(score_workspace.data, 16) ||
        stream == nullptr) {
        throw std::invalid_argument(
            "Flash-Next router requires exact aligned BF16 [2560,T] inputs");
    }
    dispatch_tokens(tokens, [&]<int Tokens>() {
        route_projection_kernel<Tokens>
            <<<kExperts + 1, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(input.data),
                                               static_cast<const __nv_bfloat16*>(router.qdata),
                                               static_cast<const __nv_bfloat16*>(shared_gate.qdata),
                                               static_cast<float*>(score_workspace.data));
        CUDA_CHECK(cudaGetLastError());
    });
    flash_next_route_scores(score_workspace, ids, alpha, shared_scale, stream);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
