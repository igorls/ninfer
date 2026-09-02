#include "targets/qwen3_8_flash_next/impl/moe_route.h"

#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/moe_shared_kernels.h"

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

__global__ void route_kernel(const float* __restrict__ scores, std::int32_t* __restrict__ ids,
                             float* __restrict__ alpha, float* __restrict__ shared_scale,
                             int tokens) {
    __shared__ float selected_logits[kTopK];
    const int token           = static_cast<int>(blockIdx.x);
    const int lane            = static_cast<int>(threadIdx.x) & 31;
    if (token >= tokens) { return; }
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
            ids[token * kTopK + rank] = winner.id;
            selected_logits[rank]     = winner.value;
        }
        if (lane == winner.origin) { ++cursor; }
        __syncwarp();
    }
    const float top_0 = selected_logits[0];
    const float weight =
        lane < kTopK ? expf(selected_logits[lane] - top_0) : 0.0F;
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

__global__ void route_projection_kernel(const __nv_bfloat16* __restrict__ input,
                                        const __nv_bfloat16* __restrict__ router,
                                        const __nv_bfloat16* __restrict__ shared_gate,
                                        float* __restrict__ scores, int tokens) {
    __shared__ float partial[8];
    const int row   = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int tid   = static_cast<int>(threadIdx.x);
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    if (row >= kExperts + 1 || token >= tokens) { return; }

    const __nv_bfloat16* weight =
        row < kExperts ? router + static_cast<std::int64_t>(row) * kHidden : shared_gate;
    const __nv_bfloat16* in_tok = input + static_cast<std::int64_t>(token) * kHidden;

    float acc = 0.0F;
    for (int column = tid; column < kHidden; column += static_cast<int>(blockDim.x)) {
        acc = fmaf(__bfloat162float(weight[column]), __bfloat162float(in_tok[column]), acc);
    }
    const float sum = warp_sum_lane0(acc);
    if (lane == 0) { partial[warp] = sum; }
    __syncthreads();
    if (warp == 0) {
        float warp_sum = (lane < 8) ? partial[lane] : 0.0F;
        warp_sum       = warp_sum_lane0(warp_sum);
        if (lane == 0) {
            scores[static_cast<std::int64_t>(token) * (kExperts + 1) + row] = warp_sum;
        }
    }
}

// Prefill router projection: Tiled GEMM amortizing router weights over tokens
// Grid.x = (513 + 31) / 32 = 17 blocks, Grid.y = (tokens + 15) / 16 blocks.
// Inside each block, weights are loaded ONCE and multiplied with 16 tokens in registers.
__global__ void __launch_bounds__(256)
route_prefill_projection_kernel(const __nv_bfloat16* __restrict__ input,
                                const __nv_bfloat16* __restrict__ router,
                                const __nv_bfloat16* __restrict__ shared_gate,
                                float* __restrict__ scores, int tokens) {
    const int warp         = static_cast<int>(threadIdx.x) >> 5;
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int row_base     = (static_cast<int>(blockIdx.x) * 8 + warp) * 4;
    const int token_base   = static_cast<int>(blockIdx.y) * 16;
    const int batch_tokens = min(16, tokens - token_base);

    if (batch_tokens <= 0) { return; }

    float acc[4][16] = {};

    for (int chunk = lane; chunk < (kHidden / 8); chunk += 32) {
        const int col_base = chunk * 8;

        // 1. Load 16 token inputs for this chunk
        ulonglong2 x_raw[16];
        #pragma unroll
        for (int b = 0; b < 16; ++b) {
            if (b < batch_tokens) {
                const int t_idx = token_base + b;
                x_raw[b] = *reinterpret_cast<const ulonglong2*>(
                    input + static_cast<std::int64_t>(t_idx) * kHidden + col_base);
            }
        }

        // 2. Load router weight rows ONCE and multiply across all 16 tokens
        #pragma unroll
        for (int r_idx = 0; r_idx < 4; ++r_idx) {
            const int row = row_base + r_idx;
            if (row < kExperts + 1) {
                const __nv_bfloat16* weight =
                    row < kExperts ? router + static_cast<std::int64_t>(row) * kHidden : shared_gate;
                const auto w_raw = *reinterpret_cast<const ulonglong2*>(weight + col_base);
                const auto* w_bf = reinterpret_cast<const __nv_bfloat16*>(&w_raw);

                #pragma unroll
                for (int b = 0; b < 16; ++b) {
                    if (b < batch_tokens) {
                        const auto* x_bf = reinterpret_cast<const __nv_bfloat16*>(&x_raw[b]);
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            acc[r_idx][b] = fmaf(__bfloat162float(w_bf[i]), __bfloat162float(x_bf[i]), acc[r_idx][b]);
                        }
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int r_idx = 0; r_idx < 4; ++r_idx) {
        const int row = row_base + r_idx;
        if (row < kExperts + 1) {
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                if (b < batch_tokens) {
                    const float sum = warp_sum_lane0(acc[r_idx][b]);
                    if (lane == 0) {
                        const int t_idx = token_base + b;
                        scores[static_cast<std::int64_t>(t_idx) * (kExperts + 1) + row] = sum;
                    }
                }
            }
        }
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
    dim3 grid(static_cast<unsigned>(tokens), 1);
    route_kernel<<<grid, 32, 0, stream>>>(
        static_cast<const float*>(scores.data), static_cast<std::int32_t*>(ids.data),
        static_cast<float*>(alpha.data), static_cast<float*>(shared_scale.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_route(const Tensor& input, const Weight& router, const Weight& shared_gate,
                      Tensor& score_workspace, Tensor& ids, Tensor& alpha, Tensor& shared_scale,
                      cudaStream_t stream) {
    const std::int32_t tokens = input.ne[1];
    if (input.dtype != DType::BF16 || input.ne[0] != kHidden || input.ne[2] != 1 ||
        input.ne[3] != 1 || tokens < 1 || !input.is_contiguous() ||
        !aligned_to(input.data, 16) || !exact_bf16_weight(router, kExperts, kHidden) ||
        !exact_bf16_weight(shared_gate, 1, kHidden) || score_workspace.dtype != DType::FP32 ||
        score_workspace.ne[0] != kExperts + 1 || score_workspace.ne[1] != tokens ||
        score_workspace.ne[2] != 1 || score_workspace.ne[3] != 1 ||
        !score_workspace.is_contiguous() || !aligned_to(score_workspace.data, 16) ||
        stream == nullptr) {
        throw std::invalid_argument(
            "Flash-Next router requires exact aligned BF16 [2560,T] inputs");
    }
    if (tokens <= 8) {
        dim3 grid(static_cast<unsigned>(kExperts + 1), static_cast<unsigned>(tokens));
        route_projection_kernel<<<grid, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(input.data),
            static_cast<const __nv_bfloat16*>(router.qdata),
            static_cast<const __nv_bfloat16*>(shared_gate.qdata),
            static_cast<float*>(score_workspace.data), tokens);
        CUDA_CHECK(cudaGetLastError());
    } else if (flash_next_moe_shared_mma_enabled()) {
        flash_next_route_projection_mma(input, router, shared_gate, score_workspace, stream);
    } else {
        const dim3 grid((kExperts + 1 + 31) / 32, (static_cast<unsigned>(tokens) + 15) / 16);
        route_prefill_projection_kernel<<<grid, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(input.data),
            static_cast<const __nv_bfloat16*>(router.qdata),
            static_cast<const __nv_bfloat16*>(shared_gate.qdata),
            static_cast<float*>(score_workspace.data), tokens);
        CUDA_CHECK(cudaGetLastError());
    }

    flash_next_route_scores(score_workspace, ids, alpha, shared_scale, stream);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
