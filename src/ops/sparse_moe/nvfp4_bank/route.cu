// Router of the NVFP4-bank SparseMoe: 513 FP32 score rows (512 experts plus the shared-expert
// gate) per token. The decode route (T <= 8) only projects the scores; its gate/up kernel selects
// from them after the kernel boundary (decode.cu). Grouped routes project with the BF16 MMA and
// select in a separate kernel. Selection itself is select.cuh.

#include "ops/sparse_moe/nvfp4_bank/nvfp4_bank_moe.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_template_launch.cuh"
#include "ops/sparse_moe/nvfp4_bank/select.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kExperts   = kNvfp4MoeExperts;
constexpr int kTopK      = kNvfp4MoeTopK;
constexpr int kHidden    = kNvfp4MoeHidden;
constexpr int kScoreRows = kNvfp4MoeScoreRows;

// ---------------------------------------------------------------------------------------------
// Decode scores (T <= 8): 129 CTAs of four warps; each warp owns one score row and keeps its 2560
// weights in registers while looping over the tokens, so the router matrix is read once per call.
// No CTA waits for another: the selection runs in the next kernel.
// ---------------------------------------------------------------------------------------------
constexpr int kScoreRowsPerBlock = 4;
constexpr int kScoreThreads      = kScoreRowsPerBlock * 32;
constexpr int kScoreBlocks       = (kScoreRows + kScoreRowsPerBlock - 1) / kScoreRowsPerBlock;
constexpr int kStepsPerLane      = kHidden / (32 * 8);

static_assert(kHidden % 256 == 0, "a lane owns whole 16-byte chunks");

__device__ __forceinline__ float decode_row_score(const uint4 (&w)[kStepsPerLane],
                                                  const uint4* __restrict__ x_chunks, int lane) {
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { acc[i] = 0.0F; }
#pragma unroll
    for (int j = 0; j < kStepsPerLane; ++j) {
        const uint4 x_raw = __ldg(x_chunks + j * 32 + lane);
        const auto* w_bf  = reinterpret_cast<const __nv_bfloat16*>(&w[j]);
        const auto* x_bf  = reinterpret_cast<const __nv_bfloat16*>(&x_raw);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[i] = fmaf(__bfloat162float(w_bf[i]), __bfloat162float(x_bf[i]), acc[i]);
        }
    }
    float sum = ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]));
    return nvfp4_moe_warp_sum(sum);
}

__global__ void __launch_bounds__(kScoreThreads)
    nvfp4_moe_decode_scores_kernel(const __nv_bfloat16* __restrict__ x,
                                   const __nv_bfloat16* __restrict__ router,
                                   const __nv_bfloat16* __restrict__ shared_gate,
                                   float* __restrict__ scores, int tokens) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row  = static_cast<int>(blockIdx.x) * kScoreRowsPerBlock + warp;
    if (row >= kScoreRows) { return; }

    const __nv_bfloat16* weight =
        row < kExperts ? router + static_cast<std::int64_t>(row) * kHidden : shared_gate;
    const auto* w_chunks = reinterpret_cast<const uint4*>(weight);
    uint4 w[kStepsPerLane];
#pragma unroll
    for (int j = 0; j < kStepsPerLane; ++j) { w[j] = __ldg(w_chunks + j * 32 + lane); }
    for (int token = 0; token < tokens; ++token) {
        const auto* x_chunks =
            reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(token) * kHidden);
        const float score = decode_row_score(w, x_chunks, lane);
        if (lane == 0) { scores[static_cast<std::int64_t>(token) * kScoreRows + row] = score; }
    }
}

// Grouped-route selection: one warp per token over the projected [513,T] scores.
__global__ void __launch_bounds__(256)
    nvfp4_moe_select_kernel(const float* __restrict__ scores, std::int32_t* __restrict__ ids,
                            float* __restrict__ weights, float* __restrict__ shared_scale,
                            int tokens) {
    const int token = static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5);
    if (token >= tokens) { return; }
    nvfp4_moe_select_token(scores + static_cast<std::int64_t>(token) * kScoreRows,
                           ids + token * kTopK, weights + token * kTopK, shared_scale + token,
                           static_cast<int>(threadIdx.x) & 31);
}

// Shared-expert gate score row (row 512 of the score plane) for grouped routes.
__global__ void __launch_bounds__(256)
    nvfp4_moe_shared_gate_score_kernel(const __nv_bfloat16* __restrict__ x,
                                       const __nv_bfloat16* __restrict__ shared_gate,
                                       float* __restrict__ scores) {
    __shared__ float partial[8];
    const int token = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    const auto* x_t = x + static_cast<std::int64_t>(token) * kHidden;
    float acc       = 0.0F;
    for (int column = tid; column < kHidden; column += 256) {
        acc = fmaf(__bfloat162float(shared_gate[column]), __bfloat162float(x_t[column]), acc);
    }
    acc = nvfp4_moe_warp_sum(acc);
    if ((tid & 31) == 0) { partial[tid >> 5] = acc; }
    __syncthreads();
    if (tid < 32) {
        float sum = tid < 8 ? partial[tid] : 0.0F;
        sum       = nvfp4_moe_warp_sum(sum);
        if (tid == 0) { scores[static_cast<std::int64_t>(token) * kScoreRows + kExperts] = sum; }
    }
}

struct Fp32ScoreOutput {
    float* data;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        data[static_cast<std::int64_t>(token) * kScoreRows + row] = value;
    }
};

using RouterMmaSchedule =
    Bf16A16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;

} // namespace

void nvfp4_moe_route(const Tensor& x, const SparseMoeNvfp4BankWeights& weights,
                     const Nvfp4MoeWorkspace& workspace, cudaStream_t stream) {
    const int tokens   = x.ne[1];
    const auto* input  = static_cast<const __nv_bfloat16*>(x.data);
    const auto* router = static_cast<const __nv_bfloat16*>(weights.router.qdata);
    const auto* gate   = static_cast<const __nv_bfloat16*>(weights.shared_expert_gate.qdata);
    auto* scores       = static_cast<float*>(workspace.scores.data);

    if (tokens <= kNvfp4MoeDecodeMaxTokens) {
        nvfp4_moe_decode_scores_kernel<<<kScoreBlocks, kScoreThreads, 0, stream>>>(
            input, router, gate, scores, tokens);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    launch_bf16_a16_mma<RouterMmaSchedule>(
        Bf16A16Operands{input, router, kExperts, kHidden, tokens}, Fp32ScoreOutput{scores},
        LinearIdentityEpilogue{}, stream);
    nvfp4_moe_shared_gate_score_kernel<<<static_cast<unsigned>(tokens), 256, 0, stream>>>(
        input, gate, scores);
    CUDA_CHECK(cudaGetLastError());
    nvfp4_moe_select_kernel<<<static_cast<unsigned>((tokens + 7) / 8), 256, 0, stream>>>(
        scores, static_cast<std::int32_t*>(workspace.ids.data),
        static_cast<float*>(workspace.weights.data),
        static_cast<float*>(workspace.shared_scale.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
