// Router of the NVFP4-bank SparseMoe: 513 BF16 score rows (512 experts plus the shared-expert
// gate), top-10 selection with the lower id first at an exact tie, renormalized route weights and
// the sigmoid shared-expert scale.

#include "ops/sparse_moe/nvfp4_bank/nvfp4_bank_moe.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_template_launch.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kExperts   = kNvfp4MoeExperts;
constexpr int kTopK      = kNvfp4MoeTopK;
constexpr int kHidden    = kNvfp4MoeHidden;
constexpr int kScoreRows = kExperts + 1;

constexpr unsigned kFullMask = 0xFFFF'FFFFU;

static_assert(kExperts == 16 * 32, "each lane ranks exactly sixteen expert scores");
static_assert(kTopK <= 16, "a lane's sorted list never runs out before the tenth winner");

// Higher score first; an exact tie (float ==, so -0 ties +0) ranks the lower expert id first.
__device__ __forceinline__ bool better(float left_value, int left_id, float right_value,
                                       int right_id) {
    return left_value > right_value || (left_value == right_value && left_id < right_id);
}

struct Ranked {
    float value;
    int id;
};

__device__ __forceinline__ bool better(const Ranked& left, const Ranked& right) {
    return better(left.value, left.id, right.value, right.id);
}

__device__ __forceinline__ Ranked warp_best(Ranked value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        Ranked other;
        other.value = __shfl_down_sync(kFullMask, value.value, offset);
        other.id    = __shfl_down_sync(kFullMask, value.id, offset);
        if (better(other, value)) { value = other; }
    }
    value.value = __shfl_sync(kFullMask, value.value, 0);
    value.id    = __shfl_sync(kFullMask, value.id, 0);
    return value;
}

__device__ __forceinline__ float warp_all_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullMask, value, offset);
    }
    return __shfl_sync(kFullMask, value, 0);
}

// One warp selects one token's top-10 from its 513 scores. Lane l ranks experts l + 32*i in
// registers (static compare-exchange insertion); ten tournament rounds pop the best head. The
// winner's lane is id & 31 because every candidate carries id = lane + 32*item.
__device__ __forceinline__ void select_token(const float* __restrict__ token_scores,
                                             std::int32_t* __restrict__ ids,
                                             float* __restrict__ weights,
                                             float* __restrict__ shared_scale, int lane) {
    Ranked local[16];
#pragma unroll
    for (int item = 0; item < 16; ++item) {
        const int id = lane + item * 32;
        local[item]  = {__ldcg(token_scores + id), id};
    }
#pragma unroll
    for (int item = 1; item < 16; ++item) {
#pragma unroll
        for (int j = item; j > 0; --j) {
            if (better(local[j], local[j - 1])) {
                const Ranked moving = local[j];
                local[j]            = local[j - 1];
                local[j - 1]        = moving;
            }
        }
    }

    float my_score = 0.0F; // lane r keeps the r-th selected score
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        const Ranked winner = warp_best(local[0]);
        if (lane == 0) { ids[rank] = winner.id; }
        if (lane == rank) { my_score = winner.value; }
        if (lane == (winner.id & 31)) {
#pragma unroll
            for (int item = 0; item < 15; ++item) { local[item] = local[item + 1]; }
        }
    }
    // Softmax over all 512 experts followed by top-10 renormalization equals the softmax of the
    // ten selected scores: the full-softmax denominator cancels.
    const float top         = __shfl_sync(kFullMask, my_score, 0);
    const float weight      = lane < kTopK ? expf(my_score - top) : 0.0F;
    const float denominator = warp_all_sum(weight);
    if (lane < kTopK) { weights[lane] = weight / denominator; }
    if (lane == 0) {
        const float gate = __ldcg(token_scores + kExperts);
        *shared_scale    = 1.0F / (1.0F + expf(-gate));
    }
}

// ---------------------------------------------------------------------------------------------
// Decode router (T <= 8): projection, selection and gate in one launch. 129 CTAs of eight warps;
// warps 0..3 each own one score row and keep its 2560 weights in registers while looping over the
// tokens, so the router matrix is read once per call. The CTA completing the arrival count is the
// last of the launch and runs the selection with one warp per token. The counter lives in the
// call's workspace and is zeroed on the stream before the launch.
// ---------------------------------------------------------------------------------------------
constexpr int kFusedWarps        = 8;
constexpr int kFusedThreads      = kFusedWarps * 32;
constexpr int kFusedRowsPerBlock = 4;
constexpr int kFusedRowBlocks    = (kScoreRows + kFusedRowsPerBlock - 1) / kFusedRowsPerBlock;
constexpr int kStepsPerLane      = kHidden / (32 * 8);

static_assert(kHidden % 256 == 0, "a lane owns whole 16-byte chunks");
static_assert(kNvfp4MoeDecodeMaxTokens <= kFusedWarps, "selection has one warp per token");

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
    return warp_all_sum(sum);
}

__global__ void __launch_bounds__(kFusedThreads)
    nvfp4_moe_route_decode_kernel(const __nv_bfloat16* __restrict__ x,
                                  const __nv_bfloat16* __restrict__ router,
                                  const __nv_bfloat16* __restrict__ shared_gate,
                                  float* __restrict__ scores, std::int32_t* __restrict__ ids,
                                  float* __restrict__ weights, float* __restrict__ shared_scale,
                                  unsigned* __restrict__ arrivals, int tokens) {
    __shared__ bool last_arrival;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row  = static_cast<int>(blockIdx.x) * kFusedRowsPerBlock + warp;

    if (warp < kFusedRowsPerBlock && row < kScoreRows) {
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

    // Every writer fences its scores, the barrier orders them before thread 0's ticket, and the
    // CTA whose ticket completes the count reads the other CTAs' scores through L2.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned ticket = atomicAdd(arrivals, 1U);
        last_arrival          = ticket + 1U == static_cast<unsigned>(kFusedRowBlocks);
        __threadfence();
    }
    __syncthreads();
    if (!last_arrival || warp >= tokens) { return; }
    select_token(scores + static_cast<std::int64_t>(warp) * kScoreRows, ids + warp * kTopK,
                 weights + warp * kTopK, shared_scale + warp, lane);
}

// Grouped-route selection: one warp per token over the projected [513,T] scores.
__global__ void __launch_bounds__(256)
    nvfp4_moe_select_kernel(const float* __restrict__ scores, std::int32_t* __restrict__ ids,
                            float* __restrict__ weights, float* __restrict__ shared_scale,
                            int tokens) {
    const int token = static_cast<int>(blockIdx.x) * 8 + (static_cast<int>(threadIdx.x) >> 5);
    if (token >= tokens) { return; }
    select_token(scores + static_cast<std::int64_t>(token) * kScoreRows, ids + token * kTopK,
                 weights + token * kTopK, shared_scale + token, static_cast<int>(threadIdx.x) & 31);
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
    acc = warp_all_sum(acc);
    if ((tid & 31) == 0) { partial[tid >> 5] = acc; }
    __syncthreads();
    if (tid < 32) {
        float sum = tid < 8 ? partial[tid] : 0.0F;
        sum       = warp_all_sum(sum);
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
    auto* ids          = static_cast<std::int32_t*>(workspace.ids.data);
    auto* route_weight = static_cast<float*>(workspace.weights.data);
    auto* shared_scale = static_cast<float*>(workspace.shared_scale.data);

    if (tokens <= kNvfp4MoeDecodeMaxTokens) {
        CUDA_CHECK(cudaMemsetAsync(workspace.arrivals.data, 0, sizeof(unsigned), stream));
        nvfp4_moe_route_decode_kernel<<<kFusedRowBlocks, kFusedThreads, 0, stream>>>(
            input, router, gate, scores, ids, route_weight, shared_scale,
            static_cast<unsigned*>(workspace.arrivals.data), tokens);
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
        scores, ids, route_weight, shared_scale, tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
