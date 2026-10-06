#pragma once

// Top-10 selection of the NVFP4-bank SparseMoe router: 513 FP32 scores per token (512 experts then
// the shared-expert gate), the ten largest experts with the lower id first at an exact tie, their
// renormalized route weights and the sigmoid shared-expert scale. One warp ranks one token. Every
// consumer of a token's selection (the decode gate/up prologue, its designated writer and the
// grouped selection kernel) runs this one algorithm over the same scores, so all of them agree.

#include "ops/sparse_moe/nvfp4_bank/nvfp4_bank_moe.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kNvfp4MoeScoreRows = kNvfp4MoeExperts + 1;
inline constexpr unsigned kNvfp4MoeFullMask = 0xFFFF'FFFFU;

static_assert(kNvfp4MoeExperts == 16 * 32, "each lane ranks exactly sixteen expert scores");
static_assert(kNvfp4MoeTopK <= 16, "a lane's sorted list never runs out before the tenth winner");

struct Nvfp4MoeRanked {
    float value;
    int id;
};

// Higher score first; an exact tie (float ==, so -0 ties +0) ranks the lower expert id first.
__device__ __forceinline__ bool nvfp4_moe_better(const Nvfp4MoeRanked& left,
                                                 const Nvfp4MoeRanked& right) {
    return left.value > right.value || (left.value == right.value && left.id < right.id);
}

__device__ __forceinline__ float nvfp4_moe_warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kNvfp4MoeFullMask, value, offset);
    }
    return __shfl_sync(kNvfp4MoeFullMask, value, 0);
}

__device__ __forceinline__ Nvfp4MoeRanked nvfp4_moe_warp_best(Nvfp4MoeRanked value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        Nvfp4MoeRanked other;
        other.value = __shfl_down_sync(kNvfp4MoeFullMask, value.value, offset);
        other.id    = __shfl_down_sync(kNvfp4MoeFullMask, value.id, offset);
        if (nvfp4_moe_better(other, value)) { value = other; }
    }
    value.value = __shfl_sync(kNvfp4MoeFullMask, value.value, 0);
    value.id    = __shfl_sync(kNvfp4MoeFullMask, value.id, 0);
    return value;
}

// Lane l owns experts l + 32*i (i = 0..15): their scores stay in registers and `taken` marks the
// ones already selected. Each tournament round takes every lane's best remaining candidate and
// the warp's best of those; the winner's lane is id & 31 and marks it taken. Ranking is a strict
// total order, so round r yields the r-th best expert.
struct Nvfp4MoeLaneScores {
    float value[16];
    unsigned taken;
};

__device__ __forceinline__ void nvfp4_moe_load_lane(const float* __restrict__ token_scores,
                                                    Nvfp4MoeLaneScores& lane_scores, int lane) {
#pragma unroll
    for (int item = 0; item < 16; ++item) {
        lane_scores.value[item] = __ldcg(token_scores + lane + item * 32);
    }
    lane_scores.taken = 0U;
}

__device__ __forceinline__ Nvfp4MoeRanked nvfp4_moe_pop_winner(Nvfp4MoeLaneScores& lane_scores,
                                                              int lane) {
    // At most ten of a lane's sixteen candidates are ever taken, so one always remains.
    Nvfp4MoeRanked best{0.0F, -1};
#pragma unroll
    for (int item = 0; item < 16; ++item) {
        const Nvfp4MoeRanked candidate{lane_scores.value[item], lane + item * 32};
        if (((lane_scores.taken >> item) & 1U) == 0U &&
            (best.id < 0 || nvfp4_moe_better(candidate, best))) {
            best = candidate;
        }
    }
    const Nvfp4MoeRanked winner = nvfp4_moe_warp_best(best);
    if (lane == (winner.id & 31)) { lane_scores.taken |= 1U << (winner.id >> 5); }
    return winner;
}

// The expert id selected at `rank` (0-based, < 10) for one token; the value is uniform over the
// warp. It is the rank-th winner of nvfp4_moe_select_token over the same scores.
__device__ __forceinline__ int nvfp4_moe_select_rank(const float* __restrict__ token_scores,
                                                     int rank, int lane) {
    Nvfp4MoeLaneScores lane_scores;
    nvfp4_moe_load_lane(token_scores, lane_scores, lane);
    Nvfp4MoeRanked winner = nvfp4_moe_pop_winner(lane_scores, lane);
    for (int round = 1; round <= rank; ++round) { winner = nvfp4_moe_pop_winner(lane_scores, lane); }
    return winner.id;
}

// The complete selection of one token: ids [10], renormalized weights [10] and the shared scale.
__device__ __forceinline__ void nvfp4_moe_select_token(const float* __restrict__ token_scores,
                                                       std::int32_t* __restrict__ ids,
                                                       float* __restrict__ weights,
                                                       float* __restrict__ shared_scale,
                                                       int lane) {
    Nvfp4MoeLaneScores lane_scores;
    nvfp4_moe_load_lane(token_scores, lane_scores, lane);
    float my_score = 0.0F; // lane r keeps the r-th selected score
#pragma unroll
    for (int rank = 0; rank < kNvfp4MoeTopK; ++rank) {
        const Nvfp4MoeRanked winner = nvfp4_moe_pop_winner(lane_scores, lane);
        if (lane == 0) { ids[rank] = winner.id; }
        if (lane == rank) { my_score = winner.value; }
    }
    // Softmax over all 512 experts followed by top-10 renormalization equals the softmax of the
    // ten selected scores: the full-softmax denominator cancels.
    const float top         = __shfl_sync(kNvfp4MoeFullMask, my_score, 0);
    const float weight      = lane < kNvfp4MoeTopK ? expf(my_score - top) : 0.0F;
    const float denominator = nvfp4_moe_warp_sum(weight);
    if (lane < kNvfp4MoeTopK) { weights[lane] = weight / denominator; }
    if (lane == 0) {
        const float gate = __ldcg(token_scores + kNvfp4MoeExperts);
        *shared_scale    = 1.0F / (1.0F + expf(-gate));
    }
}

} // namespace ninfer::ops::detail
