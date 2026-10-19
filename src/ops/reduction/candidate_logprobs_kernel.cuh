#pragma once

// Implements: include/ninfer/ops/candidate_logprobs.h
// Match: contiguous BF16 [physical_rows,C], I32 ids, optional I32 [mask_words] mask, FP32 outputs.
// Algorithm assumptions: one 256-thread CTA per column performs a stable two-pass logsumexp for
// the raw and the masked normaliser together, then each thread resolves a strided share of ids.

#include "ops/common/cuda/reduce.cuh"
#include "ops/reduction/target_logprobs_kernel.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCandidateLogprobsBlock   = 256;
inline constexpr int kCandidateLogprobsMaxTopK = 20;

__device__ __forceinline__ bool candidate_logprobs_allowed(const std::int32_t* allowed,
                                                           std::int32_t row) {
    if (allowed == nullptr) { return true; }
    const auto word = static_cast<std::uint32_t>(allowed[row >> 5]);
    return ((word >> (row & 31)) & 1U) != 0U;
}

__device__ __forceinline__ bool candidate_logprobs_better(float a_val, std::int32_t a_id,
                                                          float b_val, std::int32_t b_id) {
    if (a_id < 0) { return false; }
    if (b_id < 0) { return true; }
    return a_val > b_val || (a_val == b_val && a_id < b_id);
}

template <int BlockSize, bool HasTopK = false>
__launch_bounds__(BlockSize) __global__
    void candidate_logprobs_kernel(const __nv_bfloat16* logits, std::int32_t valid_rows,
                                   std::int32_t physical_rows, const std::int32_t* sampled_ids,
                                   const std::int32_t* candidate_ids, std::int32_t candidates,
                                   const std::int32_t* allowed, std::int32_t allowed_stride,
                                   float* sampled_out, float* candidates_out,
                                   std::int32_t* top_ids_out, float* top_out, std::int32_t top_k,
                                   std::int32_t columns) {
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x);
    if (allowed) { allowed += column * allowed_stride; }
    const std::int64_t base = static_cast<std::int64_t>(column) * physical_rows;

    float raw_max    = -CUDART_INF_F;
    float masked_max = -CUDART_INF_F;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        const float value = __bfloat162float(logits[base + row]);
        raw_max           = fmaxf(raw_max, value);
        if (candidate_logprobs_allowed(allowed, row)) { masked_max = fmaxf(masked_max, value); }
    }
    raw_max    = target_logprobs_block_max<BlockSize>(raw_max);
    masked_max = target_logprobs_block_max<BlockSize>(masked_max);

    __shared__ float top_vals[HasTopK ? kCandidateLogprobsMaxTopK : 1][HasTopK ? BlockSize : 1];
    __shared__ std::int32_t
        top_ids[HasTopK ? kCandidateLogprobsMaxTopK : 1][HasTopK ? BlockSize : 1];
    std::int32_t local_count = 0;
    float kth_val            = -CUDART_INF_F;

    float raw_sum    = 0.0f;
    float masked_sum = 0.0f;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        const float value      = __bfloat162float(logits[base + row]);
        const bool row_allowed = candidate_logprobs_allowed(allowed, row);
        raw_sum += expf(value - raw_max);
        if (row_allowed) {
            masked_sum += expf(value - masked_max);
            if constexpr (HasTopK) {
                if (local_count < top_k || value > kth_val) {
                    std::int32_t pos = local_count < top_k ? local_count++ : (top_k - 1);
                    while (pos > 0 && value > top_vals[pos - 1][threadIdx.x]) {
                        top_vals[pos][threadIdx.x] = top_vals[pos - 1][threadIdx.x];
                        top_ids[pos][threadIdx.x]  = top_ids[pos - 1][threadIdx.x];
                        --pos;
                    }
                    top_vals[pos][threadIdx.x] = value;
                    top_ids[pos][threadIdx.x]  = row;
                    kth_val                    = top_vals[local_count - 1][threadIdx.x];
                }
            }
        }
    }
    // common::block_reduce_sum leaves the total in warp 0 only; every thread resolves ids below.
    __shared__ float warp_sums[BlockSize / common::kWarpSize];
    __shared__ float totals[2];
    raw_sum = common::block_reduce_sum<BlockSize>(raw_sum, warp_sums);
    if (threadIdx.x == 0) { totals[0] = raw_sum; }
    __syncthreads();
    masked_sum = common::block_reduce_sum<BlockSize>(masked_sum, warp_sums);
    if (threadIdx.x == 0) { totals[1] = masked_sum; }
    __syncthreads();

    const float raw_normaliser    = raw_max + logf(totals[0]);
    const float masked_normaliser = masked_max + logf(totals[1]);
    const auto resolve            = [&](std::int32_t id, float& raw, float& masked) {
        const float value = __bfloat162float(logits[base + id]);
        raw               = value - raw_normaliser;
        masked =
            candidate_logprobs_allowed(allowed, id) ? value - masked_normaliser : -CUDART_INF_F;
    };
    if (threadIdx.x == 0) {
        resolve(sampled_ids[column], sampled_out[column], sampled_out[columns + column]);
    }
    for (std::int32_t n = static_cast<std::int32_t>(threadIdx.x); n < candidates; n += BlockSize) {
        const std::int64_t slot = static_cast<std::int64_t>(n) * columns + column;
        resolve(candidate_ids[n], candidates_out[slot],
                candidates_out[static_cast<std::int64_t>(candidates) * columns + slot]);
    }
    if constexpr (HasTopK) {
        __shared__ float warp_best_val[BlockSize / common::kWarpSize];
        __shared__ std::int32_t warp_best_id[BlockSize / common::kWarpSize];
        __shared__ std::int32_t chosen_id;
        std::int32_t head       = 0;
        const int lane          = static_cast<int>(threadIdx.x) & (common::kWarpSize - 1);
        const int warp          = static_cast<int>(threadIdx.x) / common::kWarpSize;
        constexpr int kNumWarps = BlockSize / common::kWarpSize;
        for (std::int32_t k = 0; k < top_k; ++k) {
            float best_val       = head < local_count ? top_vals[head][threadIdx.x] : -CUDART_INF_F;
            std::int32_t best_id = head < local_count ? top_ids[head][threadIdx.x] : -1;
            for (int offset = common::kWarpSize / 2; offset > 0; offset /= 2) {
                const float other_val       = __shfl_down_sync(common::kFullWarpMask, best_val, offset);
                const std::int32_t other_id = __shfl_down_sync(common::kFullWarpMask, best_id, offset);
                if (candidate_logprobs_better(other_val, other_id, best_val, best_id)) {
                    best_val = other_val;
                    best_id  = other_id;
                }
            }
            if (lane == 0) {
                warp_best_val[warp] = best_val;
                warp_best_id[warp]  = best_id;
            }
            __syncthreads();
            if (warp == 0) {
                best_val = lane < kNumWarps ? warp_best_val[lane] : -CUDART_INF_F;
                best_id  = lane < kNumWarps ? warp_best_id[lane] : -1;
                for (int offset = kNumWarps / 2; offset > 0; offset /= 2) {
                    const float other_val       = __shfl_down_sync(common::kFullWarpMask, best_val, offset);
                    const std::int32_t other_id = __shfl_down_sync(common::kFullWarpMask, best_id, offset);
                    if (candidate_logprobs_better(other_val, other_id, best_val, best_id)) {
                        best_val = other_val;
                        best_id  = other_id;
                    }
                }
                if (lane == 0) {
                    chosen_id               = best_id;
                    const std::int64_t slot = static_cast<std::int64_t>(k) * columns + column;
                    top_ids_out[slot]       = best_id;
                    top_out[slot] = best_id >= 0 ? best_val - raw_normaliser : -CUDART_INF_F;
                    top_out[static_cast<std::int64_t>(top_k) * columns + slot] =
                        best_id >= 0 ? best_val - masked_normaliser : -CUDART_INF_F;
                }
            }
            __syncthreads();
            const std::int32_t winner = chosen_id;
            if (winner < 0) {
                if (threadIdx.x == 0) {
                    for (std::int32_t rem = k + 1; rem < top_k; ++rem) {
                        const std::int64_t slot = static_cast<std::int64_t>(rem) * columns + column;
                        top_ids_out[slot]       = -1;
                        top_out[slot]           = -CUDART_INF_F;
                        top_out[static_cast<std::int64_t>(top_k) * columns + slot] = -CUDART_INF_F;
                    }
                }
                break;
            }
            if (head < local_count && top_ids[head][threadIdx.x] == winner) { ++head; }
        }
    }
}

} // namespace ninfer::ops
