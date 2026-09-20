#pragma once

// Implements: include/ninfer/ops/candidate_logprobs.h
// Match: contiguous BF16 [physical_rows,C], I32 ids, optional I32 [mask_words] mask, FP32 outputs.
// Algorithm assumptions: one 256-thread CTA per column performs a stable two-pass logsumexp for
// the raw and the masked normaliser together, then each thread resolves a strided share of ids.

#include "ops/common/warp.cuh"
#include "ops/kernel/target_logprobs.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCandidateLogprobsBlock = 256;

__device__ __forceinline__ bool candidate_logprobs_allowed(const std::int32_t* allowed,
                                                           std::int32_t row) {
    if (allowed == nullptr) { return true; }
    const auto word = static_cast<std::uint32_t>(allowed[row >> 5]);
    return ((word >> (row & 31)) & 1U) != 0U;
}

template <int BlockSize>
__launch_bounds__(BlockSize) __global__
    void candidate_logprobs_kernel(const __nv_bfloat16* logits, std::int32_t valid_rows,
                                   std::int32_t physical_rows, const std::int32_t* sampled_ids,
                                   const std::int32_t* candidate_ids, std::int32_t candidates,
                                   const std::int32_t* allowed, float* sampled_out,
                                   float* candidates_out, std::int32_t columns) {
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base   = static_cast<std::int64_t>(column) * physical_rows;

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

    float raw_sum    = 0.0f;
    float masked_sum = 0.0f;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        const float value = __bfloat162float(logits[base + row]);
        raw_sum += expf(value - raw_max);
        if (candidate_logprobs_allowed(allowed, row)) { masked_sum += expf(value - masked_max); }
    }
    // block_reduce_sum leaves the total in warp 0 only; every thread resolves ids below.
    __shared__ float warp_sums[BlockSize / kWarpSize];
    __shared__ float totals[2];
    raw_sum = block_reduce_sum<BlockSize>(raw_sum, warp_sums);
    if (threadIdx.x == 0) { totals[0] = raw_sum; }
    __syncthreads();
    masked_sum = block_reduce_sum<BlockSize>(masked_sum, warp_sums);
    if (threadIdx.x == 0) { totals[1] = masked_sum; }
    __syncthreads();

    const float raw_normaliser    = raw_max + logf(totals[0]);
    const float masked_normaliser = masked_max + logf(totals[1]);
    const auto resolve = [&](std::int32_t id, float& raw, float& masked) {
        const float value = __bfloat162float(logits[base + id]);
        raw               = value - raw_normaliser;
        masked = candidate_logprobs_allowed(allowed, id) ? value - masked_normaliser : -CUDART_INF_F;
    };
    if (threadIdx.x == 0) {
        resolve(sampled_ids[column], sampled_out[column], sampled_out[columns + column]);
    }
    for (std::int32_t n = static_cast<std::int32_t>(threadIdx.x); n < candidates; n += BlockSize) {
        const std::int64_t slot = static_cast<std::int64_t>(n) * columns + column;
        resolve(candidate_ids[n], candidates_out[slot],
                candidates_out[static_cast<std::int64_t>(candidates) * columns + slot]);
    }
}

} // namespace ninfer::ops
