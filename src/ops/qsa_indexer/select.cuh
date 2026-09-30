#pragma once

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/qsa_indexer/common.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Query preparation: CTA c normalizes and rotates the four query heads of column c into BF16
// q_prepared [128,4,C].
__global__ __launch_bounds__(kQsaHeadDim) void qsa_prepare_query_kernel(
    const __nv_bfloat16* __restrict__ projected, const std::int32_t* __restrict__ rope_positions,
    const __nv_bfloat16* __restrict__ query_norm, __nv_bfloat16* __restrict__ prepared,
    int columns) {
    __shared__ float2 rotation[kTextMropeR64Pairs];
    __shared__ float scratch[kQsaHeadDim + 4];
    const int dim             = static_cast<int>(threadIdx.x);
    const std::int64_t column = blockIdx.x;
    if (dim < kTextMropeR64Pairs) {
        rotation[dim] = text_mrope_r64_sincos(
            dim, text_mrope_axis_position(rope_positions, columns, column, dim));
    }
    __syncthreads();
    const float weight = __bfloat162float(query_norm[dim]);
    for (int head = 0; head < kQsaQueryHeads; ++head) {
        const float x = __bfloat162float(projected[column * kQsaProjection + head * kQsaHeadDim + dim]);
        const float out = qsa_norm_rotate(x, weight, rotation, scratch);
        prepared[(column * kQsaQueryHeads + head) * kQsaHeadDim + dim] = __float2bfloat16_rn(out);
    }
}

__device__ __forceinline__ float qsa_canonical_score(float value) {
    return value + 0.0F; // -0 -> +0, so equal scores have equal bits
}

// Warp-SIMT scoring for independent table rows. Eight lanes own one block (16 dimensions each);
// a 256-thread CTA scores 32 consecutive blocks of one column. Only blocks below the column's
// complete count are read or written.
inline constexpr int kQsaScoreThreads         = 256;
inline constexpr int kQsaScoreBlocksPerCta    = kQsaScoreThreads / 8;

__global__ __launch_bounds__(kQsaScoreThreads) void qsa_score_rows_kernel(
    const __nv_bfloat16* __restrict__ prepared, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ table_rows, const std::int32_t* __restrict__ tables,
    int logical_pages, const __nv_bfloat16* __restrict__ block_keys, float* __restrict__ scores,
    int score_stride) {
    __shared__ float query[kQsaQueryHeads * kQsaHeadDim];
    const int column   = static_cast<int>(blockIdx.y);
    const int complete = (positions[column] + 1) / 4;
    const int first    = static_cast<int>(blockIdx.x) * kQsaScoreBlocksPerCta;
    if (first >= complete) { return; } // uniform per CTA
    for (int i = static_cast<int>(threadIdx.x); i < kQsaQueryHeads * kQsaHeadDim; i += kQsaScoreThreads) {
        query[i] = __bfloat162float(prepared[static_cast<std::int64_t>(column) * kQsaQueryHeads * kQsaHeadDim + i]);
    }
    __syncthreads();
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int group = static_cast<int>(threadIdx.x) >> 3; // block within the CTA
    const int part  = lane & 7;                            // 16-dimension slice
    const int block = first + group;
    const auto* table = tables + static_cast<std::int64_t>(table_rows[column]) * logical_pages;
    float dots[kQsaQueryHeads] = {};
    if (block < complete) {
        const auto* key = block_keys + qsa_block_offset(table, block) + part * 16;
        const uint4 raw0 = *reinterpret_cast<const uint4*>(key);
        const uint4 raw1 = *reinterpret_cast<const uint4*>(key + 8);
        const auto* k0   = reinterpret_cast<const __nv_bfloat162*>(&raw0);
        const auto* k1   = reinterpret_cast<const __nv_bfloat162*>(&raw1);
        float kv[16];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 a = __bfloat1622float2(k0[i]);
            const float2 b = __bfloat1622float2(k1[i]);
            kv[2 * i]         = a.x;
            kv[2 * i + 1]     = a.y;
            kv[8 + 2 * i]     = b.x;
            kv[8 + 2 * i + 1] = b.y;
        }
#pragma unroll
        for (int h = 0; h < kQsaQueryHeads; ++h) {
            const float* q = query + h * kQsaHeadDim + part * 16;
#pragma unroll
            for (int i = 0; i < 16; ++i) { dots[h] = fmaf(q[i], kv[i], dots[h]); }
        }
    }
#pragma unroll
    for (int h = 0; h < kQsaQueryHeads; ++h) { dots[h] = warp_sum<8>(dots[h]); }
    if (block < complete && part == 0) {
        float score = 0.0F;
#pragma unroll
        for (int h = 0; h < kQsaQueryHeads; ++h) { score += fmaxf(dots[h], 0.0F); }
        scores[static_cast<std::int64_t>(column) * score_stride + block] =
            qsa_canonical_score(score * kQsaScoreScale);
    }
}

// Tensor-core scoring for columns sharing one table row. A CTA owns 64 blocks x 64 columns,
// stages its block keys once and reuses them for the four query heads.
inline constexpr int kQsaMmaBlocks  = 64;
inline constexpr int kQsaMmaColumns = 64;
inline constexpr int kQsaMmaThreads = 128;

__global__ __launch_bounds__(kQsaMmaThreads) void qsa_score_shared_row_kernel(
    const __nv_bfloat16* __restrict__ prepared, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ table, const __nv_bfloat16* __restrict__ block_keys,
    float* __restrict__ scores, int score_stride, int columns) {
    using detail::gemm_swz64;
    __shared__ __align__(16) __nv_bfloat16 keys[kQsaMmaBlocks * kQsaHeadDim];
    __shared__ __align__(16) __nv_bfloat16 query[kQsaMmaColumns * kQsaHeadDim];
    __shared__ int complete_of[kQsaMmaColumns];
    __shared__ int tile_complete;
    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int block0 = static_cast<int>(blockIdx.x) * kQsaMmaBlocks;
    const int col0   = static_cast<int>(blockIdx.y) * kQsaMmaColumns;
    if (tid == 0) { tile_complete = 0; }
    __syncthreads();
    if (tid < kQsaMmaColumns) {
        const int c       = col0 + tid;
        complete_of[tid]  = c < columns ? (positions[c] + 1) / 4 : 0;
        atomicMax(&tile_complete, complete_of[tid]);
    }
    __syncthreads();
    const int limit = tile_complete;
    if (block0 >= limit) { return; } // uniform per CTA
    for (int item = tid; item < kQsaMmaBlocks * (kQsaHeadDim / 8); item += kQsaMmaThreads) {
        const int row    = item / (kQsaHeadDim / 8);
        const int k8     = item % (kQsaHeadDim / 8);
        const int block  = block0 + row;
        const bool valid = block < limit;
        const __nv_bfloat16* source =
            valid ? block_keys + qsa_block_offset(table, block) + k8 * 8 : block_keys;
        cp_async_zfill<16, Cache::cg>(&keys[row * kQsaHeadDim + gemm_swz64(row, k8 * 8)], source,
                                      valid ? 16 : 0);
    }
    cp_commit();

    float total[8][4] = {};
    const int a_row = warp * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
    const int a_col = (lane >> 4) << 3;
    const int b_row = lane & 7;
    const int b_col = ((lane >> 3) & 1) << 3;
    for (int head = 0; head < kQsaQueryHeads; ++head) {
        for (int item = tid; item < kQsaMmaColumns * (kQsaHeadDim / 8); item += kQsaMmaThreads) {
            const int col    = item / (kQsaHeadDim / 8);
            const int k8     = item % (kQsaHeadDim / 8);
            const int c      = col0 + col;
            const bool valid = c < columns;
            const __nv_bfloat16* source =
                prepared + (valid ? (static_cast<std::int64_t>(c) * kQsaQueryHeads + head) * kQsaHeadDim + k8 * 8 : 0);
            cp_async_zfill<16, Cache::cg>(&query[col * kQsaHeadDim + gemm_swz64(col, k8 * 8)],
                                          source, valid ? 16 : 0);
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();
        float dot[8][4] = {};
#pragma unroll
        for (int kstep = 0; kstep < kQsaHeadDim / 16; ++kstep) {
            unsigned a[4];
            ldmatrix_x4(a[0], a[1], a[2], a[3],
                        smem_addr(&keys[a_row * kQsaHeadDim + gemm_swz64(a_row, kstep * 16 + a_col)]));
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int col = n * 8 + b_row;
                unsigned b[2];
                ldmatrix_x2(b[0], b[1],
                            smem_addr(&query[col * kQsaHeadDim + gemm_swz64(col, kstep * 16 + b_col)]));
                mma_bf16(dot[n][0], dot[n][1], dot[n][2], dot[n][3], a[0], a[1], a[2], a[3], b[0],
                         b[1]);
            }
        }
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) total[n][e] += fmaxf(dot[n][e], 0.0F);
        __syncthreads(); // the next head overwrites the query tile
    }
    const int group = lane >> 2;
    const int quad  = lane & 3;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int block = block0 + warp * 16 + group + ((e >> 1) << 3);
            const int col   = n * 8 + 2 * quad + (e & 1);
            const int c     = col0 + col;
            if (c < columns && block < complete_of[col]) {
                scores[static_cast<std::int64_t>(c) * score_stride + block] =
                    qsa_canonical_score(total[n][e] * kQsaScoreScale);
            }
        }
    }
}

// Exact top-512 selection per column. An MSB-first radix select over four 8-bit digits
// finds the 512th largest order key; one id-ordered pass then emits every larger key and the
// lowest-id ties, so the output is ascending. Each thread owns four consecutive scores per round
// (the score row stride is a multiple of four).
inline constexpr int kQsaSelectThreads = 512;
inline constexpr int kQsaSelectWarps   = kQsaSelectThreads / 32;
inline constexpr int kQsaSelectBins    = 256;
inline constexpr int kQsaSelectRound   = kQsaSelectThreads * 4;

__device__ __forceinline__ unsigned qsa_order_key(float score) {
    const unsigned bits = __float_as_uint(score);
    return (bits & 0x80000000U) != 0U ? ~bits : (bits | 0x80000000U);
}

// Loads the order keys of ids [base, base+4); ids at or beyond `complete` are invalid.
__device__ __forceinline__ void qsa_load4(const float* row, int base, int complete,
                                          unsigned (&key)[4], bool (&valid)[4]) {
    if (base + 3 < complete) {
        const float4 v = *reinterpret_cast<const float4*>(row + base);
        key[0]         = qsa_order_key(v.x);
        key[1]         = qsa_order_key(v.y);
        key[2]         = qsa_order_key(v.z);
        key[3]         = qsa_order_key(v.w);
#pragma unroll
        for (int i = 0; i < 4; ++i) { valid[i] = true; }
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            valid[i] = base + i < complete;
            key[i]   = valid[i] ? qsa_order_key(row[base + i]) : 0U;
        }
    }
}

// Block-wide exclusive prefix sum of `value`; stores the block total. One barrier publishes the
// warp totals; the trailing one keeps the next call from overwriting them early.
__device__ __forceinline__ int qsa_block_exclusive_scan(int value, int* warp_totals, int* total) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    int inclusive  = value;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(kFullWarpMask, inclusive, offset);
        if (lane >= offset) { inclusive += other; }
    }
    if (lane == 31) { warp_totals[warp] = inclusive; }
    __syncthreads();
    int before = 0;
    int sum    = 0;
#pragma unroll
    for (int w = 0; w < kQsaSelectWarps; ++w) {
        const int v = warp_totals[w];
        before += w < warp ? v : 0;
        sum += v;
    }
    *total = sum;
    __syncthreads();
    return before + inclusive - value;
}

__global__ __launch_bounds__(kQsaSelectThreads) void qsa_select_kernel(
    const float* __restrict__ scores, int score_stride, const std::int32_t* __restrict__ positions,
    std::int32_t* __restrict__ selections, std::int32_t* __restrict__ counts, bool score) {
    __shared__ unsigned histogram[kQsaSelectBins];
    __shared__ unsigned threshold_state[2]; // prefix, remaining k
    __shared__ int warp_totals[kQsaSelectWarps];
    const int tid             = static_cast<int>(threadIdx.x);
    const std::int64_t column = blockIdx.x;
    const int complete        = (positions[column] + 1) / 4;
    auto* out                 = selections + column * kQsaMaxSelected;
    if (!score || complete <= kQsaMaxSelected) {
        const int count = complete < kQsaMaxSelected ? complete : kQsaMaxSelected;
        if (tid == 0) { counts[column] = count; }
        for (int i = tid; i < kQsaMaxSelected; i += kQsaSelectThreads) {
            out[i] = i < count ? i : -1;
        }
        return;
    }
    const float* row = scores + column * score_stride;
    unsigned prefix  = 0;
    unsigned mask    = 0;
    unsigned remain  = kQsaMaxSelected;
#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        const int shift           = 24 - 8 * pass;
        const unsigned digit_mask = 0xffU;
        for (int i = tid; i < kQsaSelectBins; i += kQsaSelectThreads) { histogram[i] = 0; }
        __syncthreads();
        for (int base = tid * 4; base < complete; base += kQsaSelectRound) {
            unsigned key[4];
            bool valid[4];
            qsa_load4(row, base, complete, key, valid);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (valid[i] && (key[i] & mask) == prefix) {
                    atomicAdd(&histogram[(key[i] >> shift) & digit_mask], 1U);
                }
            }
        }
        __syncthreads();
        if (tid < 32) {
            // Lane l owns 8 consecutive digits; `above` counts candidates with larger digits.
            constexpr int kPerLane = kQsaSelectBins / 32;
            unsigned lane_sum      = 0;
            for (int j = 0; j < kPerLane; ++j) { lane_sum += histogram[tid * kPerLane + j]; }
            unsigned above = 0;
            for (int other = 1; other < 32; ++other) {
                const unsigned value = __shfl_down_sync(kFullWarpMask, lane_sum, other);
                if (tid + other < 32) { above += value; }
            }
            if (above < remain && above + lane_sum >= remain) {
                for (int j = kPerLane - 1; j >= 0; --j) {
                    const unsigned count = histogram[tid * kPerLane + j];
                    if (above + count >= remain) {
                        threshold_state[0] =
                            prefix | (static_cast<unsigned>(tid * kPerLane + j) << shift);
                        threshold_state[1] = remain - above;
                        break;
                    }
                    above += count;
                }
            }
        }
        __syncthreads();
        prefix = threshold_state[0];
        remain = threshold_state[1];
        mask |= digit_mask << shift;
        __syncthreads(); // histogram and threshold_state are rewritten by the next pass
    }
    // prefix is the 512th largest key; `remain` keys equal to it are taken, lowest ids first.
    int emitted = 0;
    int ties    = 0;
    for (int round = 0; round < complete; round += kQsaSelectRound) {
        const int base = round + tid * 4;
        unsigned key[4];
        bool valid[4];
        qsa_load4(row, base, complete, key, valid);
        int equal = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) { equal += (valid[i] && key[i] == prefix) ? 1 : 0; }
        int equal_total = 0;
        int tie_rank    = ties + qsa_block_exclusive_scan(equal, warp_totals, &equal_total);
        bool take[4];
        int taken = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const bool is_equal = valid[i] && key[i] == prefix;
            take[i] = valid[i] && (key[i] > prefix ||
                                   (is_equal && tie_rank < static_cast<int>(remain)));
            tie_rank += is_equal ? 1 : 0;
            taken += take[i] ? 1 : 0;
        }
        int take_total = 0;
        int slot       = emitted + qsa_block_exclusive_scan(taken, warp_totals, &take_total);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (take[i]) { out[slot++] = base + i; }
        }
        ties += equal_total;
        emitted += take_total;
    }
    if (tid == 0) { counts[column] = kQsaMaxSelected; }
}

} // namespace ninfer::ops::detail
