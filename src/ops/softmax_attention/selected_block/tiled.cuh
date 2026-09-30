#pragma once

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/softmax_attention/selected_block/common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Tensor-core route. One CTA owns one (query column, KV head): the twelve query heads that share
// the KV head (padded to an m16 tile) stage each 64-token K tile once. Q.K^T runs as
// m16n8k16 on all four warps (BF16 operands for the BF16 cache; FP16 rotated queries and decoded
// FP8 keys for the FP8 cache); probabilities are staged as FP16 and P.V runs as FP16
// m16n8k16 against the FP16 value tile. The value tile reuses the key tile's shared storage after
// the score barrier, and its asynchronous copy overlaps the softmax. Online-softmax statistics and
// both accumulators stay FP32.
//
// Shared-row mode (Partial = false) walks a column's whole visible set through one table row and
// stores the normalized BF16 output. Partial mode (the batched route) splits each column's visible
// set into gridDim.y contiguous partitions, reads the column's own table row, and stores each
// partition's unnormalized numerators with its running maximum and denominator for the merge.
inline constexpr int kSelectedPrefillWarps   = 4;
inline constexpr int kSelectedPrefillThreads = 32 * kSelectedPrefillWarps;
inline constexpr int kSelectedPrefillRows    = 16; // 12 heads, 4 padded
inline constexpr int kSelectedPrefillTile    = 64;

template <SelectedKvProfile Profile>
__device__ __forceinline__ void selected_stage_key(std::uint16_t* destination,
                                                   const SelectedKvPlanes& kv, std::int64_t row,
                                                   int k8, bool valid) {
    if constexpr (Profile == SelectedKvProfile::Bf16KFp16V) {
        const auto* source = static_cast<const __nv_bfloat16*>(kv.k) +
                             (valid ? row * kSelectedHeadDim + k8 * 8 : 0);
        cp_async_zfill<16, Cache::cg>(destination, source, valid ? 16 : 0);
    } else {
        uint4 packed = make_uint4(0, 0, 0, 0);
        if (valid) {
            const uint2 raw = *reinterpret_cast<const uint2*>(
                static_cast<const std::uint8_t*>(kv.k) + row * kSelectedHeadDim + k8 * 8);
            const float scale = __half2float(__ushort_as_half(kv.k_scale[row]));
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(&raw);
            auto* out         = reinterpret_cast<__half2*>(&packed);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                out[i] = __floats2half2_rn(fp8_value(bytes[2 * i]) * scale,
                                           fp8_value(bytes[2 * i + 1]) * scale);
            }
        }
        *reinterpret_cast<uint4*>(destination) = packed;
    }
}

template <SelectedKvProfile Profile>
__device__ __forceinline__ void selected_stage_value(std::uint16_t* destination,
                                                     const SelectedKvPlanes& kv, std::int64_t row,
                                                     int k8, bool valid) {
    if constexpr (Profile == SelectedKvProfile::Bf16KFp16V) {
        const auto* source =
            static_cast<const __half*>(kv.v) + (valid ? row * kSelectedHeadDim + k8 * 8 : 0);
        cp_async_zfill<16, Cache::cg>(destination, source, valid ? 16 : 0);
    } else {
        uint4 packed = make_uint4(0, 0, 0, 0);
        if (valid) {
            const uint2 raw = *reinterpret_cast<const uint2*>(
                static_cast<const std::uint8_t*>(kv.v) + row * kSelectedHeadDim + k8 * 8);
            const float scale = __half2float(__ushort_as_half(kv.v_scale[row]));
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(&raw);
            auto* out         = reinterpret_cast<__half2*>(&packed);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                out[i] = __floats2half2_rn(fp8_value(bytes[2 * i]) * scale,
                                           fp8_value(bytes[2 * i + 1]) * scale);
            }
        }
        *reinterpret_cast<uint4*>(destination) = packed;
    }
}

template <SelectedKvProfile Profile, bool Partial>
__global__ __launch_bounds__(kSelectedPrefillThreads) void selected_block_tile_kernel(
    const __nv_bfloat16* __restrict__ query, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ table_rows, const std::int32_t* __restrict__ selections,
    const std::int32_t* __restrict__ counts, SelectedKvPlanes kv, __nv_bfloat16* __restrict__ output,
    float* __restrict__ partial) {
    using detail::gemm_swz64;
    constexpr int kDim   = kSelectedHeadDim;
    constexpr int kTile  = kSelectedPrefillTile;
    constexpr int kRows  = kSelectedPrefillRows;
    constexpr int kHeads = kSelectedHeadsPerKv;

    __shared__ __align__(16) std::uint16_t query_tile[kRows * kDim];
    __shared__ __align__(16) std::uint16_t kv_tile[kTile * kDim]; // K (BF16), then V (FP16)
    __shared__ float scores[kRows * kTile];
    __shared__ __align__(16) std::uint16_t probabilities[kRows * kTile]; // FP16
    __shared__ float prior_scale[kRows];
    __shared__ float inverse_sum[kRows];

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int kv_head = static_cast<int>(blockIdx.x) & 1;
    const std::int64_t column = static_cast<std::int64_t>(blockIdx.x) >> 1;
    const int first_head      = kv_head * kHeads;

    const int position = positions[column];
    const int complete = (position + 1) / 4;
    const int count    = counts[column];
    const int total    = count * 4 + ((position + 1) & 3);
    const auto* selected = selections + column * kSelectedMaxBlocks;
    const bool dense     = count == complete; // the selection is exactly 0..complete-1
    const auto* table    = Partial ? kv.tables + static_cast<std::int64_t>(table_rows[column]) *
                                                  kv.logical_pages
                                   : kv.tables; // the shared row
    const int split      = static_cast<int>(blockIdx.y);
    const int splits     = static_cast<int>(gridDim.y);
    const int begin      = static_cast<int>(static_cast<std::int64_t>(total) * split / splits);
    const int end        = static_cast<int>(static_cast<std::int64_t>(total) * (split + 1) / splits);

    // Query rows in the cache basis; padded rows and padded probabilities stay zero.
    const auto* q_base = query + (column * kSelectedQueryHeads + first_head) * kDim;
    if constexpr (Profile == SelectedKvProfile::Bf16KFp16V) {
        for (int item = tid; item < kRows * (kDim / 8); item += kSelectedPrefillThreads) {
            const int row    = item / (kDim / 8);
            const int k8     = item % (kDim / 8);
            const bool valid = row < kHeads;
            cp_async_zfill<16, Cache::cg>(&query_tile[row * kDim + gemm_swz64(row, k8 * 8)],
                                          q_base + (valid ? row * kDim + k8 * 8 : 0),
                                          valid ? 16 : 0);
        }
        cp_commit();
    } else {
        for (int row = warp; row < kRows; row += kSelectedPrefillWarps) {
            float values[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                values[r] = row < kHeads ? __bfloat162float(q_base[row * kDim + lane + 32 * r]) : 0.0F;
            }
            normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const __half value = __float2half_rn(values[r]);
                query_tile[row * kDim + gemm_swz64(row, lane + 32 * r)] =
                    *reinterpret_cast<const std::uint16_t*>(&value);
            }
        }
    }
    for (int item = tid; item < kRows * kTile; item += kSelectedPrefillThreads) {
        probabilities[item] = 0;
    }

    float running_max[3] = {-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F};
    float running_sum[3] = {0.0F, 0.0F, 0.0F};
    float accumulator[8][4] = {};

    const int a_row   = (lane & 7) + (((lane >> 3) & 1) << 3);
    const int a_col   = (lane >> 4) << 3;
    const int b_row   = lane & 7;
    const int b_col   = ((lane >> 3) & 1) << 3;
    const int group   = lane >> 2;
    const int quad    = lane & 3;

    for (int start = begin; start < end; start += kTile) {
        const int tile_count = min(kTile, end - start);
        // K tile.
        for (int item = tid; item < kTile * (kDim / 8); item += kSelectedPrefillThreads) {
            const int row    = item / (kDim / 8);
            const int k8     = item % (kDim / 8);
            const bool valid = row < tile_count;
            std::int64_t kv_row = 0;
            if (valid) {
                const int token = dense ? start + row
                                        : selected_token(start + row, count, complete, selected);
                kv_row = selected_kv_row(table, token, kv_head);
            }
            selected_stage_key<Profile>(&kv_tile[row * kDim + gemm_swz64(row, k8 * 8)], kv, kv_row,
                                        k8, valid);
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();

        // S = Q.K^T: warp w owns keys [16w,16w+16).
        float tile_score[2][4] = {};
#pragma unroll
        for (int kstep = 0; kstep < kDim / 16; ++kstep) {
            unsigned a[4];
            ldmatrix_x4(a[0], a[1], a[2], a[3],
                        smem_addr(&query_tile[a_row * kDim + gemm_swz64(a_row, kstep * 16 + a_col)]));
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                const int key_row = (warp * 2 + n) * 8 + b_row;
                unsigned b[2];
                ldmatrix_x2(b[0], b[1],
                            smem_addr(&kv_tile[key_row * kDim +
                                               gemm_swz64(key_row, kstep * 16 + b_col)]));
                if constexpr (Profile == SelectedKvProfile::Bf16KFp16V) {
                    mma_bf16(tile_score[n][0], tile_score[n][1], tile_score[n][2],
                             tile_score[n][3], a[0], a[1], a[2], a[3], b[0], b[1]);
                } else {
                    mma_f16(tile_score[n][0], tile_score[n][1], tile_score[n][2],
                            tile_score[n][3], a[0], a[1], a[2], a[3], b[0], b[1]);
                }
            }
        }
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            const int c0 = (warp * 2 + n) * 8 + 2 * quad;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = group + ((e >> 1) << 3);
                const int col = c0 + (e & 1);
                scores[row * kTile + col] = (row < kHeads && col < tile_count)
                                                ? tile_score[n][e] * kSelectedScale
                                                : -CUDART_INF_F;
            }
        }
        __syncthreads(); // every warp has consumed the K tile and published its scores

        // V tile into the same storage; the copy overlaps the softmax below.
        for (int item = tid; item < kTile * (kDim / 8); item += kSelectedPrefillThreads) {
            const int row    = item / (kDim / 8);
            const int k8     = item % (kDim / 8);
            const bool valid = row < tile_count;
            std::int64_t kv_row = 0;
            if (valid) {
                const int token = dense ? start + row
                                        : selected_token(start + row, count, complete, selected);
                kv_row = selected_kv_row(table, token, kv_head);
            }
            selected_stage_value<Profile>(&kv_tile[row * kDim + gemm_swz64(row, k8 * 8)], kv,
                                          kv_row, k8, valid);
        }
        cp_commit();

        // Online softmax: warp w owns heads w, w+4, w+8.
#pragma unroll
        for (int slot = 0; slot < 3; ++slot) {
            const int head = warp + slot * kSelectedPrefillWarps;
            const float s0 = scores[head * kTile + lane];
            const float s1 = scores[head * kTile + lane + 32];
            const float tile_max = warp_max(fmaxf(s0, s1));
            const float p0       = expf(s0 - tile_max);
            const float p1       = expf(s1 - tile_max);
            const float tile_sum = warp_sum(p0 + p1);
            const float next_max = fmaxf(running_max[slot], tile_max);
            const float prior    = running_sum[slot] == 0.0F ? 0.0F : expf(running_max[slot] - next_max);
            const float chunk    = expf(tile_max - next_max);
            running_sum[slot]    = running_sum[slot] * prior + tile_sum * chunk;
            running_max[slot]    = next_max;
            const __half h0      = __float2half_rn(p0 * chunk);
            const __half h1      = __float2half_rn(p1 * chunk);
            probabilities[head * kTile + gemm_swz64(head, lane)] =
                *reinterpret_cast<const std::uint16_t*>(&h0);
            probabilities[head * kTile + gemm_swz64(head, lane + 32)] =
                *reinterpret_cast<const std::uint16_t*>(&h1);
            if (lane == 0) { prior_scale[head] = prior; }
        }
        cp_wait<0>();
        __syncthreads(); // V tile, probabilities and prior scales are complete

        // O = O * prior + P.V: warp w owns dimensions [64w,64w+64).
        const float prior0 = group < kHeads ? prior_scale[group] : 0.0F;
        const float prior1 = group + 8 < kHeads ? prior_scale[group + 8] : 0.0F;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            accumulator[n][0] *= prior0;
            accumulator[n][1] *= prior0;
            accumulator[n][2] *= prior1;
            accumulator[n][3] *= prior1;
        }
#pragma unroll
        for (int kstep = 0; kstep < kTile / 16; ++kstep) {
            unsigned a[4];
            ldmatrix_x4(a[0], a[1], a[2], a[3],
                        smem_addr(&probabilities[a_row * kTile +
                                                 gemm_swz64(a_row, kstep * 16 + a_col)]));
            const int key_row = kstep * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                const int dim0 = (warp * 8 + n) * 8;
                unsigned b[2];
                ldmatrix_x2_t(b[0], b[1],
                              smem_addr(&kv_tile[key_row * kDim + gemm_swz64(key_row, dim0)]));
                mma_f16(accumulator[n][0], accumulator[n][1], accumulator[n][2],
                        accumulator[n][3], a[0], a[1], a[2], a[3], b[0], b[1]);
            }
        }
        __syncthreads(); // the next tile overwrites K/V, scores and probabilities
    }

    if constexpr (Partial) {
        float* base = partial + ((column * kSelectedQueryHeads + first_head) * splits + split) *
                                    kSelectedPartialStride;
        const auto row_of = [&](int head) { return base + static_cast<std::int64_t>(head) * splits *
                                                              kSelectedPartialStride; };
#pragma unroll
        for (int slot = 0; slot < 3; ++slot) {
            const int head = warp + slot * kSelectedPrefillWarps;
            if (lane == 0) {
                row_of(head)[kDim]     = running_max[slot];
                row_of(head)[kDim + 1] = running_sum[slot];
            }
        }
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            const int d0 = (warp * 8 + n) * 8 + 2 * quad;
            if (group < kHeads) {
                *reinterpret_cast<float2*>(&row_of(group)[d0]) =
                    make_float2(accumulator[n][0], accumulator[n][1]);
            }
            if (group + 8 < kHeads) {
                *reinterpret_cast<float2*>(&row_of(group + 8)[d0]) =
                    make_float2(accumulator[n][2], accumulator[n][3]);
            }
        }
        return;
    } else {
#pragma unroll
        for (int slot = 0; slot < 3; ++slot) {
            const int head = warp + slot * kSelectedPrefillWarps;
            if (lane == 0) {
                inverse_sum[head] = running_sum[slot] > 0.0F ? 1.0F / running_sum[slot] : 0.0F;
            }
        }
        __syncthreads();
        auto* out_base = output + (column * kSelectedQueryHeads + first_head) * kDim;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            const int d0 = (warp * 8 + n) * 8 + 2 * quad;
            if (group < kHeads) {
                const float inv = inverse_sum[group];
                *reinterpret_cast<__nv_bfloat162*>(&out_base[group * kDim + d0]) =
                    __floats2bfloat162_rn(accumulator[n][0] * inv, accumulator[n][1] * inv);
            }
            if (group + 8 < kHeads) {
                const float inv = inverse_sum[group + 8];
                *reinterpret_cast<__nv_bfloat162*>(&out_base[(group + 8) * kDim + d0]) =
                    __floats2bfloat162_rn(accumulator[n][2] * inv, accumulator[n][3] * inv);
            }
        }
    }
}

} // namespace ninfer::ops::detail
