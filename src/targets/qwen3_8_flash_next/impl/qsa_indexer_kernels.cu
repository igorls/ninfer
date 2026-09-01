#include "targets/qwen3_8_flash_next/impl/qsa_indexer_kernels.h"

#include "core/device.h"
#include "ops/common/warp.cuh"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/device_topk.cuh>
#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

constexpr int kHeadDim          = 128;
constexpr int kQueryHeads       = 4;
constexpr int kProjectionRows   = 640;
constexpr int kRawKeyOffset     = 512;
constexpr int kCompressedPage   = 64;
constexpr int kSelectedBlocks   = 512;
constexpr float kRopeTheta      = 1.0e7F;
constexpr float kIndexerScaling = 0.08838834764831845F; // 1/sqrt(128)

__device__ float rope_frequency(int pair) {
    return expf((-2.0F * static_cast<float>(pair) / 64.0F) * logf(kRopeTheta));
}

__device__ void store_rotated(const __nv_bfloat16* normalized, const std::int32_t* positions,
                              __nv_bfloat16* output, int dim) {
    if (dim < 32) {
        const float angle = static_cast<float>(positions[dim % 3]) * rope_frequency(dim);
        float sine        = 0.0F;
        float cosine      = 0.0F;
        sincosf(angle, &sine, &cosine);
        const float first  = __bfloat162float(normalized[dim]);
        const float second = __bfloat162float(normalized[dim + 32]);
        output[dim]        = __float2bfloat16_rn(first * cosine - second * sine);
        output[dim + 32]   = __float2bfloat16_rn(second * cosine + first * sine);
    } else if (dim >= 64) {
        output[dim] = normalized[dim];
    }
}

__global__ void prepare_query_kernel(const __nv_bfloat16* __restrict__ projected,
                                     const __nv_bfloat16* __restrict__ norm,
                                     const std::int32_t* __restrict__ positions,
                                     __nv_bfloat16* __restrict__ query, int batch_size) {
    __shared__ float warp_squares[4];
    __shared__ __nv_bfloat16 normalized[kHeadDim];
    const int dim   = static_cast<int>(threadIdx.x);
    const int head  = static_cast<int>(blockIdx.x);
    const int batch = static_cast<int>(blockIdx.y);
    const int warp  = dim >> 5;
    const int lane  = dim & 31;
    const float x   = __bfloat162float(
        projected[static_cast<std::int64_t>(batch) * kProjectionRows + head * kHeadDim + dim]);
    const float square = ops::warp_reduce_sum(x * x);
    if (lane == 0) { warp_squares[warp] = square; }
    __syncthreads();
    const float sum = warp_squares[0] + warp_squares[1] + warp_squares[2] + warp_squares[3];
    normalized[dim] = __float2bfloat16_rn(x * rsqrtf(sum / static_cast<float>(kHeadDim) + 1.0e-6F) *
                                          (1.0F + __bfloat162float(norm[dim])));
    __syncthreads();
    std::int32_t local_positions[3] = {positions[batch], positions[batch_size + batch],
                                       positions[2 * batch_size + batch]};
    auto* destination =
        query + static_cast<std::int64_t>(batch) * kQueryHeads * kHeadDim + head * kHeadDim;
    store_rotated(normalized, local_positions, destination, dim);
}

__global__ void update_key_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ norm,
    const std::int32_t* __restrict__ token_indices, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ table_rows, const std::int32_t* __restrict__ source_slots,
    const std::int32_t* __restrict__ destination_slots, __nv_bfloat16* __restrict__ block_keys,
    const std::int32_t* __restrict__ block_tables, int logical_pages,
    __nv_bfloat16* __restrict__ raw_keys, std::int32_t* __restrict__ raw_positions,
    int batch_size) {
    __shared__ float warp_squares[4];
    __shared__ __nv_bfloat16 pooled[kHeadDim];
    __shared__ __nv_bfloat16 normalized[kHeadDim];
    const int dim                       = static_cast<int>(threadIdx.x);
    const int batch                     = static_cast<int>(blockIdx.x);
    const int warp                      = dim >> 5;
    const int lane                      = dim & 31;
    const int source                    = source_slots[batch];
    const int destination               = destination_slots[batch];
    const int token_index               = token_indices[batch];
    const int tail                      = token_index & 3;
    const std::int64_t source_base      = static_cast<std::int64_t>(source) * 4 * kHeadDim;
    const std::int64_t destination_base = static_cast<std::int64_t>(destination) * 4 * kHeadDim;
    for (int slot = 0; slot < 4; ++slot) {
        raw_keys[destination_base + slot * kHeadDim + dim] =
            raw_keys[source_base + slot * kHeadDim + dim];
    }
    if (dim < 12) {
        raw_positions[static_cast<std::int64_t>(destination) * 12 + dim] =
            raw_positions[static_cast<std::int64_t>(source) * 12 + dim];
    }
    __syncthreads();
    raw_keys[destination_base + tail * kHeadDim + dim] =
        projected[static_cast<std::int64_t>(batch) * kProjectionRows + kRawKeyOffset + dim];
    if (dim < 3) {
        raw_positions[static_cast<std::int64_t>(destination) * 12 + tail * 3 + dim] =
            positions[dim * batch_size + batch];
    }
    __syncthreads();
    if (tail != 3) { return; }

    float mean = 0.0F;
    for (int slot = 0; slot < 4; ++slot) {
        mean += __bfloat162float(raw_keys[destination_base + slot * kHeadDim + dim]);
    }
    pooled[dim]              = __float2bfloat16_rn(mean * 0.25F);
    const float pooled_float = __bfloat162float(pooled[dim]);
    const float square       = ops::warp_reduce_sum(pooled_float * pooled_float);
    if (lane == 0) { warp_squares[warp] = square; }
    __syncthreads();
    const float sum = warp_squares[0] + warp_squares[1] + warp_squares[2] + warp_squares[3];
    normalized[dim] =
        __float2bfloat16_rn(pooled_float * rsqrtf(sum / static_cast<float>(kHeadDim) + 1.0e-6F) *
                            (1.0F + __bfloat162float(norm[dim])));
    __syncthreads();

    const int block         = token_index / 4;
    const int logical_page  = block / kCompressedPage;
    const int page_offset   = block % kCompressedPage;
    const int table_row     = table_rows[batch];
    const int physical_page = block_tables[table_row * logical_pages + logical_page];
    auto* destination_key   = block_keys +
                            static_cast<std::int64_t>(physical_page) * kCompressedPage * kHeadDim +
                            page_offset * kHeadDim;
    const std::int32_t* block_positions =
        raw_positions + static_cast<std::int64_t>(destination) * 12;
    store_rotated(normalized, block_positions, destination_key, dim);
}

__global__ void initialize_sort_kernel(std::int32_t* ids, std::int32_t* offsets, int active_blocks,
                                       int batch_size) {
    const int index =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int items = active_blocks * batch_size;
    if (index < items) { ids[index] = index % active_blocks; }
    if (index <= batch_size) { offsets[index] = index * active_blocks; }
}

__global__ void score_blocks_kernel(const __nv_bfloat16* __restrict__ query,
                                    const __nv_bfloat16* __restrict__ block_keys,
                                    const std::int32_t* __restrict__ block_tables,
                                    const std::int32_t* __restrict__ table_rows,
                                    const std::int32_t* __restrict__ token_indices,
                                    int logical_pages, int active_blocks,
                                    float* __restrict__ scores) {
    __shared__ float head_scores[kQueryHeads];
    const int block                = static_cast<int>(blockIdx.x);
    const int batch                = static_cast<int>(blockIdx.y);
    const int tid                  = static_cast<int>(threadIdx.x);
    const int head                 = tid >> 5;
    const int lane                 = tid & 31;
    const int complete_blocks      = (token_indices[batch] + 1) / 4;
    const std::int64_t score_index = static_cast<std::int64_t>(batch) * active_blocks + block;
    if (block >= complete_blocks) {
        if (tid == 0) { scores[score_index] = -__int_as_float(0x7F800000); }
        return;
    }
    const int logical_page  = block / kCompressedPage;
    const int page_offset   = block % kCompressedPage;
    const int table_row     = table_rows[batch];
    const int physical_page = block_tables[table_row * logical_pages + logical_page];
    const auto* key         = block_keys +
                      static_cast<std::int64_t>(physical_page) * kCompressedPage * kHeadDim +
                      page_offset * kHeadDim;
    const auto* q =
        query + static_cast<std::int64_t>(batch) * kQueryHeads * kHeadDim + head * kHeadDim;
    float dot = 0.0F;
    for (int dim = lane; dim < kHeadDim; dim += 32) {
        dot = fmaf(__bfloat162float(q[dim]), __bfloat162float(key[dim]), dot);
    }
    dot = ops::warp_reduce_sum(dot);
    if (lane == 0) { head_scores[head] = dot; }
    __syncthreads();
    if (tid == 0) {
        float score = 0.0F;
        for (float value : head_scores) { score += fmaxf(value, 0.0F); }
        scores[score_index] = score * kIndexerScaling;
    }
}

// Exact fully-selected fast path: every complete block is kept, so the selection is the identity.
// Launched only when the host-known complete-block envelope is <= kSelectedBlocks.
__global__ void publish_identity_selection_kernel(const std::int32_t* __restrict__ token_indices,
                                                  std::int32_t* __restrict__ selected_blocks,
                                                  std::int32_t* __restrict__ selected_counts) {
    const int row             = static_cast<int>(blockIdx.x);
    const int tid             = static_cast<int>(threadIdx.x);
    const int complete_blocks = (token_indices[row] + 1) / 4;
    const int count           = min(complete_blocks, kSelectedBlocks);
    if (tid == 0) { selected_counts[row] = count; }
    for (int index = tid; index < kSelectedBlocks; index += static_cast<int>(blockDim.x)) {
        selected_blocks[static_cast<std::int64_t>(row) * kSelectedBlocks + index] =
            index < count ? index : -1;
    }
}

cudaError_t sort_pairs_descending(void* temp, std::size_t& temp_bytes, const float* keys_in,
                                  float* keys_out, const std::int32_t* values_in,
                                  std::int32_t* values_out, int items, int segments,
                                  const std::int32_t* offsets, cudaStream_t stream) {
    return cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp, temp_bytes, keys_in, keys_out, values_in, values_out, items, segments, offsets,
        offsets + 1, 0, static_cast<int>(sizeof(float) * 8), stream);
}

__device__ __forceinline__ std::uint32_t float_ascending_bits(float value) {
    const std::uint32_t bits = __float_as_uint(value);
    const std::uint32_t mask =
        (static_cast<std::int32_t>(bits) < 0) ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

__global__ void pack_topk_keys_kernel(const float* __restrict__ scores,
                                      const std::int32_t* __restrict__ ids,
                                      std::uint64_t* __restrict__ packed, int items) {
    const int index =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= items) { return; }
    const std::uint32_t rank = float_ascending_bits(scores[index]);
    const std::uint32_t tie  = ~static_cast<std::uint32_t>(ids[index]);
    packed[index]            = (static_cast<std::uint64_t>(rank) << 32) | tie;
}

__global__ void bitonic_sort_512_descending(std::uint64_t* keys, std::int32_t* ids) {
    __shared__ std::uint64_t skey[kSelectedBlocks];
    __shared__ std::int32_t sid[kSelectedBlocks];
    const int row = static_cast<int>(blockIdx.x);
    const int t   = static_cast<int>(threadIdx.x);
    keys += static_cast<std::int64_t>(row) * kSelectedBlocks;
    ids += static_cast<std::int64_t>(row) * kSelectedBlocks;
    skey[t] = keys[t];
    sid[t]  = ids[t];
    __syncthreads();
    for (int k = 2; k <= kSelectedBlocks; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int ixj = t ^ j;
            if (ixj > t) {
                const bool descending_half = (t & k) == 0;
                const bool out_of_order =
                    descending_half ? (skey[t] < skey[ixj]) : (skey[t] > skey[ixj]);
                if (out_of_order) {
                    const std::uint64_t tk = skey[t];
                    skey[t]                = skey[ixj];
                    skey[ixj]              = tk;
                    const std::int32_t ti  = sid[t];
                    sid[t]                 = sid[ixj];
                    sid[ixj]               = ti;
                }
            }
            __syncthreads();
        }
    }
    keys[t] = skey[t];
    ids[t]  = sid[t];
}

__global__ void publish_compact_selection_kernel(const std::int32_t* __restrict__ selected_ids,
                                                 const std::int32_t* __restrict__ token_indices,
                                                 std::int32_t* __restrict__ selected_blocks,
                                                 std::int32_t* __restrict__ selected_counts) {
    const int row             = static_cast<int>(blockIdx.x);
    const int tid             = static_cast<int>(threadIdx.x);
    const int complete_blocks = (token_indices[row] + 1) / 4;
    const int count           = min(complete_blocks, kSelectedBlocks);
    if (tid == 0) { selected_counts[row] = count; }
    for (int index = tid; index < kSelectedBlocks; index += static_cast<int>(blockDim.x)) {
        selected_blocks[static_cast<std::int64_t>(row) * kSelectedBlocks + index] =
            index < count ? selected_ids[static_cast<std::int64_t>(row) * kSelectedBlocks + index]
                          : -1;
    }
}

auto topk_env(cudaStream_t stream) {
    return cuda::std::execution::env(
        cuda::stream_ref{stream},
        cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                 cuda::execution::output_ordering::unsorted));
}

cudaError_t topk_max_pairs(void* temp, std::size_t& temp_bytes, const std::uint64_t* keys_in,
                           std::uint64_t* keys_out, const std::int32_t* values_in,
                           std::int32_t* values_out, int num_items, int k, cudaStream_t stream) {
    return cub::DeviceTopK::MaxPairs(temp, temp_bytes, keys_in, keys_out, values_in, values_out,
                                     num_items, k, topk_env(stream));
}

std::size_t topk_temp_bytes(int num_items, int k) {
    std::size_t bytes = 0;
    const auto* keys_in =
        reinterpret_cast<const std::uint64_t*>(std::uintptr_t{0x1000});
    auto* keys_out = reinterpret_cast<std::uint64_t*>(std::uintptr_t{0x2000});
    const auto* values_in =
        reinterpret_cast<const std::int32_t*>(std::uintptr_t{0x3000});
    auto* values_out = reinterpret_cast<std::int32_t*>(std::uintptr_t{0x4000});
    CUDA_CHECK(topk_max_pairs(nullptr, bytes, keys_in, keys_out, values_in, values_out, num_items,
                              k, nullptr));
    return bytes;
}

std::size_t packed_sort512_temp_bytes() {
    std::size_t bytes     = 0;
    const auto* keys_in   = reinterpret_cast<const std::uint64_t*>(std::uintptr_t{0x1000});
    auto* keys_out        = reinterpret_cast<std::uint64_t*>(std::uintptr_t{0x2000});
    const auto* values_in = reinterpret_cast<const std::int32_t*>(std::uintptr_t{0x3000});
    auto* values_out      = reinterpret_cast<std::int32_t*>(std::uintptr_t{0x4000});
    CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(nullptr, bytes, keys_in, keys_out,
                                                         values_in, values_out, kSelectedBlocks, 0,
                                                         64, nullptr));
    return bytes;
}

void select_top512_topk(const float* scores, const std::int32_t* ids, std::uint64_t* packed_keys,
                        std::uint64_t* packed_selected, std::uint64_t* packed_sorted,
                        std::int32_t* topk_ids, std::int32_t* sorted_topk_ids, std::int32_t* offsets,
                        void* temp, std::size_t temp_capacity, int active_blocks, int batch,
                        cudaStream_t stream) {
    constexpr int threads = 256;
    const int items       = active_blocks * batch;
    pack_topk_keys_kernel<<<(items + threads - 1) / threads, threads, 0, stream>>>(
        scores, ids, packed_keys, items);
    CUDA_CHECK(cudaGetLastError());
    for (int row = 0; row < batch; ++row) {
        const std::int64_t packed_row = static_cast<std::int64_t>(row) * active_blocks;
        const std::int64_t topk_row   = static_cast<std::int64_t>(row) * kSelectedBlocks;
        std::size_t temp_bytes        = temp_capacity;
        CUDA_CHECK(topk_max_pairs(temp, temp_bytes, packed_keys + packed_row,
                                  packed_selected + topk_row, ids + packed_row, topk_ids + topk_row,
                                  active_blocks, kSelectedBlocks, stream));
    }
    bitonic_sort_512_descending<<<batch, kSelectedBlocks, 0, stream>>>(packed_selected, topk_ids);
    CUDA_CHECK(cudaGetLastError());
    (void)packed_sorted;
    (void)sorted_topk_ids;
    (void)offsets;
}

} // namespace

std::size_t flash_next_qsa_indexer_sort_temp_bytes(std::int32_t maximum_blocks,
                                                   std::int32_t batch) {
    std::size_t sort_bytes = 0;
    const auto* keys_in    = reinterpret_cast<const float*>(std::uintptr_t{0x1000});
    auto* keys_out         = reinterpret_cast<float*>(std::uintptr_t{0x2000});
    const auto* values_in  = reinterpret_cast<const std::int32_t*>(std::uintptr_t{0x3000});
    auto* values_out       = reinterpret_cast<std::int32_t*>(std::uintptr_t{0x4000});
    const auto* offsets    = reinterpret_cast<const std::int32_t*>(std::uintptr_t{0x5000});
    CUDA_CHECK(sort_pairs_descending(nullptr, sort_bytes, keys_in, keys_out, values_in, values_out,
                                     maximum_blocks * batch, batch, offsets, nullptr));
    const std::size_t topk_bytes = topk_temp_bytes(maximum_blocks, kSelectedBlocks);
    const std::size_t sort512    = packed_sort512_temp_bytes();
    return std::max(sort_bytes, std::max(topk_bytes, sort512));
}

void flash_next_qsa_indexer_launch(const Tensor& token_indices, const Tensor& mrope_positions,
                                   const Tensor& table_rows, const Tensor& source_state_slots,
                                   const Tensor& destination_state_slots, const Tensor& query_norm,
                                   const Tensor& key_norm, QsaIndexerCacheView cache,
                                   FlashNextQsaIndexerWorkspace& scratch, int active_blocks,
                                   Tensor& selected_blocks, Tensor& selected_counts,
                                   cudaStream_t stream) {
    const int batch = token_indices.ne[0];
    update_key_kernel<<<batch, kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.projected.data),
        static_cast<const __nv_bfloat16*>(key_norm.data),
        static_cast<const std::int32_t*>(token_indices.data),
        static_cast<const std::int32_t*>(mrope_positions.data),
        static_cast<const std::int32_t*>(table_rows.data),
        static_cast<const std::int32_t*>(source_state_slots.data),
        static_cast<const std::int32_t*>(destination_state_slots.data),
        static_cast<__nv_bfloat16*>(cache.block_keys.data),
        static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
        static_cast<__nv_bfloat16*>(cache.raw_keys.data),
        static_cast<std::int32_t*>(cache.raw_positions.data), batch);
    CUDA_CHECK(cudaGetLastError());

    if (active_blocks == 0) {
        CUDA_CHECK(cudaMemsetAsync(selected_counts.data, 0,
                                   static_cast<std::size_t>(batch) * sizeof(std::int32_t), stream));
        return;
    }

    if (active_blocks <= kSelectedBlocks) {
        publish_identity_selection_kernel<<<batch, 256, 0, stream>>>(
            static_cast<const std::int32_t*>(token_indices.data),
            static_cast<std::int32_t*>(selected_blocks.data),
            static_cast<std::int32_t*>(selected_counts.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    prepare_query_kernel<<<dim3(kQueryHeads, batch), kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.projected.data),
        static_cast<const __nv_bfloat16*>(query_norm.data),
        static_cast<const std::int32_t*>(mrope_positions.data),
        static_cast<__nv_bfloat16*>(scratch.query.data), batch);
    CUDA_CHECK(cudaGetLastError());
    constexpr int threads = 256;
    const int items       = active_blocks * batch;
    initialize_sort_kernel<<<(items + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<std::int32_t*>(scratch.ids.data),
        static_cast<std::int32_t*>(scratch.offsets.data), active_blocks, batch);
    CUDA_CHECK(cudaGetLastError());
    score_blocks_kernel<<<dim3(active_blocks, batch), kQueryHeads * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.query.data),
        static_cast<const __nv_bfloat16*>(cache.block_keys.data),
        static_cast<const std::int32_t*>(cache.block_tables.data),
        static_cast<const std::int32_t*>(table_rows.data),
        static_cast<const std::int32_t*>(token_indices.data), cache.block_tables.ne[0],
        active_blocks, static_cast<float*>(scratch.scores.data));
    CUDA_CHECK(cudaGetLastError());
    select_top512_topk(
        static_cast<const float*>(scratch.scores.data),
        static_cast<const std::int32_t*>(scratch.ids.data),
        static_cast<std::uint64_t*>(scratch.packed_keys.data),
        static_cast<std::uint64_t*>(scratch.packed_selected.data),
        static_cast<std::uint64_t*>(scratch.packed_sorted.data),
        static_cast<std::int32_t*>(scratch.topk_ids.data),
        static_cast<std::int32_t*>(scratch.sorted_ids.data),
        static_cast<std::int32_t*>(scratch.offsets.data), scratch.sort_temp.data,
        scratch.sort_temp.bytes, active_blocks, batch, stream);
    publish_compact_selection_kernel<<<batch, 256, 0, stream>>>(
        static_cast<const std::int32_t*>(scratch.topk_ids.data),
        static_cast<const std::int32_t*>(token_indices.data),
        static_cast<std::int32_t*>(selected_blocks.data),
        static_cast<std::int32_t*>(selected_counts.data));
    CUDA_CHECK(cudaGetLastError());
}

__global__ void indexer_publish_complete_blocks_chunk_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ norm,
    const std::int32_t* __restrict__ token_indices, const std::int32_t* __restrict__ positions,
    int source_slot, int destination_slot, int table_row,
    const std::int32_t* __restrict__ block_tables, int logical_pages,
    __nv_bfloat16* __restrict__ block_keys, const __nv_bfloat16* __restrict__ raw_keys,
    const std::int32_t* __restrict__ raw_positions, int tokens) {
    __shared__ float warp_squares[4];
    __shared__ __nv_bfloat16 pooled[kHeadDim];
    __shared__ __nv_bfloat16 normalized[kHeadDim];
    __shared__ std::int32_t first_token_pos[3];

    const int dim               = static_cast<int>(threadIdx.x);
    const int b                 = static_cast<int>(blockIdx.x);
    const int warp              = dim >> 5;
    const int lane              = dim & 31;
    const int first_token_index = token_indices[0];
    const int leftover_in       = first_token_index & 3;
    const int complete_blocks   = (leftover_in + tokens) / 4;
    if (b >= complete_blocks) { return; }

    const std::int64_t source_base = static_cast<std::int64_t>(source_slot) * 4 * kHeadDim;

    // First token of block b: s0 = 4 * b + 0
    if (dim < 3) {
        const int s0 = 4 * b;
        if (s0 < leftover_in) {
            first_token_pos[dim] =
                raw_positions[static_cast<std::int64_t>(source_slot) * 12 + s0 * 3 + dim];
        } else {
            const int t0         = s0 - leftover_in;
            first_token_pos[dim] = positions[dim * tokens + t0];
        }
    }
    __syncthreads();

    // 4 tokens of block b: s = 4 * b + slot (slot = 0, 1, 2, 3)
    float sum_raw = 0.0F;
    for (int slot = 0; slot < 4; ++slot) {
        const int s = 4 * b + slot;
        float val   = 0.0F;
        if (s < leftover_in) {
            val = __bfloat162float(raw_keys[source_base + slot * kHeadDim + dim]);
        } else {
            const int t = s - leftover_in;
            val         = __bfloat162float(
                projected[static_cast<std::int64_t>(t) * kProjectionRows + kRawKeyOffset + dim]);
        }
        sum_raw += val;
    }

    pooled[dim]              = __float2bfloat16_rn(sum_raw * 0.25F);
    const float pooled_float = __bfloat162float(pooled[dim]);
    const float square       = ops::warp_reduce_sum(pooled_float * pooled_float);
    if (lane == 0) { warp_squares[warp] = square; }
    __syncthreads();
    const float sum = warp_squares[0] + warp_squares[1] + warp_squares[2] + warp_squares[3];
    normalized[dim] =
        __float2bfloat16_rn(pooled_float * rsqrtf(sum / static_cast<float>(kHeadDim) + 1.0e-6F) *
                            (1.0F + __bfloat162float(norm[dim])));
    __syncthreads();

    const int global_block  = (first_token_index - leftover_in) / 4 + b;
    const int logical_page  = global_block / kCompressedPage;
    const int page_offset   = global_block % kCompressedPage;
    const int physical_page = block_tables[table_row * logical_pages + logical_page];
    auto* destination_key   = block_keys +
                            static_cast<std::int64_t>(physical_page) * kCompressedPage * kHeadDim +
                            page_offset * kHeadDim;
    store_rotated(normalized, first_token_pos, destination_key, dim);
}

__global__ void indexer_update_leftover_chunk_kernel(
    const __nv_bfloat16* __restrict__ projected, const std::int32_t* __restrict__ token_indices,
    const std::int32_t* __restrict__ positions, int source_slot, int destination_slot,
    __nv_bfloat16* __restrict__ raw_keys, std::int32_t* __restrict__ raw_positions, int tokens) {
    const int dim               = static_cast<int>(threadIdx.x);
    const int first_token_index = token_indices[0];
    const int leftover_in       = first_token_index & 3;
    const int complete_blocks   = (leftover_in + tokens) / 4;
    const int leftover_out      = (leftover_in + tokens) & 3;

    const std::int64_t source_base      = static_cast<std::int64_t>(source_slot) * 4 * kHeadDim;
    const std::int64_t destination_base = static_cast<std::int64_t>(destination_slot) * 4 * kHeadDim;

    for (int slot = 0; slot < leftover_out; ++slot) {
        const int s = complete_blocks * 4 + slot;
        if (s < leftover_in) {
            raw_keys[destination_base + slot * kHeadDim + dim] =
                raw_keys[source_base + s * kHeadDim + dim];
            if (dim < 3) {
                raw_positions[static_cast<std::int64_t>(destination_slot) * 12 + slot * 3 + dim] =
                    raw_positions[static_cast<std::int64_t>(source_slot) * 12 + s * 3 + dim];
            }
        } else {
            const int t = s - leftover_in;
            raw_keys[destination_base + slot * kHeadDim + dim] =
                projected[static_cast<std::int64_t>(t) * kProjectionRows + kRawKeyOffset + dim];
            if (dim < 3) {
                raw_positions[static_cast<std::int64_t>(destination_slot) * 12 + slot * 3 + dim] =
                    positions[dim * tokens + t];
            }
        }
    }
}

__global__ void score_blocks_chunk_kernel(const __nv_bfloat16* __restrict__ query,
                                          const __nv_bfloat16* __restrict__ block_keys,
                                          const std::int32_t* __restrict__ block_tables,
                                          int table_row,
                                          const std::int32_t* __restrict__ token_indices,
                                          int logical_pages, int active_blocks,
                                          float* __restrict__ scores) {
    __shared__ float head_scores[kQueryHeads];
    const int block                = static_cast<int>(blockIdx.x);
    const int row                  = static_cast<int>(blockIdx.y);
    const int tid                  = static_cast<int>(threadIdx.x);
    const int head                 = tid >> 5;
    const int lane                 = tid & 31;
    const int token_index          = token_indices[row];
    const int complete_blocks      = (token_index + 1) / 4;
    const std::int64_t score_index = static_cast<std::int64_t>(row) * active_blocks + block;
    if (block >= complete_blocks) {
        if (tid == 0) { scores[score_index] = -__int_as_float(0x7F800000); }
        return;
    }
    const int logical_page  = block / kCompressedPage;
    const int page_offset   = block % kCompressedPage;
    const int physical_page = block_tables[table_row * logical_pages + logical_page];
    const auto* key         = block_keys +
                      static_cast<std::int64_t>(physical_page) * kCompressedPage * kHeadDim +
                      page_offset * kHeadDim;
    const auto* q =
        query + static_cast<std::int64_t>(row) * kQueryHeads * kHeadDim + head * kHeadDim;
    float dot = 0.0F;
    for (int dim = lane; dim < kHeadDim; dim += 32) {
        dot = fmaf(__bfloat162float(q[dim]), __bfloat162float(key[dim]), dot);
    }
    dot = ops::warp_reduce_sum(dot);
    if (lane == 0) { head_scores[head] = dot; }
    __syncthreads();
    if (tid == 0) {
        float score = 0.0F;
        for (float value : head_scores) { score += fmaxf(value, 0.0F); }
        scores[score_index] = score * kIndexerScaling;
    }
}

void flash_next_qsa_indexer_prefill_launch(
    const Tensor& token_indices, const Tensor& mrope_positions, std::int32_t table_row,
    std::int32_t source_state_slot, std::int32_t destination_state_slot, const Tensor& query_norm,
    const Tensor& key_norm, QsaIndexerCacheView cache, FlashNextQsaIndexerWorkspace& scratch,
    std::int32_t maximum_blocks, Tensor& selected_blocks, Tensor& selected_counts,
    cudaStream_t stream) {
    const int tokens = token_indices.ne[0];
    const int max_complete_blocks = (tokens + 3) / 4;
    if (max_complete_blocks > 0) {
        indexer_publish_complete_blocks_chunk_kernel<<<max_complete_blocks, kHeadDim, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(scratch.projected.data),
            static_cast<const __nv_bfloat16*>(key_norm.data),
            static_cast<const std::int32_t*>(token_indices.data),
            static_cast<const std::int32_t*>(mrope_positions.data), source_state_slot,
            destination_state_slot, table_row,
            static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
            static_cast<__nv_bfloat16*>(cache.block_keys.data),
            static_cast<const __nv_bfloat16*>(cache.raw_keys.data),
            static_cast<const std::int32_t*>(cache.raw_positions.data), tokens);
        CUDA_CHECK(cudaGetLastError());
    }

    indexer_update_leftover_chunk_kernel<<<1, kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.projected.data),
        static_cast<const std::int32_t*>(token_indices.data),
        static_cast<const std::int32_t*>(mrope_positions.data), source_state_slot,
        destination_state_slot, static_cast<__nv_bfloat16*>(cache.raw_keys.data),
        static_cast<std::int32_t*>(cache.raw_positions.data), tokens);
    CUDA_CHECK(cudaGetLastError());

    if (maximum_blocks <= kSelectedBlocks) {
        publish_identity_selection_kernel<<<tokens, 256, 0, stream>>>(
            static_cast<const std::int32_t*>(token_indices.data),
            static_cast<std::int32_t*>(selected_blocks.data),
            static_cast<std::int32_t*>(selected_counts.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    prepare_query_kernel<<<dim3(kQueryHeads, tokens), kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.projected.data),
        static_cast<const __nv_bfloat16*>(query_norm.data),
        static_cast<const std::int32_t*>(mrope_positions.data),
        static_cast<__nv_bfloat16*>(scratch.query.data), tokens);
    CUDA_CHECK(cudaGetLastError());

    const int tile_size     = flash_next_qsa_indexer_tile_size(maximum_blocks, tokens);
    const int active_blocks = maximum_blocks;

    for (int t_start = 0; t_start < tokens; t_start += tile_size) {
        const int current_tile = min(tile_size, tokens - t_start);
        constexpr int threads  = 256;
        const int items        = active_blocks * current_tile;

        initialize_sort_kernel<<<(items + threads - 1) / threads, threads, 0, stream>>>(
            static_cast<std::int32_t*>(scratch.ids.data),
            static_cast<std::int32_t*>(scratch.offsets.data), active_blocks, current_tile);
        CUDA_CHECK(cudaGetLastError());

        score_blocks_chunk_kernel<<<dim3(active_blocks, current_tile), kQueryHeads * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(scratch.query.data) +
                static_cast<std::int64_t>(t_start) * kQueryHeads * kHeadDim,
            static_cast<const __nv_bfloat16*>(cache.block_keys.data),
            static_cast<const std::int32_t*>(cache.block_tables.data), table_row,
            static_cast<const std::int32_t*>(token_indices.data) + t_start, cache.block_tables.ne[0],
            active_blocks, static_cast<float*>(scratch.scores.data));
        CUDA_CHECK(cudaGetLastError());

        select_top512_topk(
            static_cast<const float*>(scratch.scores.data),
            static_cast<const std::int32_t*>(scratch.ids.data),
            static_cast<std::uint64_t*>(scratch.packed_keys.data),
            static_cast<std::uint64_t*>(scratch.packed_selected.data),
            static_cast<std::uint64_t*>(scratch.packed_sorted.data),
            static_cast<std::int32_t*>(scratch.topk_ids.data),
            static_cast<std::int32_t*>(scratch.sorted_ids.data),
            static_cast<std::int32_t*>(scratch.offsets.data), scratch.sort_temp.data,
            scratch.sort_temp.bytes, active_blocks, current_tile, stream);

        publish_compact_selection_kernel<<<current_tile, 256, 0, stream>>>(
            static_cast<const std::int32_t*>(scratch.topk_ids.data),
            static_cast<const std::int32_t*>(token_indices.data) + t_start,
            static_cast<std::int32_t*>(selected_blocks.data) +
                static_cast<std::int64_t>(t_start) * kSelectedBlocks,
            static_cast<std::int32_t*>(selected_counts.data) + t_start);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
