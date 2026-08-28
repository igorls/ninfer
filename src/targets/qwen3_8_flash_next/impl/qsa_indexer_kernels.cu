#include "targets/qwen3_8_flash_next/impl/qsa_indexer_kernels.h"

#include "core/device.h"
#include "ops/common/warp.cuh"

#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

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

__global__ void publish_selection_kernel(const std::int32_t* __restrict__ sorted_ids,
                                         const std::int32_t* __restrict__ token_indices,
                                         int active_blocks,
                                         std::int32_t* __restrict__ selected_blocks,
                                         std::int32_t* __restrict__ selected_counts) {
    const int batch           = static_cast<int>(blockIdx.x);
    const int tid             = static_cast<int>(threadIdx.x);
    const int complete_blocks = (token_indices[batch] + 1) / 4;
    const int count           = min(complete_blocks, kSelectedBlocks);
    if (tid == 0) { selected_counts[batch] = count; }
    for (int index = tid; index < kSelectedBlocks; index += static_cast<int>(blockDim.x)) {
        selected_blocks[static_cast<std::int64_t>(batch) * kSelectedBlocks + index] =
            index < count ? sorted_ids[static_cast<std::int64_t>(batch) * active_blocks + index]
                          : -1;
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

} // namespace

std::size_t flash_next_qsa_indexer_sort_temp_bytes(std::int32_t maximum_blocks,
                                                   std::int32_t batch) {
    std::size_t bytes     = 0;
    const auto* keys_in   = reinterpret_cast<const float*>(std::uintptr_t{0x1000});
    auto* keys_out        = reinterpret_cast<float*>(std::uintptr_t{0x2000});
    const auto* values_in = reinterpret_cast<const std::int32_t*>(std::uintptr_t{0x3000});
    auto* values_out      = reinterpret_cast<std::int32_t*>(std::uintptr_t{0x4000});
    const auto* offsets   = reinterpret_cast<const std::int32_t*>(std::uintptr_t{0x5000});
    CUDA_CHECK(sort_pairs_descending(nullptr, bytes, keys_in, keys_out, values_in, values_out,
                                     maximum_blocks * batch, batch, offsets, nullptr));
    return bytes;
}

void flash_next_qsa_indexer_launch(const Tensor& token_indices, const Tensor& mrope_positions,
                                   const Tensor& table_rows, const Tensor& source_state_slots,
                                   const Tensor& destination_state_slots, const Tensor& query_norm,
                                   const Tensor& key_norm, QsaIndexerCacheView cache,
                                   FlashNextQsaIndexerWorkspace& scratch, int active_blocks,
                                   Tensor& selected_blocks, Tensor& selected_counts,
                                   cudaStream_t stream) {
    const int batch = token_indices.ne[0];
    prepare_query_kernel<<<dim3(kQueryHeads, batch), kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.projected.data),
        static_cast<const __nv_bfloat16*>(query_norm.data),
        static_cast<const std::int32_t*>(mrope_positions.data),
        static_cast<__nv_bfloat16*>(scratch.query.data), batch);
    CUDA_CHECK(cudaGetLastError());
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
    std::size_t temp_bytes = scratch.sort_temp.bytes;
    CUDA_CHECK(sort_pairs_descending(
        scratch.sort_temp.data, temp_bytes, static_cast<const float*>(scratch.scores.data),
        static_cast<float*>(scratch.sorted_scores.data),
        static_cast<const std::int32_t*>(scratch.ids.data),
        static_cast<std::int32_t*>(scratch.sorted_ids.data), items, batch,
        static_cast<const std::int32_t*>(scratch.offsets.data), stream));
    publish_selection_kernel<<<batch, 256, 0, stream>>>(
        static_cast<const std::int32_t*>(scratch.sorted_ids.data),
        static_cast<const std::int32_t*>(token_indices.data), active_blocks,
        static_cast<std::int32_t*>(selected_blocks.data),
        static_cast<std::int32_t*>(selected_counts.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
