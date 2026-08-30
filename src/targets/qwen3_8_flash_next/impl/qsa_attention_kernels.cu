#include "targets/qwen3_8_flash_next/impl/qsa_attention_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

constexpr int kHeadDim         = 256;
constexpr int kQueryHeads      = 24;
constexpr int kKvHeads         = 2;
constexpr int kProjectedRows   = 13'312;
constexpr int kMainKeyOffset   = 12'288;
constexpr int kMainValueOffset = 12'800;
constexpr int kPageTokens      = 64;
constexpr int kSelectedBlocks  = 512;
constexpr float kRopeTheta     = 1.0e7F;
constexpr float kScale         = 0.0625F; // 1/sqrt(256)

__device__ float rope_frequency(int pair) {
    return expf((-2.0F * static_cast<float>(pair) / 64.0F) * logf(kRopeTheta));
}

__device__ void store_mrope(const __nv_bfloat16* normalized, const std::int32_t* positions,
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
                                     __nv_bfloat16* __restrict__ query,
                                     __nv_bfloat16* __restrict__ gate, int batch_size) {
    __shared__ float warp_squares[8];
    __shared__ __nv_bfloat16 normalized[kHeadDim];
    const int dim   = static_cast<int>(threadIdx.x);
    const int head  = static_cast<int>(blockIdx.x);
    const int batch = static_cast<int>(blockIdx.y);
    const int warp  = dim >> 5;
    const int lane  = dim & 31;
    const std::int64_t source =
        static_cast<std::int64_t>(batch) * kProjectedRows + head * 2 * kHeadDim;
    const float x      = __bfloat162float(projected[source + dim]);
    const float square = ops::warp_reduce_sum(x * x);
    if (lane == 0) { warp_squares[warp] = square; }
    gate[static_cast<std::int64_t>(batch) * kQueryHeads * kHeadDim + head * kHeadDim + dim] =
        projected[source + kHeadDim + dim];
    __syncthreads();
    float sum = 0.0F;
    for (float value : warp_squares) { sum += value; }
    normalized[dim] = __float2bfloat16_rn(x * rsqrtf(sum / static_cast<float>(kHeadDim) + 1.0e-6F) *
                                          (1.0F + __bfloat162float(norm[dim])));
    __syncthreads();
    std::int32_t local_positions[3] = {positions[batch], positions[batch_size + batch],
                                       positions[2 * batch_size + batch]};
    auto* destination =
        query + static_cast<std::int64_t>(batch) * kQueryHeads * kHeadDim + head * kHeadDim;
    store_mrope(normalized, local_positions, destination, dim);
}

__global__ void prepare_append_kv_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ norm,
    const std::int32_t* __restrict__ token_indices, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ table_rows, const std::int32_t* __restrict__ block_tables,
    int logical_pages, __nv_bfloat16* __restrict__ key_pages,
    __nv_bfloat16* __restrict__ value_pages, __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value, int batch_size) {
    __shared__ float warp_squares[8];
    __shared__ __nv_bfloat16 normalized[kHeadDim];
    const int dim                     = static_cast<int>(threadIdx.x);
    const int head                    = static_cast<int>(blockIdx.x);
    const int batch                   = static_cast<int>(blockIdx.y);
    const int warp                    = dim >> 5;
    const int lane                    = dim & 31;
    const std::int64_t projected_base = static_cast<std::int64_t>(batch) * kProjectedRows;
    const float x =
        __bfloat162float(projected[projected_base + kMainKeyOffset + head * kHeadDim + dim]);
    const float square = ops::warp_reduce_sum(x * x);
    if (lane == 0) { warp_squares[warp] = square; }
    const auto value_word = projected[projected_base + kMainValueOffset + head * kHeadDim + dim];
    value[static_cast<std::int64_t>(batch) * kKvHeads * kHeadDim + head * kHeadDim + dim] =
        value_word;
    __syncthreads();
    float sum = 0.0F;
    for (float entry : warp_squares) { sum += entry; }
    normalized[dim] = __float2bfloat16_rn(x * rsqrtf(sum / static_cast<float>(kHeadDim) + 1.0e-6F) *
                                          (1.0F + __bfloat162float(norm[dim])));
    __syncthreads();
    std::int32_t local_positions[3] = {positions[batch], positions[batch_size + batch],
                                       positions[2 * batch_size + batch]};
    auto* key_destination =
        key + static_cast<std::int64_t>(batch) * kKvHeads * kHeadDim + head * kHeadDim;
    store_mrope(normalized, local_positions, key_destination, dim);
    __syncthreads();

    const int token         = token_indices[batch];
    const int logical_page  = token / kPageTokens;
    const int page_offset   = token % kPageTokens;
    const int physical_page = block_tables[table_rows[batch] * logical_pages + logical_page];
    const std::int64_t page_index =
        ((static_cast<std::int64_t>(physical_page) * kKvHeads + head) * kPageTokens + page_offset) *
            kHeadDim +
        dim;
    key_pages[page_index]   = key_destination[dim];
    value_pages[page_index] = value_word;
}

__device__ int selected_token(int ordinal, int selected_count, int complete_blocks,
                              const std::int32_t* selected) {
    const int selected_tokens = selected_count * 4;
    if (ordinal < selected_tokens) { return selected[ordinal / 4] * 4 + ordinal % 4; }
    return complete_blocks * 4 + ordinal - selected_tokens;
}

__global__ void sparse_attention_kernel(
    const __nv_bfloat16* __restrict__ query, const std::int32_t* __restrict__ token_indices,
    const std::int32_t* __restrict__ table_rows, const std::int32_t* __restrict__ selected_blocks,
    const std::int32_t* __restrict__ selected_counts, const std::int32_t* __restrict__ block_tables,
    int logical_pages, const __nv_bfloat16* __restrict__ key_pages,
    const __nv_bfloat16* __restrict__ value_pages, __nv_bfloat16* __restrict__ output) {
    __shared__ float scores[256];
    __shared__ float reduction[256];
    const int dim             = static_cast<int>(threadIdx.x);
    const int head            = static_cast<int>(blockIdx.x);
    const int batch           = static_cast<int>(blockIdx.y);
    const int kv_head         = head / 12;
    const int token_index     = token_indices[batch];
    const int complete_blocks = (token_index + 1) / 4;
    const int tail_count      = (token_index + 1) & 3;
    const int selected_count  = selected_counts[batch];
    const int total           = selected_count * 4 + tail_count;
    const auto* selected = selected_blocks + static_cast<std::int64_t>(batch) * kSelectedBlocks;
    const auto* q =
        query + static_cast<std::int64_t>(batch) * kQueryHeads * kHeadDim + head * kHeadDim;
    float accumulator = 0.0F;
    float running_max = -__int_as_float(0x7F800000);
    float running_sum = 0.0F;

    for (int start = 0; start < total; start += 256) {
        const int count = min(256, total - start);
        float score     = -__int_as_float(0x7F800000);
        if (dim < count) {
            const int token =
                selected_token(start + dim, selected_count, complete_blocks, selected);
            const int logical_page = token / kPageTokens;
            const int page_offset  = token % kPageTokens;
            const int physical_page =
                block_tables[table_rows[batch] * logical_pages + logical_page];
            const auto* cache_key =
                key_pages +
                ((static_cast<std::int64_t>(physical_page) * kKvHeads + kv_head) * kPageTokens +
                 page_offset) *
                    kHeadDim;
            score = 0.0F;
            for (int feature = 0; feature < kHeadDim; ++feature) {
                score =
                    fmaf(__bfloat162float(q[feature]), __bfloat162float(cache_key[feature]), score);
            }
            score *= kScale;
        }
        scores[dim]    = score;
        reduction[dim] = score;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (dim < stride) { reduction[dim] = fmaxf(reduction[dim], reduction[dim + stride]); }
            __syncthreads();
        }
        const float chunk_max = reduction[0];
        // Every thread must have read reduction[0] before thread 0 overwrites it below; without
        // this barrier a late warp reads a probability instead of the max (intermittent, one
        // (head, token) output per hit).
        __syncthreads();
        const float probability = dim < count ? expf(scores[dim] - chunk_max) : 0.0F;
        scores[dim]             = probability;
        reduction[dim]          = probability;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (dim < stride) { reduction[dim] += reduction[dim + stride]; }
            __syncthreads();
        }
        const float chunk_sum   = reduction[0];
        const float next_max    = fmaxf(running_max, chunk_max);
        const float prior_scale = running_sum == 0.0F ? 0.0F : expf(running_max - next_max);
        const float chunk_scale = expf(chunk_max - next_max);
        float chunk_value       = 0.0F;
        for (int local = 0; local < count; ++local) {
            const int token =
                selected_token(start + local, selected_count, complete_blocks, selected);
            const int logical_page = token / kPageTokens;
            const int page_offset  = token % kPageTokens;
            const int physical_page =
                block_tables[table_rows[batch] * logical_pages + logical_page];
            const std::int64_t cache_index =
                ((static_cast<std::int64_t>(physical_page) * kKvHeads + kv_head) * kPageTokens +
                 page_offset) *
                    kHeadDim +
                dim;
            chunk_value =
                fmaf(scores[local], __bfloat162float(value_pages[cache_index]), chunk_value);
        }
        accumulator = accumulator * prior_scale + chunk_value * chunk_scale;
        running_sum = running_sum * prior_scale + chunk_sum * chunk_scale;
        running_max = next_max;
        __syncthreads();
    }
    output[static_cast<std::int64_t>(batch) * kQueryHeads * kHeadDim + head * kHeadDim + dim] =
        __float2bfloat16_rn(accumulator / running_sum);
}

__global__ void gate_output_kernel(const __nv_bfloat16* __restrict__ attended,
                                   const __nv_bfloat16* __restrict__ gate,
                                   __nv_bfloat16* __restrict__ gated, int elements) {
    const int index =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index < elements) {
        gated[index] = __float2bfloat16_rn(__bfloat162float(attended[index]) *
                                           ops::sigmoid(__bfloat162float(gate[index])));
    }
}

} // namespace

void flash_next_qsa_attention_launch(const Tensor& token_indices, const Tensor& mrope_positions,
                                     const Tensor& table_rows, const Tensor& selected_blocks,
                                     const Tensor& selected_counts, const Tensor& query_norm,
                                     const Tensor& key_norm, QsaAttentionCacheView cache,
                                     FlashNextQsaAttentionWorkspace& scratch, cudaStream_t stream) {
    const int batch = token_indices.ne[0];
    prepare_query_kernel<<<dim3(kQueryHeads, batch), kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.projected.data),
        static_cast<const __nv_bfloat16*>(query_norm.data),
        static_cast<const std::int32_t*>(mrope_positions.data),
        static_cast<__nv_bfloat16*>(scratch.query.data),
        static_cast<__nv_bfloat16*>(scratch.gate.data), batch);
    CUDA_CHECK(cudaGetLastError());
    prepare_append_kv_kernel<<<dim3(kKvHeads, batch), kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.projected.data),
        static_cast<const __nv_bfloat16*>(key_norm.data),
        static_cast<const std::int32_t*>(token_indices.data),
        static_cast<const std::int32_t*>(mrope_positions.data),
        static_cast<const std::int32_t*>(table_rows.data),
        static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
        static_cast<__nv_bfloat16*>(cache.key_pages.data),
        static_cast<__nv_bfloat16*>(cache.value_pages.data),
        static_cast<__nv_bfloat16*>(scratch.key.data),
        static_cast<__nv_bfloat16*>(scratch.value.data), batch);
    CUDA_CHECK(cudaGetLastError());
    sparse_attention_kernel<<<dim3(kQueryHeads, batch), kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.query.data),
        static_cast<const std::int32_t*>(token_indices.data),
        static_cast<const std::int32_t*>(table_rows.data),
        static_cast<const std::int32_t*>(selected_blocks.data),
        static_cast<const std::int32_t*>(selected_counts.data),
        static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
        static_cast<const __nv_bfloat16*>(cache.key_pages.data),
        static_cast<const __nv_bfloat16*>(cache.value_pages.data),
        static_cast<__nv_bfloat16*>(scratch.attended.data));
    CUDA_CHECK(cudaGetLastError());
    const int elements    = kQueryHeads * kHeadDim * batch;
    constexpr int threads = 256;
    gate_output_kernel<<<(elements + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.attended.data),
        static_cast<const __nv_bfloat16*>(scratch.gate.data),
        static_cast<__nv_bfloat16*>(scratch.gated.data), elements);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void qsa_prefill_prepare_query_kernel(const __nv_bfloat16* __restrict__ projected,
                                                 const __nv_bfloat16* __restrict__ norm,
                                                 const std::int32_t* __restrict__ positions,
                                                 __nv_bfloat16* __restrict__ query,
                                                 __nv_bfloat16* __restrict__ gate, int tokens) {
    __shared__ float warp_squares[8];
    __shared__ __nv_bfloat16 normalized[kHeadDim];
    const int dim   = static_cast<int>(threadIdx.x);
    const int head  = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int warp  = dim >> 5;
    const int lane  = dim & 31;
    const std::int64_t source =
        static_cast<std::int64_t>(token) * kProjectedRows + head * 2 * kHeadDim;
    const float x      = __bfloat162float(projected[source + dim]);
    const float square = ops::warp_reduce_sum(x * x);
    if (lane == 0) { warp_squares[warp] = square; }
    gate[static_cast<std::int64_t>(token) * kQueryHeads * kHeadDim + head * kHeadDim + dim] =
        projected[source + kHeadDim + dim];
    __syncthreads();
    float sum = 0.0F;
    for (float value : warp_squares) { sum += value; }
    normalized[dim] = __float2bfloat16_rn(x * rsqrtf(sum / static_cast<float>(kHeadDim) + 1.0e-6F) *
                                          (1.0F + __bfloat162float(norm[dim])));
    __syncthreads();
    std::int32_t local_positions[3] = {positions[token], positions[tokens + token],
                                       positions[2 * tokens + token]};
    auto* destination =
        query + static_cast<std::int64_t>(token) * kQueryHeads * kHeadDim + head * kHeadDim;
    store_mrope(normalized, local_positions, destination, dim);
}

__global__ void qsa_prefill_prepare_append_kv_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ norm,
    const std::int32_t* __restrict__ token_indices, const std::int32_t* __restrict__ positions,
    int table_row, const std::int32_t* __restrict__ block_tables, int logical_pages,
    __nv_bfloat16* __restrict__ key_pages, __nv_bfloat16* __restrict__ value_pages,
    __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value, int tokens) {
    __shared__ float warp_squares[8];
    __shared__ __nv_bfloat16 normalized[kHeadDim];
    const int dim   = static_cast<int>(threadIdx.x);
    const int head  = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int warp  = dim >> 5;
    const int lane  = dim & 31;
    const std::int64_t projected_base = static_cast<std::int64_t>(token) * kProjectedRows;
    const float x =
        __bfloat162float(projected[projected_base + kMainKeyOffset + head * kHeadDim + dim]);
    const float square = ops::warp_reduce_sum(x * x);
    if (lane == 0) { warp_squares[warp] = square; }
    const auto value_word = projected[projected_base + kMainValueOffset + head * kHeadDim + dim];
    value[static_cast<std::int64_t>(token) * kKvHeads * kHeadDim + head * kHeadDim + dim] =
        value_word;
    __syncthreads();
    float sum = 0.0F;
    for (float entry : warp_squares) { sum += entry; }
    normalized[dim] = __float2bfloat16_rn(x * rsqrtf(sum / static_cast<float>(kHeadDim) + 1.0e-6F) *
                                          (1.0F + __bfloat162float(norm[dim])));
    __syncthreads();
    std::int32_t local_positions[3] = {positions[token], positions[tokens + token],
                                       positions[2 * tokens + token]};
    auto* key_destination =
        key + static_cast<std::int64_t>(token) * kKvHeads * kHeadDim + head * kHeadDim;
    store_mrope(normalized, local_positions, key_destination, dim);
    __syncthreads();

    const int token_idx     = token_indices[token];
    const int logical_page  = token_idx / kPageTokens;
    const int page_offset   = token_idx % kPageTokens;
    const int physical_page = block_tables[table_row * logical_pages + logical_page];
    const std::int64_t page_index =
        ((static_cast<std::int64_t>(physical_page) * kKvHeads + head) * kPageTokens + page_offset) *
            kHeadDim +
        dim;
    key_pages[page_index]   = key_destination[dim];
    value_pages[page_index] = value_word;
}

__global__ void qsa_prefill_sparse_attention_kernel(
    const __nv_bfloat16* __restrict__ query, const std::int32_t* __restrict__ token_indices,
    int table_row, const std::int32_t* __restrict__ selected_blocks,
    const std::int32_t* __restrict__ selected_counts, const std::int32_t* __restrict__ block_tables,
    int logical_pages, const __nv_bfloat16* __restrict__ key_pages,
    const __nv_bfloat16* __restrict__ value_pages, __nv_bfloat16* __restrict__ output) {
    __shared__ float scores[256];
    __shared__ float reduction[256];
    const int dim             = static_cast<int>(threadIdx.x);
    const int head            = static_cast<int>(blockIdx.x);
    const int token           = static_cast<int>(blockIdx.y);
    const int kv_head         = head / 12;
    const int token_index     = token_indices[token];
    const int complete_blocks = (token_index + 1) / 4;
    const int tail_count      = (token_index + 1) & 3;
    const int selected_count  = selected_counts[token];
    const int total           = selected_count * 4 + tail_count;
    const auto* selected = selected_blocks + static_cast<std::int64_t>(token) * kSelectedBlocks;
    const auto* q =
        query + static_cast<std::int64_t>(token) * kQueryHeads * kHeadDim + head * kHeadDim;
    float accumulator = 0.0F;
    float running_max = -__int_as_float(0x7F800000);
    float running_sum = 0.0F;

    for (int start = 0; start < total; start += 256) {
        const int count = min(256, total - start);
        float score     = -__int_as_float(0x7F800000);
        if (dim < count) {
            const int cand_token =
                selected_token(start + dim, selected_count, complete_blocks, selected);
            const int logical_page = cand_token / kPageTokens;
            const int page_offset  = cand_token % kPageTokens;
            const int physical_page =
                block_tables[table_row * logical_pages + logical_page];
            const auto* cache_key =
                key_pages +
                ((static_cast<std::int64_t>(physical_page) * kKvHeads + kv_head) * kPageTokens +
                 page_offset) *
                    kHeadDim;
            score = 0.0F;
            for (int feature = 0; feature < kHeadDim; ++feature) {
                score =
                    fmaf(__bfloat162float(q[feature]), __bfloat162float(cache_key[feature]), score);
            }
            score *= kScale;
        }
        scores[dim]    = score;
        reduction[dim] = score;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (dim < stride) { reduction[dim] = fmaxf(reduction[dim], reduction[dim + stride]); }
            __syncthreads();
        }
        const float chunk_max = reduction[0];
        // Every thread must have read reduction[0] before thread 0 overwrites it below; without
        // this barrier a late warp reads a probability instead of the max (intermittent, one
        // (head, token) output per hit).
        __syncthreads();
        const float probability = dim < count ? expf(scores[dim] - chunk_max) : 0.0F;
        scores[dim]             = probability;
        reduction[dim]          = probability;
        __syncthreads();
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (dim < stride) { reduction[dim] += reduction[dim + stride]; }
            __syncthreads();
        }
        const float chunk_sum   = reduction[0];
        const float next_max    = fmaxf(running_max, chunk_max);
        const float prior_scale = running_sum == 0.0F ? 0.0F : expf(running_max - next_max);
        const float chunk_scale = expf(chunk_max - next_max);
        float chunk_value       = 0.0F;
        for (int local = 0; local < count; ++local) {
            const int cand_token =
                selected_token(start + local, selected_count, complete_blocks, selected);
            const int logical_page = cand_token / kPageTokens;
            const int page_offset  = cand_token % kPageTokens;
            const int physical_page =
                block_tables[table_row * logical_pages + logical_page];
            const std::int64_t cache_index =
                ((static_cast<std::int64_t>(physical_page) * kKvHeads + kv_head) * kPageTokens +
                 page_offset) *
                    kHeadDim +
                dim;
            chunk_value =
                fmaf(scores[local], __bfloat162float(value_pages[cache_index]), chunk_value);
        }
        accumulator = accumulator * prior_scale + chunk_value * chunk_scale;
        running_sum = running_sum * prior_scale + chunk_sum * chunk_scale;
        running_max = next_max;
        __syncthreads();
    }
    output[static_cast<std::int64_t>(token) * kQueryHeads * kHeadDim + head * kHeadDim + dim] =
        __float2bfloat16_rn(accumulator / running_sum);
}

void flash_next_qsa_attention_prefill_launch(
    const Tensor& token_indices, const Tensor& mrope_positions, std::int32_t table_row,
    const Tensor& selected_blocks, const Tensor& selected_counts, const Tensor& query_norm,
    const Tensor& key_norm, QsaAttentionCacheView cache, FlashNextQsaAttentionWorkspace& scratch,
    cudaStream_t stream) {
    const int tokens = token_indices.ne[0];
    qsa_prefill_prepare_query_kernel<<<dim3(kQueryHeads, tokens), kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.projected.data),
        static_cast<const __nv_bfloat16*>(query_norm.data),
        static_cast<const std::int32_t*>(mrope_positions.data),
        static_cast<__nv_bfloat16*>(scratch.query.data),
        static_cast<__nv_bfloat16*>(scratch.gate.data), tokens);
    CUDA_CHECK(cudaGetLastError());
    qsa_prefill_prepare_append_kv_kernel<<<dim3(kKvHeads, tokens), kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.projected.data),
        static_cast<const __nv_bfloat16*>(key_norm.data),
        static_cast<const std::int32_t*>(token_indices.data),
        static_cast<const std::int32_t*>(mrope_positions.data),
        table_row,
        static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
        static_cast<__nv_bfloat16*>(cache.key_pages.data),
        static_cast<__nv_bfloat16*>(cache.value_pages.data),
        static_cast<__nv_bfloat16*>(scratch.key.data),
        static_cast<__nv_bfloat16*>(scratch.value.data), tokens);
    CUDA_CHECK(cudaGetLastError());
    qsa_prefill_sparse_attention_kernel<<<dim3(kQueryHeads, tokens), kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.query.data),
        static_cast<const std::int32_t*>(token_indices.data),
        table_row,
        static_cast<const std::int32_t*>(selected_blocks.data),
        static_cast<const std::int32_t*>(selected_counts.data),
        static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
        static_cast<const __nv_bfloat16*>(cache.key_pages.data),
        static_cast<const __nv_bfloat16*>(cache.value_pages.data),
        static_cast<__nv_bfloat16*>(scratch.attended.data));
    CUDA_CHECK(cudaGetLastError());
    const int elements    = kQueryHeads * kHeadDim * tokens;
    constexpr int threads = 256;
    gate_output_kernel<<<(elements + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.attended.data),
        static_cast<const __nv_bfloat16*>(scratch.gate.data),
        static_cast<__nv_bfloat16*>(scratch.gated.data), elements);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
