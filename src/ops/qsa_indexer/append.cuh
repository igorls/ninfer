#pragma once

#include "ops/qsa_indexer/common.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Publishes one completed block from its four raw keys (one dimension per thread) and the MRoPE
// positions of its first token. Every thread of the 128-thread CTA must call it.
__device__ __forceinline__ void qsa_publish_block(float k0, float k1, float k2, float k3,
                                                  const std::int32_t* first_positions,
                                                  float key_norm, __nv_bfloat16* destination,
                                                  float2* rotation, float* scratch) {
    const int dim = static_cast<int>(threadIdx.x);
    if (dim < kTextMropeR64Pairs) {
        rotation[dim] = text_mrope_r64_sincos(dim, first_positions[dim % 3]);
    }
    __syncthreads(); // rotation is complete before qsa_norm_rotate reads it
    // The block mean is a BF16 tensor in the checkpoint: FP32 mean, one BF16 rounding.
    const float mean = __bfloat162float(__float2bfloat16_rn((k0 + k1 + k2 + k3) * 0.25F));
    const float out  = qsa_norm_rotate(mean, key_norm, rotation, scratch);
    destination[dim] = __float2bfloat16_rn(out);
}

// Shared-row append, stage 1: every block completed by the chunk. CTA b owns chunk block b.
__global__ __launch_bounds__(kQsaHeadDim) void qsa_append_publish_kernel(
    const __nv_bfloat16* __restrict__ projected, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ rope_positions, const __nv_bfloat16* __restrict__ key_norm,
    int source_slot, const __nv_bfloat16* __restrict__ raw_keys,
    const std::int32_t* __restrict__ raw_positions, const std::int32_t* __restrict__ table,
    __nv_bfloat16* __restrict__ block_keys, int tokens) {
    __shared__ float2 rotation[kTextMropeR64Pairs];
    __shared__ float scratch[kQsaHeadDim + 4];
    __shared__ std::int32_t first_positions[3];
    const int dim      = static_cast<int>(threadIdx.x);
    const int block    = static_cast<int>(blockIdx.x);
    const int first    = positions[0];
    const int leftover = first & 3;
    if (block >= (leftover + tokens) / 4) { return; } // uniform per CTA
    // Chunk-local ordinal s covers the source slot's leftover entries, then the chunk tokens.
    const auto raw_value = [&](int s) {
        if (s < leftover) {
            return __bfloat162float(
                raw_keys[(static_cast<std::int64_t>(source_slot) * 4 + s) * kQsaHeadDim + dim]);
        }
        return __bfloat162float(
            projected[static_cast<std::int64_t>(s - leftover) * kQsaProjection + kQsaRawKeyOffset +
                      dim]);
    };
    if (dim < 3) {
        const int s          = 4 * block;
        first_positions[dim] = s < leftover
                                   ? raw_positions[(static_cast<std::int64_t>(source_slot) * 4 + s) *
                                                       3 +
                                                   dim]
                                   : rope_positions[static_cast<std::int64_t>(dim) * tokens + s -
                                                    leftover];
    }
    const float k0 = raw_value(4 * block);
    const float k1 = raw_value(4 * block + 1);
    const float k2 = raw_value(4 * block + 2);
    const float k3 = raw_value(4 * block + 3);
    __syncthreads(); // first_positions
    const int global_block = (first - leftover) / 4 + block;
    qsa_publish_block(k0, k1, k2, k3, first_positions, __bfloat162float(key_norm[dim]),
                      block_keys + qsa_block_offset(table, global_block), rotation, scratch);
}

// Shared-row append, stage 2: the forming block after the chunk into the destination slot. It
// runs after stage 1 on the same stream, so a destination equal to the source is safe: an entry
// is read from the source only when no block completed, and then by the thread that writes it.
__global__ __launch_bounds__(kQsaHeadDim) void qsa_append_leftover_kernel(
    const __nv_bfloat16* __restrict__ projected, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ rope_positions, int source_slot, int destination_slot,
    __nv_bfloat16* raw_keys, std::int32_t* raw_positions, int tokens) {
    const int dim      = static_cast<int>(threadIdx.x);
    const int leftover = positions[0] & 3;
    const int complete = (leftover + tokens) / 4;
    const int out      = (leftover + tokens) & 3;
    for (int slot = 0; slot < out; ++slot) {
        const int s          = complete * 4 + slot;
        const std::int64_t d = static_cast<std::int64_t>(destination_slot) * 4 + slot;
        if (s < leftover) {
            const std::int64_t source = static_cast<std::int64_t>(source_slot) * 4 + s;
            raw_keys[d * kQsaHeadDim + dim] = raw_keys[source * kQsaHeadDim + dim];
            if (dim < 3) { raw_positions[d * 3 + dim] = raw_positions[source * 3 + dim]; }
        } else {
            const int t = s - leftover;
            raw_keys[d * kQsaHeadDim + dim] =
                projected[static_cast<std::int64_t>(t) * kQsaProjection + kQsaRawKeyOffset + dim];
            if (dim < 3) {
                raw_positions[d * 3 + dim] = rope_positions[static_cast<std::int64_t>(dim) * tokens + t];
            }
        }
    }
}

// Speculative/decode append: CTA b walks row b's W columns in order from its initial slot and
// writes a snapshot after every column.
__global__ __launch_bounds__(kQsaHeadDim) void qsa_append_snapshot_kernel(
    const __nv_bfloat16* __restrict__ projected, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ rope_positions, const std::int32_t* __restrict__ table_rows,
    const std::int32_t* __restrict__ initial_slots,
    const std::int32_t* __restrict__ snapshot_base_slots,
    const std::int32_t* __restrict__ valid_columns,
    const __nv_bfloat16* __restrict__ key_norm, __nv_bfloat16* raw_keys,
    std::int32_t* raw_positions, const std::int32_t* __restrict__ tables, int logical_pages,
    __nv_bfloat16* __restrict__ block_keys, int width, int batch) {
    __shared__ float forming[4][kQsaHeadDim];
    __shared__ std::int32_t forming_positions[4][3];
    __shared__ float2 rotation[kTextMropeR64Pairs];
    __shared__ float scratch[kQsaHeadDim + 4];
    const int dim     = static_cast<int>(threadIdx.x);
    const int row     = static_cast<int>(blockIdx.x);
    const int columns = width * batch;
    const std::int64_t initial = initial_slots[row];
    const int base             = snapshot_base_slots[row];
    const auto* table          = tables + static_cast<std::int64_t>(table_rows[row]) * logical_pages;
    const float weight         = __bfloat162float(key_norm[dim]);
    // The complete initial state is read before any snapshot write (the initial slot may lie in
    // this row's snapshot interval).
    for (int r = 0; r < 4; ++r) {
        forming[r][dim] = __bfloat162float(raw_keys[(initial * 4 + r) * kQsaHeadDim + dim]);
    }
    if (dim < 12) { forming_positions[dim / 3][dim % 3] = raw_positions[initial * 12 + dim]; }
    __syncthreads();
    for (int w = 0; w < (valid_columns ? valid_columns[row] : width); ++w) {
        const int column = w + width * row;
        const int entry  = positions[column] & 3;
        forming[entry][dim] = __bfloat162float(
            projected[static_cast<std::int64_t>(column) * kQsaProjection + kQsaRawKeyOffset + dim]);
        if (dim < 3) {
            forming_positions[entry][dim] = rope_positions[static_cast<std::int64_t>(dim) * columns + column];
        }
        __syncthreads(); // the updated forming block is visible to every thread
        const std::int64_t snapshot = static_cast<std::int64_t>(base) + w;
        for (int r = 0; r < 4; ++r) {
            raw_keys[(snapshot * 4 + r) * kQsaHeadDim + dim] = __float2bfloat16_rn(forming[r][dim]);
        }
        if (dim < 12) { raw_positions[snapshot * 12 + dim] = forming_positions[dim / 3][dim % 3]; }
        if (entry == 3) { // uniform per CTA
            qsa_publish_block(forming[0][dim], forming[1][dim], forming[2][dim], forming[3][dim],
                              forming_positions[0], weight,
                              block_keys + qsa_block_offset(table, positions[column] / 4), rotation,
                              scratch);
        }
        __syncthreads(); // the next column overwrites a forming entry
    }
}

} // namespace ninfer::ops::detail
