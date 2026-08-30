#include "targets/qwen3_8_flash_next/impl/text_decode_kernels.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

namespace {

__global__ void repeat_embedding_kernel(const __nv_bfloat16* __restrict__ embedding,
                                        __nv_bfloat16* __restrict__ hyper_hidden, int batch) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (d >= 2'560 || b >= batch) return;
    const auto val                 = embedding[static_cast<std::int64_t>(b) * 2'560 + d];
    const std::int64_t base        = static_cast<std::int64_t>(b) * 10'240 + d;
    hyper_hidden[base + 0 * 2'560] = val;
    hyper_hidden[base + 1 * 2'560] = val;
    hyper_hidden[base + 2 * 2'560] = val;
    hyper_hidden[base + 3 * 2'560] = val;
}

} // namespace

void repeat_embedding_to_hyper_streams(const Tensor& embedding, Tensor& hyper_hidden,
                                       cudaStream_t stream) {
    const int batch = embedding.ne[1];
    dim3 grid((2'560 + 255) / 256, batch);
    repeat_embedding_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(embedding.data),
        static_cast<__nv_bfloat16*>(hyper_hidden.data), batch);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void set_qsa_step_metadata_kernel(
    const std::int32_t* __restrict__ token_indices,
    const std::int32_t* __restrict__ mrope_positions, int t, std::int32_t table_row,
    std::int32_t src_slot, std::int32_t dst_slot, std::int32_t* __restrict__ out_tok_idx,
    std::int32_t* __restrict__ out_mrope_pos, std::int32_t* __restrict__ out_table_row,
    std::int32_t* __restrict__ out_src_slot, std::int32_t* __restrict__ out_dst_slot) {
    if (threadIdx.x == 0) {
        out_tok_idx[0]   = token_indices[t];
        out_mrope_pos[0] = mrope_positions[t * 3 + 0];
        out_mrope_pos[1] = mrope_positions[t * 3 + 1];
        out_mrope_pos[2] = mrope_positions[t * 3 + 2];
        out_table_row[0] = table_row;
        out_src_slot[0]  = src_slot;
        out_dst_slot[0]  = dst_slot;
    }
}

void set_qsa_step_metadata(const Tensor& token_indices, const Tensor& mrope_positions,
                           std::int32_t t, std::int32_t table_row, std::int32_t src_slot,
                           std::int32_t dst_slot, Tensor& out_tok_idx, Tensor& out_mrope_pos,
                           Tensor& out_table_row, Tensor& out_src_slot, Tensor& out_dst_slot,
                           cudaStream_t stream) {
    set_qsa_step_metadata_kernel<<<1, 1, 0, stream>>>(
        static_cast<const std::int32_t*>(token_indices.data),
        static_cast<const std::int32_t*>(mrope_positions.data), t, table_row, src_slot, dst_slot,
        static_cast<std::int32_t*>(out_tok_idx.data),
        static_cast<std::int32_t*>(out_mrope_pos.data),
        static_cast<std::int32_t*>(out_table_row.data),
        static_cast<std::int32_t*>(out_src_slot.data),
        static_cast<std::int32_t*>(out_dst_slot.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
