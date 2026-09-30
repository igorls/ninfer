#include "ops/softmax_attention/selected_block/launch.h"

#include "core/device.h"
#include "ops/softmax_attention/selected_block/merge.cuh"
#include "ops/softmax_attention/selected_block/tiled.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

// Batched route partitions: enough CTAs to cover the device at B=1 (a visible set holds at most
// 2051 tokens, so 32 partitions leave about one 64-token tile each) and two tiles per partition
// from four columns up.
int decode_partitions(int columns) { return columns < 4 ? kSelectedMaxPartitions : 16; }

template <SelectedKvProfile Profile>
void launch_decode(const Tensor& q, const Tensor& positions, const Tensor& table_rows,
                   const Tensor& selections, const Tensor& counts, const SelectedKvPlanes& kv,
                   float* partial, Tensor& out, cudaStream_t stream) {
    const int columns    = q.ne[2];
    const int partitions = decode_partitions(columns);
    selected_block_tile_kernel<Profile, true>
        <<<dim3(static_cast<unsigned>(columns) * kSelectedKvHeads, partitions),
           kSelectedPrefillThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int32_t*>(positions.data),
            static_cast<const std::int32_t*>(table_rows.data),
            static_cast<const std::int32_t*>(selections.data),
            static_cast<const std::int32_t*>(counts.data), kv, nullptr, partial);
    CUDA_CHECK(cudaGetLastError());
    selected_block_decode_merge_kernel<<<dim3(kSelectedQueryHeads, columns), kSelectedHeadDim, 0,
                                         stream>>>(partial, static_cast<__nv_bfloat16*>(out.data),
                                                   partitions);
    CUDA_CHECK(cudaGetLastError());
}

template <SelectedKvProfile Profile>
void launch_prefill(const Tensor& q, const Tensor& positions, const Tensor& selections,
                    const Tensor& counts, const SelectedKvPlanes& kv, Tensor& out,
                    cudaStream_t stream) {
    const unsigned blocks = static_cast<unsigned>(q.ne[2]) * kSelectedKvHeads;
    selected_block_tile_kernel<Profile, false><<<blocks, kSelectedPrefillThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const std::int32_t*>(positions.data),
        nullptr, static_cast<const std::int32_t*>(selections.data),
        static_cast<const std::int32_t*>(counts.data), kv, static_cast<__nv_bfloat16*>(out.data),
        nullptr);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::size_t selected_block_decode_partial_bytes(std::int32_t columns) {
    return static_cast<std::size_t>(columns) * kSelectedQueryHeads * decode_partitions(columns) *
           kSelectedPartialStride * sizeof(float);
}

void selected_block_decode_launch(SelectedKvProfile profile, const Tensor& q,
                                  const Tensor& positions, const Tensor& table_rows,
                                  const Tensor& selections, const Tensor& counts,
                                  const SelectedKvPlanes& kv, float* partial, Tensor& out,
                                  cudaStream_t stream) {
    if (profile == SelectedKvProfile::Bf16KFp16V) {
        launch_decode<SelectedKvProfile::Bf16KFp16V>(q, positions, table_rows, selections, counts,
                                                     kv, partial, out, stream);
    } else {
        launch_decode<SelectedKvProfile::Fp8Row256>(q, positions, table_rows, selections, counts,
                                                    kv, partial, out, stream);
    }
}

void selected_block_prefill_launch(SelectedKvProfile profile, const Tensor& q,
                                   const Tensor& positions, const Tensor& selections,
                                   const Tensor& counts, const SelectedKvPlanes& kv, Tensor& out,
                                   cudaStream_t stream) {
    if (profile == SelectedKvProfile::Bf16KFp16V) {
        launch_prefill<SelectedKvProfile::Bf16KFp16V>(q, positions, selections, counts, kv, out,
                                                      stream);
    } else {
        launch_prefill<SelectedKvProfile::Fp8Row256>(q, positions, selections, counts, kv, out,
                                                     stream);
    }
}

} // namespace ninfer::ops::detail
