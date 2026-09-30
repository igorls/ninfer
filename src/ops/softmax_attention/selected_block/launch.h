#pragma once

#include "core/tensor.h"
#include "ops/softmax_attention/selected_block/types.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t selected_block_decode_partial_bytes(std::int32_t columns);

void selected_block_decode_launch(SelectedKvProfile profile, const Tensor& q,
                                  const Tensor& positions, const Tensor& table_rows,
                                  const Tensor& selections, const Tensor& counts,
                                  const SelectedKvPlanes& kv, float* partial, Tensor& out,
                                  cudaStream_t stream);

void selected_block_prefill_launch(SelectedKvProfile profile, const Tensor& q,
                                   const Tensor& positions, const Tensor& selections,
                                   const Tensor& counts, const SelectedKvPlanes& kv, Tensor& out,
                                   cudaStream_t stream);

} // namespace ninfer::ops::detail
