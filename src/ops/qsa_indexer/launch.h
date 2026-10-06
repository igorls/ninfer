#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

struct QsaTableView {
    const std::int32_t* tables = nullptr; ///< [logical_pages, rows]
    std::int32_t logical_pages = 0;
};

void qsa_append_shared_row_launch(const Tensor& projected, const Tensor& positions,
                                  const Tensor& rope_positions, const Tensor& key_norm,
                                  std::int32_t source_slot, std::int32_t destination_slot,
                                  const Tensor& raw_keys, const Tensor& raw_positions,
                                  QsaTableView table, const Tensor& block_keys,
                                  cudaStream_t stream);

void qsa_append_snapshot_launch(const Tensor& projected, const Tensor& positions,
                                const Tensor& rope_positions, const Tensor& table_rows,
                                const Tensor& initial_slots, const Tensor& snapshot_base_slots,
                        const Tensor& valid_columns,
                                const Tensor& key_norm, const Tensor& raw_keys,
                                const Tensor& raw_positions, QsaTableView table,
                                const Tensor& block_keys, std::int32_t width, std::int32_t batch,
                                cudaStream_t stream);

// Row stride (elements) of the FP32 score tile.
[[nodiscard]] std::int32_t qsa_select_score_stride(std::int32_t max_complete_blocks);

// Columns per scoring tile so that the FP32 score tile stays within the workspace budget.
[[nodiscard]] std::int32_t qsa_select_tile_columns(std::int32_t columns,
                                                   std::int32_t max_complete_blocks);

struct QsaSelectWorkspace {
    void* prepared = nullptr; ///< BF16 [128,4,C]
    float* scores  = nullptr; ///< FP32 [max_complete_blocks, tile]
};

void qsa_select_identity_launch(const Tensor& positions, Tensor& selections, Tensor& counts,
                                cudaStream_t stream);

// table_rows == nullptr selects the shared-row route (single-row table, tensor-core scoring).
void qsa_select_score_launch(const Tensor& projected, const Tensor& positions,
                             const Tensor& rope_positions, const Tensor* table_rows,
                             const Tensor& query_norm, QsaTableView table,
                             const Tensor& block_keys, std::int32_t max_complete_blocks,
                             QsaSelectWorkspace workspace, Tensor& selections, Tensor& counts,
                             cudaStream_t stream);

} // namespace ninfer::ops::detail
