#include "ops/qsa_indexer/launch.h"

#include "core/device.h"
#include "ops/qsa_indexer/append.cuh"
#include "ops/qsa_indexer/select.cuh"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr std::size_t kScoreBudgetBytes = 64ULL << 20;

template <class T>
const T* as(const Tensor& tensor) {
    return static_cast<const T*>(tensor.data);
}

template <class T>
T* as_mutable(const Tensor& tensor) {
    return static_cast<T*>(tensor.data);
}

} // namespace

void qsa_append_shared_row_launch(const Tensor& projected, const Tensor& positions,
                                  const Tensor& rope_positions, const Tensor& key_norm,
                                  std::int32_t source_slot, std::int32_t destination_slot,
                                  const Tensor& raw_keys, const Tensor& raw_positions,
                                  QsaTableView table, const Tensor& block_keys,
                                  cudaStream_t stream) {
    const int tokens = projected.ne[1];
    // Upper bound of completed blocks over every leftover in [0,3]; excess CTAs exit.
    const int blocks = (tokens + 3) / 4;
    if (blocks > 0) {
        qsa_append_publish_kernel<<<blocks, kQsaHeadDim, 0, stream>>>(
            as<__nv_bfloat16>(projected), as<std::int32_t>(positions),
            as<std::int32_t>(rope_positions), as<__nv_bfloat16>(key_norm), source_slot,
            as<__nv_bfloat16>(raw_keys), as<std::int32_t>(raw_positions), table.tables,
            as_mutable<__nv_bfloat16>(block_keys), tokens);
        CUDA_CHECK(cudaGetLastError());
    }
    qsa_append_leftover_kernel<<<1, kQsaHeadDim, 0, stream>>>(
        as<__nv_bfloat16>(projected), as<std::int32_t>(positions), as<std::int32_t>(rope_positions),
        source_slot, destination_slot, as_mutable<__nv_bfloat16>(raw_keys),
        as_mutable<std::int32_t>(raw_positions), tokens);
    CUDA_CHECK(cudaGetLastError());
}

void qsa_append_snapshot_launch(const Tensor& projected, const Tensor& positions,
                                const Tensor& rope_positions, const Tensor& table_rows,
                                const Tensor& initial_slots, const Tensor& snapshot_base_slots,
                        const Tensor& valid_columns,
                                const Tensor& key_norm, const Tensor& raw_keys,
                                const Tensor& raw_positions, QsaTableView table,
                                const Tensor& block_keys, std::int32_t width, std::int32_t batch,
                                cudaStream_t stream) {
    qsa_append_snapshot_kernel<<<batch, kQsaHeadDim, 0, stream>>>(
        as<__nv_bfloat16>(projected), as<std::int32_t>(positions), as<std::int32_t>(rope_positions),
        as<std::int32_t>(table_rows), as<std::int32_t>(initial_slots),
        as<std::int32_t>(snapshot_base_slots), as<std::int32_t>(valid_columns), as<__nv_bfloat16>(key_norm),
        as_mutable<__nv_bfloat16>(raw_keys), as_mutable<std::int32_t>(raw_positions), table.tables,
        table.logical_pages, as_mutable<__nv_bfloat16>(block_keys), width, batch);
    CUDA_CHECK(cudaGetLastError());
}

std::int32_t qsa_select_score_stride(std::int32_t max_complete_blocks) {
    return (max_complete_blocks + 3) / 4 * 4; // 16-byte score rows
}

std::int32_t qsa_select_tile_columns(std::int32_t columns, std::int32_t max_complete_blocks) {
    const std::size_t per_column =
        static_cast<std::size_t>(std::max(qsa_select_score_stride(max_complete_blocks), 4)) * 4U;
    const auto budget            = static_cast<std::int64_t>(kScoreBudgetBytes / per_column);
    return static_cast<std::int32_t>(std::clamp<std::int64_t>(budget, 1, columns));
}

void qsa_select_identity_launch(const Tensor& positions, Tensor& selections, Tensor& counts,
                                cudaStream_t stream) {
    qsa_select_kernel<<<positions.ne[0], kQsaSelectThreads, 0, stream>>>(
        nullptr, 0, as<std::int32_t>(positions), as_mutable<std::int32_t>(selections),
        as_mutable<std::int32_t>(counts), false);
    CUDA_CHECK(cudaGetLastError());
}

void qsa_select_score_launch(const Tensor& projected, const Tensor& positions,
                             const Tensor& rope_positions, const Tensor* table_rows,
                             const Tensor& query_norm, QsaTableView table,
                             const Tensor& block_keys, std::int32_t max_complete_blocks,
                             QsaSelectWorkspace workspace, Tensor& selections, Tensor& counts,
                             cudaStream_t stream) {
    const int columns = positions.ne[0];
    auto* prepared    = static_cast<__nv_bfloat16*>(workspace.prepared);
    qsa_prepare_query_kernel<<<columns, kQsaHeadDim, 0, stream>>>(
        as<__nv_bfloat16>(projected), as<std::int32_t>(rope_positions),
        as<__nv_bfloat16>(query_norm), prepared, columns);
    CUDA_CHECK(cudaGetLastError());
    const int tile     = qsa_select_tile_columns(columns, max_complete_blocks);
    const int stride   = qsa_select_score_stride(max_complete_blocks);
    const auto* pos    = as<std::int32_t>(positions);
    auto* selected     = as_mutable<std::int32_t>(selections);
    auto* count_out    = as_mutable<std::int32_t>(counts);
    for (int start = 0; start < columns; start += tile) {
        const int width = std::min(tile, columns - start);
        if (table_rows != nullptr) {
            const dim3 grid((max_complete_blocks + kQsaScoreBlocksPerCta - 1) / kQsaScoreBlocksPerCta,
                            width);
            qsa_score_rows_kernel<<<grid, kQsaScoreThreads, 0, stream>>>(
                prepared + static_cast<std::int64_t>(start) * kQsaQueryHeads * kQsaHeadDim,
                pos + start, as<std::int32_t>(*table_rows) + start, table.tables,
                table.logical_pages, as<__nv_bfloat16>(block_keys), workspace.scores, stride);
        } else {
            const dim3 grid((max_complete_blocks + kQsaMmaBlocks - 1) / kQsaMmaBlocks,
                            (width + kQsaMmaColumns - 1) / kQsaMmaColumns);
            qsa_score_shared_row_kernel<<<grid, kQsaMmaThreads, 0, stream>>>(
                prepared + static_cast<std::int64_t>(start) * kQsaQueryHeads * kQsaHeadDim,
                pos + start, table.tables, as<__nv_bfloat16>(block_keys), workspace.scores, stride,
                width);
        }
        CUDA_CHECK(cudaGetLastError());
        qsa_select_kernel<<<width, kQsaSelectThreads, 0, stream>>>(
            workspace.scores, stride, pos + start,
            selected + static_cast<std::int64_t>(start) * kQsaMaxSelected, count_out + start, true);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
