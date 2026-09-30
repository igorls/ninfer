#include "ninfer/ops/qsa_indexer.h"

#include "core/layout.h"
#include "ops/qsa_indexer/launch.h"

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim       = 128;
constexpr std::int32_t kProjection    = 640;
constexpr std::int32_t kBlocksPerPage = 16;
constexpr std::int32_t kMaxSelected   = 512;
constexpr std::int32_t kMaxWidth      = 16;
constexpr std::int32_t kMaxBatch      = 8;
constexpr std::int32_t kMaxRowColumns = 128;

[[noreturn]] void fail(const char* op, const std::string& message) {
    throw std::invalid_argument(std::string(op) + ": " + message);
}

bool aligned16(const void* pointer) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & 15U) == 0;
}

void require(const char* op, const Tensor& tensor, DType dtype,
             std::initializer_list<std::int32_t> shape, const char* label) {
    std::int32_t expected[4] = {1, 1, 1, 1};
    int index                = 0;
    for (const std::int32_t extent : shape) { expected[index++] = extent; }
    for (int dim = 0; dim < 4; ++dim) {
        if (tensor.ne[dim] != expected[dim]) { fail(op, std::string("invalid shape for ") + label); }
    }
    if (tensor.dtype != dtype || !tensor.is_contiguous() || !aligned16(tensor.data)) {
        fail(op, std::string("invalid dtype, layout or alignment for ") + label);
    }
}

bool overlaps(const Tensor& first, const Tensor& second) {
    if (first.data == nullptr || second.data == nullptr) { return false; }
    const auto a = reinterpret_cast<std::uintptr_t>(first.data);
    const auto b = reinterpret_cast<std::uintptr_t>(second.data);
    return a < b + second.bytes() && b < a + first.bytes();
}

void require_disjoint(const char* op, std::initializer_list<const Tensor*> mutated,
                      std::initializer_list<const Tensor*> others) {
    for (const Tensor* m : mutated) {
        for (const Tensor* o : others) {
            if (m != o && overlaps(*m, *o)) { fail(op, "a mutated tensor overlaps another operand"); }
        }
    }
}

detail::QsaTableView require_blocks(const char* op, const QsaIndexerBlockKeys& blocks,
                                    bool single_row) {
    const std::int32_t pages = blocks.keys.ne[2];
    if (pages <= 0) { fail(op, "empty block-key plane"); }
    require(op, blocks.keys, DType::BF16, {kHeadDim, kBlocksPerPage, pages}, "block keys");
    const Tensor& tables = blocks.block_tables;
    if (tables.dtype != DType::I32 || tables.ne[0] <= 0 || tables.ne[1] <= 0 ||
        tables.ne[2] != 1 || tables.ne[3] != 1 || !tables.is_contiguous() ||
        tables.data == nullptr || (single_row && tables.ne[1] != 1)) {
        fail(op, "invalid block tables");
    }
    return {static_cast<const std::int32_t*>(tables.data), tables.ne[0]};
}

void require_state(const char* op, const QsaIndexerKeyState& state) {
    const std::int32_t slots = state.raw_keys.ne[2];
    if (slots <= 0) { fail(op, "empty key state"); }
    require(op, state.raw_keys, DType::BF16, {kHeadDim, 4, slots}, "raw keys");
    require(op, state.raw_positions, DType::I32, {3, 4, slots}, "raw positions");
}

constexpr const char* kAppend = "qsa_indexer_append";
constexpr const char* kSelect = "qsa_indexer_select";

template <class Arena>
detail::QsaSelectWorkspace allocate_select(Arena& arena, std::int32_t columns,
                                           std::int32_t max_complete_blocks) {
    detail::QsaSelectWorkspace workspace;
    workspace.prepared =
        arena.alloc_bytes(static_cast<std::size_t>(columns) * 4 * kHeadDim * 2, 256).data;
    const std::int32_t tile = detail::qsa_select_tile_columns(columns, max_complete_blocks);
    workspace.scores        = static_cast<float*>(
        arena.alloc_bytes(static_cast<std::size_t>(tile) *
                              detail::qsa_select_score_stride(max_complete_blocks) *
                              sizeof(float),
                          256)
            .data);
    return workspace;
}

void select_common(const Tensor& projected, const Tensor& positions, const Tensor& rope_positions,
                   const Tensor* table_rows, const Tensor& query_norm,
                   const QsaIndexerBlockKeys& blocks, QsaIndexerSelectEnvelope envelope,
                   WorkspaceArena& workspace, Tensor& selections, Tensor& counts,
                   cudaStream_t stream) {
    const std::int32_t columns = projected.ne[1];
    if (columns < 1 || (table_rows != nullptr && columns > kMaxRowColumns)) {
        fail(kSelect, "invalid column count");
    }
    if (envelope.max_complete_blocks < 0) { fail(kSelect, "invalid envelope"); }
    require(kSelect, projected, DType::BF16, {kProjection, columns}, "projection");
    require(kSelect, positions, DType::I32, {columns}, "positions");
    require(kSelect, rope_positions, DType::I32, {columns, 3}, "rope positions");
    require(kSelect, query_norm, DType::BF16, {kHeadDim}, "query norm");
    require(kSelect, selections, DType::I32, {kMaxSelected, columns}, "selections");
    require(kSelect, counts, DType::I32, {columns}, "counts");
    if (table_rows != nullptr) { require(kSelect, *table_rows, DType::I32, {columns}, "table rows"); }
    const detail::QsaTableView table = require_blocks(kSelect, blocks, table_rows == nullptr);
    const Tensor empty{};
    require_disjoint(kSelect, {&selections, &counts},
                     {&selections, &counts, &projected, &positions, &rope_positions,
                      table_rows != nullptr ? table_rows : &empty, &query_norm, &blocks.keys,
                      &blocks.block_tables});
    if (static_cast<std::int64_t>(envelope.max_complete_blocks) >
        static_cast<std::int64_t>(table.logical_pages) * kBlocksPerPage) {
        fail(kSelect, "envelope exceeds the block-table capacity");
    }
    if (envelope.max_complete_blocks <= kMaxSelected) {
        detail::qsa_select_identity_launch(positions, selections, counts, stream);
        return;
    }
    const auto scope = workspace.scope();
    const auto scratch = allocate_select(workspace, columns, envelope.max_complete_blocks);
    detail::qsa_select_score_launch(projected, positions, rope_positions, table_rows, query_norm,
                                    table, blocks.keys, envelope.max_complete_blocks, scratch,
                                    selections, counts, stream);
}

} // namespace

void qsa_indexer_append(const Tensor& projected, const Tensor& positions,
                        const Tensor& rope_positions, const Tensor& key_norm,
                        std::int32_t source_slot, std::int32_t destination_slot,
                        const QsaIndexerKeyState& state, const QsaIndexerBlockKeys& blocks,
                        cudaStream_t stream) {
    const std::int32_t tokens = projected.ne[1];
    if (tokens < 1) { fail(kAppend, "T must be positive"); }
    require(kAppend, projected, DType::BF16, {kProjection, tokens}, "projection");
    require(kAppend, positions, DType::I32, {tokens}, "positions");
    require(kAppend, rope_positions, DType::I32, {tokens, 3}, "rope positions");
    require(kAppend, key_norm, DType::BF16, {kHeadDim}, "key norm");
    require_state(kAppend, state);
    const detail::QsaTableView table = require_blocks(kAppend, blocks, true);
    const std::int32_t slots         = state.raw_keys.ne[2];
    if (source_slot < 0 || source_slot >= slots || destination_slot < 0 ||
        destination_slot >= slots) {
        fail(kAppend, "state slot out of range");
    }
    require_disjoint(kAppend, {&state.raw_keys, &state.raw_positions, &blocks.keys},
                     {&state.raw_keys, &state.raw_positions, &blocks.keys, &projected, &positions,
                      &rope_positions, &key_norm, &blocks.block_tables});
    detail::qsa_append_shared_row_launch(projected, positions, rope_positions, key_norm,
                                         source_slot, destination_slot, state.raw_keys,
                                         state.raw_positions, table, blocks.keys, stream);
}

void qsa_indexer_append(const Tensor& projected, const Tensor& positions,
                        const Tensor& rope_positions, const Tensor& table_rows,
                        const Tensor& initial_slots, const Tensor& snapshot_base_slots,
                        const Tensor& key_norm, const QsaIndexerKeyState& state,
                        const QsaIndexerBlockKeys& blocks, cudaStream_t stream) {
    const std::int32_t width = projected.ne[1];
    const std::int32_t batch = projected.ne[2];
    if (width < 1 || width > kMaxWidth || batch < 1 || batch > kMaxBatch) {
        fail(kAppend, "speculative form requires W in [1,16] and B in [1,8]");
    }
    require(kAppend, projected, DType::BF16, {kProjection, width, batch}, "projection");
    require(kAppend, positions, DType::I32, {width, batch}, "positions");
    require(kAppend, rope_positions, DType::I32, {width * batch, 3}, "rope positions");
    require(kAppend, table_rows, DType::I32, {batch}, "table rows");
    require(kAppend, initial_slots, DType::I32, {batch}, "initial slots");
    require(kAppend, snapshot_base_slots, DType::I32, {batch}, "snapshot base slots");
    require(kAppend, key_norm, DType::BF16, {kHeadDim}, "key norm");
    require_state(kAppend, state);
    const detail::QsaTableView table = require_blocks(kAppend, blocks, false);
    require_disjoint(kAppend, {&state.raw_keys, &state.raw_positions, &blocks.keys},
                     {&state.raw_keys, &state.raw_positions, &blocks.keys, &projected, &positions,
                      &rope_positions, &table_rows, &initial_slots, &snapshot_base_slots,
                      &key_norm, &blocks.block_tables});
    detail::qsa_append_snapshot_launch(projected, positions, rope_positions, table_rows,
                                       initial_slots, snapshot_base_slots, key_norm, state.raw_keys,
                                       state.raw_positions, table, blocks.keys, width, batch,
                                       stream);
}

std::size_t qsa_indexer_select_workspace_capacity_bytes(std::int32_t min_columns,
                                                        std::int32_t max_columns,
                                                        QsaIndexerSelectEnvelope envelope) {
    if (min_columns < 1 || max_columns < min_columns || envelope.max_complete_blocks < 0) {
        fail(kSelect, "invalid column interval or envelope");
    }
    if (envelope.max_complete_blocks <= kMaxSelected) { return 0; }
    WorkspaceLayoutBuilder layout;
    (void)allocate_select(layout, max_columns, envelope.max_complete_blocks);
    return layout.peak_bytes(1);
}

void qsa_indexer_select(const Tensor& projected, const Tensor& positions,
                        const Tensor& rope_positions, const Tensor& table_rows,
                        const Tensor& query_norm, const QsaIndexerBlockKeys& blocks,
                        QsaIndexerSelectEnvelope envelope, WorkspaceArena& workspace,
                        Tensor& selections, Tensor& counts, cudaStream_t stream) {
    select_common(projected, positions, rope_positions, &table_rows, query_norm, blocks, envelope,
                  workspace, selections, counts, stream);
}

void qsa_indexer_select(const Tensor& projected, const Tensor& positions,
                        const Tensor& rope_positions, const Tensor& query_norm,
                        const QsaIndexerBlockKeys& blocks, QsaIndexerSelectEnvelope envelope,
                        WorkspaceArena& workspace, Tensor& selections, Tensor& counts,
                        cudaStream_t stream) {
    select_common(projected, positions, rope_positions, nullptr, query_norm, blocks, envelope,
                  workspace, selections, counts, stream);
}

} // namespace ninfer::ops
