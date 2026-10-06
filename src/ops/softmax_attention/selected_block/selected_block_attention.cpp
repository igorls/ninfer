#include "ninfer/ops/selected_block_attention.h"

#include "core/layout.h"
#include "ops/softmax_attention/selected_block/launch.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr const char* kOp              = "selected_block_attention";
constexpr std::int32_t kHeadDim        = 256;
constexpr std::int32_t kQueryHeads     = 24;
constexpr std::int32_t kKvHeads        = 2;
constexpr std::int32_t kMaxSelected    = 512;
constexpr std::int32_t kMaxColumns     = 48;
constexpr std::int32_t kMaxSharedRow   = 262'144;

[[noreturn]] void fail(const std::string& message) {
    throw std::invalid_argument(std::string(kOp) + ": " + message);
}

bool aligned16(const void* pointer) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & 15U) == 0;
}

void require(const Tensor& tensor, DType dtype, std::initializer_list<std::int32_t> shape,
             const char* label) {
    std::int32_t expected[4] = {1, 1, 1, 1};
    int index                = 0;
    for (const std::int32_t extent : shape) { expected[index++] = extent; }
    for (int dim = 0; dim < 4; ++dim) {
        if (tensor.ne[dim] != expected[dim]) { fail(std::string("invalid shape for ") + label); }
    }
    if (tensor.dtype != dtype || !tensor.is_contiguous() || !aligned16(tensor.data)) {
        fail(std::string("invalid dtype, layout or alignment for ") + label);
    }
}

bool overlaps(const Tensor& first, const Tensor& second) {
    if (first.data == nullptr || second.data == nullptr) { return false; }
    const auto a = reinterpret_cast<std::uintptr_t>(first.data);
    const auto b = reinterpret_cast<std::uintptr_t>(second.data);
    return a < b + second.bytes() && b < a + first.bytes();
}

struct ResolvedCache {
    detail::SelectedKvProfile profile;
    detail::SelectedKvPlanes planes;
};

ResolvedCache resolve_cache(const Tensor& k_pages, const Tensor& v_pages,
                            const Tensor& k_scale_pages, const Tensor& v_scale_pages,
                            const Tensor& tables, std::int32_t head_dim,
                            std::int32_t num_kv_heads, KvCacheStorage storage) {
    if (head_dim != kHeadDim || num_kv_heads != kKvHeads) { fail("unsupported cache geometry"); }
    const std::int32_t pages = k_pages.ne[3];
    if (pages <= 0 || tables.ne[0] <= 0 || tables.ne[1] <= 0) { fail("empty cache"); }
    ResolvedCache resolved{};
    if (storage == KvCacheStorage::BFloat16) {
        resolved.profile = detail::SelectedKvProfile::Bf16KFp16V;
        require(k_pages, DType::BF16, {kHeadDim, 64, kKvHeads, pages}, "K pages");
        require(v_pages, DType::FP16, {kHeadDim, 64, kKvHeads, pages}, "V pages");
        if (k_scale_pages.data != nullptr || v_scale_pages.data != nullptr) {
            fail("BF16 cache must not carry scale planes");
        }
    } else if (storage == KvCacheStorage::Fp8E4M3Row256) {
        resolved.profile = detail::SelectedKvProfile::Fp8Row256;
        require(k_pages, DType::FP8_E4M3FN, {kHeadDim, 64, kKvHeads, pages}, "K code pages");
        require(v_pages, DType::FP8_E4M3FN, {kHeadDim, 64, kKvHeads, pages}, "V code pages");
        require(k_scale_pages, DType::FP16, {1, 64, kKvHeads, pages}, "K scale pages");
        require(v_scale_pages, DType::FP16, {1, 64, kKvHeads, pages}, "V scale pages");
    } else {
        fail("unsupported KV cache storage");
    }
    if (tables.dtype != DType::I32 || tables.ne[2] != 1 || tables.ne[3] != 1 ||
        !tables.is_contiguous() || tables.data == nullptr) {
        fail("invalid block tables");
    }
    resolved.planes.k             = k_pages.data;
    resolved.planes.v             = v_pages.data;
    resolved.planes.k_scale       = static_cast<const std::uint16_t*>(k_scale_pages.data);
    resolved.planes.v_scale       = static_cast<const std::uint16_t*>(v_scale_pages.data);
    resolved.planes.tables        = static_cast<const std::int32_t*>(tables.data);
    resolved.planes.logical_pages = tables.ne[0];
    return resolved;
}

void require_columns(const Tensor& q, const Tensor& positions, const Tensor& selections,
                     const Tensor& counts, const Tensor& out, std::int32_t columns) {
    require(q, DType::BF16, {kHeadDim, kQueryHeads, columns}, "q");
    require(out, DType::BF16, {kHeadDim, kQueryHeads, columns}, "out");
    require(positions, DType::I32, {columns}, "positions");
    require(selections, DType::I32, {kMaxSelected, columns}, "selections");
    require(counts, DType::I32, {columns}, "counts");
}

void require_output_disjoint(const Tensor& out, std::initializer_list<const Tensor*> others) {
    for (const Tensor* other : others) {
        if (overlaps(out, *other)) { fail("out overlaps an input"); }
    }
}

template <class Arena>
float* allocate_partial(Arena& arena, std::int32_t columns) {
    return static_cast<float*>(
        arena.alloc_bytes(detail::selected_block_decode_partial_bytes(columns), 256).data);
}

} // namespace

std::size_t selected_block_attention_workspace_capacity_bytes(std::int32_t min_columns,
                                                              std::int32_t max_columns) {
    if (min_columns < 1 || max_columns < min_columns || max_columns > kMaxColumns) {
        fail("invalid column interval");
    }
    // The partition count depends on C, so the requirement is the largest over the interval.
    std::size_t capacity = 0;
    for (std::int32_t columns = min_columns; columns <= max_columns; ++columns) {
        WorkspaceLayoutBuilder layout;
        (void)allocate_partial(layout, columns);
        capacity = std::max(capacity, layout.peak_bytes(1));
    }
    return capacity;
}

void selected_block_attention(const Tensor& q, const Tensor& positions, const Tensor& table_rows,
                              const Tensor& selections, const Tensor& counts,
                              const PagedKVBatchLayerView& cache, WorkspaceArena& workspace,
                              Tensor& out, cudaStream_t stream) {
    const std::int32_t columns = q.ne[2];
    if (columns < 1 || columns > kMaxColumns) { fail("batched C must be 1..48"); }
    require_columns(q, positions, selections, counts, out, columns);
    require(table_rows, DType::I32, {columns}, "table rows");
    const ResolvedCache resolved =
        resolve_cache(cache.k_pages, cache.v_pages, cache.k_scale_pages, cache.v_scale_pages,
                      cache.block_tables, cache.head_dim, cache.num_kv_heads, cache.storage);
    require_output_disjoint(out, {&q, &positions, &table_rows, &selections, &counts, &cache.k_pages,
                                  &cache.v_pages, &cache.k_scale_pages, &cache.v_scale_pages,
                                  &cache.block_tables});
    const auto scope = workspace.scope();
    float* partial   = allocate_partial(workspace, columns);
    detail::selected_block_decode_launch(resolved.profile, q, positions, table_rows, selections,
                                         counts, resolved.planes, partial, out, stream);
}

void selected_block_attention(const Tensor& q, const Tensor& positions, const Tensor& selections,
                              const Tensor& counts, const PagedKVLayerView& cache, Tensor& out,
                              cudaStream_t stream) {
    const std::int32_t columns = q.ne[2];
    if (columns < 1 || columns > kMaxSharedRow) { fail("shared-row C must be 1..262144"); }
    require_columns(q, positions, selections, counts, out, columns);
    const ResolvedCache resolved =
        resolve_cache(cache.k_pages, cache.v_pages, cache.k_scale_pages, cache.v_scale_pages,
                      cache.block_table, cache.head_dim, cache.num_kv_heads, cache.storage);
    if (cache.block_table.ne[1] != 1) { fail("shared-row form takes one block-table row"); }
    require_output_disjoint(out, {&q, &positions, &selections, &counts, &cache.k_pages,
                                  &cache.v_pages, &cache.k_scale_pages, &cache.v_scale_pages,
                                  &cache.block_table});
    detail::selected_block_prefill_launch(resolved.profile, q, positions, selections, counts,
                                          resolved.planes, out, stream);
}

} // namespace ninfer::ops
