#include "ninfer/ops/ple_ngram.h"

#include "core/layout.h"
#include "ops/ple_ngram/launch.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kStreamWidth = 2560;
constexpr std::int32_t kWidth       = 10240;
constexpr std::int32_t kHistory     = 9;
constexpr std::int32_t kRowWidth    = 160;
constexpr std::int32_t kMaximumRows = 8;

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

[[noreturn]] void fail(const char* operation, const std::string& what) {
    throw std::invalid_argument(std::string(operation) + ": invalid " + what);
}

void require(const Tensor& tensor, DType dtype, const std::array<std::int32_t, 4>& shape,
             const char* operation, const char* label) {
    for (std::size_t i = 0; i < shape.size(); ++i)
        if (tensor.ne[i] != shape[i]) fail(operation, label);
    // Index tensors are read as scalars; value planes use 16-byte vector access.
    const std::uintptr_t alignment = dtype == DType::I32 ? 4 : 16;
    if (tensor.dtype != dtype || !tensor.is_contiguous() || !aligned_to(tensor.data, alignment))
        fail(operation, label);
}

std::uintptr_t address(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); }

bool overlaps(const Tensor& a, const Tensor& b) {
    return address(a.data) < address(b.data) + b.bytes() &&
           address(b.data) < address(a.data) + a.bytes();
}

template <class Arena>
detail::PleScratch allocate(Arena& arena, std::int32_t columns) {
    const auto n = static_cast<std::size_t>(columns);
    auto* gates  = static_cast<float*>(arena.alloc_bytes(4 * n * sizeof(float)).data);
    return {gates, arena.alloc_bytes(static_cast<std::size_t>(kWidth) * n * 2).data};
}

detail::PleWeights validate_weights(const Tensor& query_norm, const Tensor& key_norm,
                                    const Tensor& conv_norm, const Tensor& conv_weight,
                                    const char* operation) {
    require(query_norm, DType::BF16, {kWidth, 1, 1, 1}, operation, "query_norm");
    require(key_norm, DType::BF16, {kWidth, 1, 1, 1}, operation, "key_norm");
    require(conv_norm, DType::BF16, {kWidth, 1, 1, 1}, operation, "conv_norm");
    require(conv_weight, DType::BF16, {kWidth, 4, 1, 1}, operation, "conv_weight");
    return {query_norm.data, key_norm.data, conv_norm.data, conv_weight.data};
}

void require_output_disjoint(const Tensor& out, std::initializer_list<const Tensor*> inputs,
                             const char* operation) {
    for (const Tensor* input : inputs)
        if (overlaps(out, *input)) fail(operation, "output alias");
}

} // namespace

void ple_ngram_decode(const Tensor& codes, const Tensor& scales, Tensor& out,
                      cudaStream_t stream) {
    constexpr const char* op = "ple_ngram_decode";
    const std::int32_t rows  = codes.ne[1];
    if (rows <= 0) fail(op, "row extent");
    require(codes, DType::U8, {kRowWidth / 2, rows, 1, 1}, op, "codes");
    require(scales, DType::FP16, {kRowWidth / 16, rows, 1, 1}, op, "scales");
    require(out, DType::BF16, {kRowWidth, rows, 1, 1}, op, "out");
    if (overlaps(out, codes) || overlaps(out, scales)) fail(op, "output alias");
    detail::ple_decode_launch(codes.data, scales.data, out.data, rows, stream);
}

std::size_t ple_ngram_workspace_capacity_bytes(std::int32_t min_columns,
                                               std::int32_t max_columns) {
    if (min_columns <= 0 || max_columns < min_columns)
        throw std::invalid_argument("ple_ngram workspace: invalid column interval");
    WorkspaceLayoutBuilder layout;
    (void)allocate(layout, max_columns);
    return layout.peak_bytes(256);
}

void ple_ngram(const Tensor& hidden, const Tensor& key, const Tensor& value,
               const Tensor& query_norm, const Tensor& key_norm, const Tensor& conv_norm,
               const Tensor& conv_weight, const Tensor& state_in, Tensor& state_out, Tensor& out,
               WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr const char* op = "ple_ngram";
    const std::int32_t T     = hidden.ne[1];
    if (T <= 0) fail(op, "column extent");
    require(hidden, DType::BF16, {kWidth, T, 1, 1}, op, "hidden");
    require(key, DType::BF16, {kWidth, T, 1, 1}, op, "key");
    require(value, DType::BF16, {kStreamWidth, T, 1, 1}, op, "value");
    require(out, DType::BF16, {kWidth, T, 1, 1}, op, "out");
    require(state_in, DType::BF16, {kWidth, kHistory, 1, 1}, op, "state_in");
    require(state_out, DType::BF16, {kWidth, kHistory, 1, 1}, op, "state_out");
    const auto weights = validate_weights(query_norm, key_norm, conv_norm, conv_weight, op);
    if (state_in.data != state_out.data && overlaps(state_in, state_out)) fail(op, "state alias");
    require_output_disjoint(out, {&hidden, &key, &value, &query_norm, &key_norm, &conv_norm,
                                  &conv_weight, &state_in, &state_out},
                            op);
    for (const Tensor* input :
         {&hidden, &key, &value, &query_norm, &key_norm, &conv_norm, &conv_weight})
        if (overlaps(state_out, *input)) fail(op, "state alias");

    const auto scope   = workspace.scope();
    const auto scratch = allocate(workspace, T);
    detail::ple_gate_launch(hidden.data, key.data, value.data, weights, scratch, T, stream);
    detail::ple_conv_launch(weights, scratch, value.data, state_in.data, state_out.data, out.data,
                            T, stream);
}

void ple_ngram_snapshot(const Tensor& hidden, const Tensor& key, const Tensor& value,
                        const Tensor& query_norm, const Tensor& key_norm, const Tensor& conv_norm,
                        const Tensor& conv_weight, Tensor& states, const Tensor& valid_columns,
                        const Tensor& initial_slots, const Tensor& snapshot_base_slots,
                        Tensor& out, WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr const char* op = "ple_ngram_snapshot";
    const std::int32_t W     = hidden.ne[1];
    const std::int32_t B     = hidden.ne[2];
    if (W <= 0 || B <= 0 || B > kMaximumRows) fail(op, "column or row extent");
    require(hidden, DType::BF16, {kWidth, W, B, 1}, op, "hidden");
    require(key, DType::BF16, {kWidth, W, B, 1}, op, "key");
    require(value, DType::BF16, {kStreamWidth, W, B, 1}, op, "value");
    require(out, DType::BF16, {kWidth, W, B, 1}, op, "out");
    const std::int32_t slots = states.ne[2];
    if (slots <= 0) fail(op, "states");
    require(states, DType::BF16, {kWidth, kHistory, slots, 1}, op, "states");
    require(initial_slots, DType::I32, {B, 1, 1, 1}, op, "initial_slots");
    require(snapshot_base_slots, DType::I32, {B, 1, 1, 1}, op, "snapshot_base_slots");
    const bool all_valid = valid_columns.data == nullptr;
    if (!all_valid) require(valid_columns, DType::I32, {B, 1, 1, 1}, op, "valid_columns");
    const auto weights = validate_weights(query_norm, key_norm, conv_norm, conv_weight, op);
    require_output_disjoint(out, {&hidden, &key, &value, &query_norm, &key_norm, &conv_norm,
                                  &conv_weight, &states, &initial_slots, &snapshot_base_slots},
                            op);
    for (const Tensor* input : {&hidden, &key, &value, &query_norm, &key_norm, &conv_norm,
                                &conv_weight, &initial_slots, &snapshot_base_slots})
        if (overlaps(states, *input)) fail(op, "state alias");
    if (!all_valid && (overlaps(out, valid_columns) || overlaps(states, valid_columns)))
        fail(op, "valid_columns alias");

    const auto scope   = workspace.scope();
    const auto scratch = allocate(workspace, W * B);
    detail::ple_gate_launch(hidden.data, key.data, value.data, weights, scratch, W * B, stream);
    detail::ple_snapshot_launch(
        weights, scratch, value.data, states.data,
        all_valid ? nullptr : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<const std::int32_t*>(initial_slots.data),
        static_cast<const std::int32_t*>(snapshot_base_slots.data), out.data, W, B, stream);
}

} // namespace ninfer::ops
