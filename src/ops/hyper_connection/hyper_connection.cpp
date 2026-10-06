#include "ninfer/ops/hyper_connection.h"

#include "core/layout.h"
#include "ops/hyper_connection/launch.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHidden  = 2560;
constexpr std::int32_t kConcat  = 10240;
constexpr std::int32_t kLowRank = 320;
constexpr std::int32_t kStreams = 4;

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

[[noreturn]] void fail(const char* operation, const char* what) {
    throw std::invalid_argument(std::string(operation) + ": invalid " + what);
}

void require_matrix(const Tensor& tensor, DType dtype, std::int32_t rows, std::int32_t tokens,
                    const char* operation, const char* label) {
    if (tensor.dtype != dtype || tensor.ne[0] != rows || tensor.ne[1] != tokens ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        fail(operation, label);
    }
}

void require_bf16_weight(const Weight& weight, std::int32_t rows, std::int32_t columns,
                         const char* operation, const char* label) {
    if (weight.qtype != QType::BF16 || weight.layout != QuantLayout::Contiguous ||
        weight.n != rows || weight.k != columns || weight.ndim != 2 || weight.shape[0] != rows ||
        weight.shape[1] != columns || weight.qdata == nullptr || !aligned_to(weight.qdata, 16)) {
        fail(operation, label);
    }
}

std::uintptr_t begin(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); }

bool overlaps(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    return begin(a) < begin(b) + b_bytes && begin(b) < begin(a) + a_bytes;
}

std::size_t weight_bytes(const Weight& weight) {
    return static_cast<std::size_t>(weight.n) * static_cast<std::size_t>(weight.k) * 2;
}

void require_disjoint_output(const Tensor& output, const Tensor& hidden,
                             const HyperConnectionWeights& weights, const Weight* inject,
                             const char* operation, const char* label) {
    const std::size_t bytes = output.bytes();
    if (overlaps(output.data, bytes, hidden.data, hidden.bytes()) ||
        overlaps(output.data, bytes, weights.norm.data, weights.norm.bytes()) ||
        overlaps(output.data, bytes, weights.input_mix_down.qdata,
                 weight_bytes(weights.input_mix_down)) ||
        overlaps(output.data, bytes, weights.input_mix_up.qdata,
                 weight_bytes(weights.input_mix_up)) ||
        (inject != nullptr && overlaps(output.data, bytes, inject->qdata, weight_bytes(*inject)))) {
        fail(operation, label);
    }
}

void validate_mix(const Tensor& hidden, const HyperConnectionWeights& weights,
                  const Tensor& block_input, const char* operation) {
    const std::int32_t tokens = hidden.ne[1];
    if (tokens <= 0) fail(operation, "token extent");
    require_matrix(hidden, DType::BF16, kConcat, tokens, operation, "hidden");
    require_matrix(block_input, DType::BF16, kHidden, tokens, operation, "block_input");
    require_matrix(weights.norm, DType::BF16, kConcat, 1, operation, "norm");
    require_bf16_weight(weights.input_mix_down, kLowRank, kConcat, operation, "input_mix_down");
    require_bf16_weight(weights.input_mix_up, kConcat, kLowRank, operation, "input_mix_up");
}

template <class Arena>
detail::HyperScratch allocate_scratch(Arena& arena, std::int32_t tokens) {
    detail::HyperScratch scratch;
    scratch.normalized = arena.alloc_bytes(static_cast<std::size_t>(kConcat) * tokens * 2).data;
    scratch.low_rank   = arena.alloc_bytes(static_cast<std::size_t>(kLowRank) * tokens * 2).data;
    if (tokens > detail::kHyperDecodeTokens) {
        scratch.partial_bytes = static_cast<std::size_t>(detail::kHyperDownSplits) * kLowRank *
                                static_cast<std::size_t>(tokens) * sizeof(float);
        scratch.partials = static_cast<float*>(arena.alloc_bytes(scratch.partial_bytes).data);
        scratch.up       = static_cast<float*>(
            arena.alloc_bytes(static_cast<std::size_t>(kConcat) * tokens * sizeof(float)).data);
    }
    return scratch;
}

void run(const Tensor& hidden, const HyperConnectionWeights& weights, const Weight* inject,
         Tensor& block_input, Tensor* injection, WorkspaceArena& workspace, cudaStream_t stream) {
    const auto scope   = workspace.scope();
    const auto scratch = allocate_scratch(workspace, hidden.ne[1]);
    detail::hyper_mix_launch(hidden.data, weights.norm.data, weights.input_mix_down.qdata,
                             weights.input_mix_up.qdata,
                             inject != nullptr ? inject->qdata : nullptr, block_input.data,
                             injection != nullptr ? static_cast<float*>(injection->data) : nullptr,
                             scratch, hidden.ne[1], stream);
}

} // namespace

std::size_t hyper_connection_workspace_capacity_bytes(std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("hyper_connection workspace: invalid token interval");
    }
    // Every scratch plane grows with T and the wide route is a superset of the decode route.
    WorkspaceLayoutBuilder layout;
    (void)allocate_scratch(layout, max_tokens);
    return layout.peak_bytes(256);
}

void hyper_connection_prepare(const Tensor& hidden, const HyperConnectionWeights& weights,
                              const Weight& block_inject, Tensor& block_input, Tensor& injection,
                              WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr const char* op = "hyper_connection_prepare";
    validate_mix(hidden, weights, block_input, op);
    require_bf16_weight(block_inject, kStreams, kConcat, op, "block_inject");
    require_matrix(injection, DType::FP32, kStreams, hidden.ne[1], op, "injection");
    require_disjoint_output(block_input, hidden, weights, &block_inject, op, "block_input alias");
    require_disjoint_output(injection, hidden, weights, &block_inject, op, "injection alias");
    if (overlaps(block_input.data, block_input.bytes(), injection.data, injection.bytes())) {
        fail(op, "output alias");
    }
    run(hidden, weights, &block_inject, block_input, &injection, workspace, stream);
}

void hyper_connection_mix(const Tensor& hidden, const HyperConnectionWeights& weights,
                          Tensor& block_input, WorkspaceArena& workspace, cudaStream_t stream) {
    constexpr const char* op = "hyper_connection_mix";
    validate_mix(hidden, weights, block_input, op);
    require_disjoint_output(block_input, hidden, weights, nullptr, op, "block_input alias");
    run(hidden, weights, nullptr, block_input, nullptr, workspace, stream);
}

void hyper_connection_inject(const Tensor& block_output, const Tensor& injection, Tensor& hidden,
                             cudaStream_t stream) {
    constexpr const char* op  = "hyper_connection_inject";
    const std::int32_t tokens = hidden.ne[1];
    if (tokens <= 0) fail(op, "token extent");
    require_matrix(hidden, DType::BF16, kConcat, tokens, op, "hidden");
    require_matrix(block_output, DType::BF16, kHidden, tokens, op, "block_output");
    require_matrix(injection, DType::FP32, kStreams, tokens, op, "injection");
    if (overlaps(hidden.data, hidden.bytes(), block_output.data, block_output.bytes()) ||
        overlaps(hidden.data, hidden.bytes(), injection.data, injection.bytes())) {
        fail(op, "alias");
    }
    detail::hyper_inject_launch(block_output.data, static_cast<const float*>(injection.data),
                                hidden.data, tokens, stream);
}

void hyper_connection_expand(const Tensor& x, Tensor& hidden, cudaStream_t stream) {
    constexpr const char* op  = "hyper_connection_expand";
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 0) fail(op, "token extent");
    require_matrix(x, DType::BF16, kHidden, tokens, op, "x");
    require_matrix(hidden, DType::BF16, kConcat, tokens, op, "hidden");
    if (overlaps(x.data, x.bytes(), hidden.data, hidden.bytes())) fail(op, "alias");
    detail::hyper_expand_launch(x.data, hidden.data, tokens, stream);
}

} // namespace ninfer::ops
