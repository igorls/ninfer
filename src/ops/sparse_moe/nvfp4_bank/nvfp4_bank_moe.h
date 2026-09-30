#pragma once

// Private plan, workspace and launch entries of the NVFP4-bank SparseMoe overload
// (512 experts, top-10, K 2560, intermediate 640). Only the wrapper includes this header.

#include "core/arena.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "ninfer/ops/sparse_moe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kNvfp4MoeHidden       = 2'560;
inline constexpr std::int32_t kNvfp4MoeExperts      = 512;
inline constexpr std::int32_t kNvfp4MoeTopK         = 10;
inline constexpr std::int32_t kNvfp4MoePaths        = kNvfp4MoeTopK + 1; // ten routed + shared
inline constexpr std::int32_t kNvfp4MoeIntermediate = 640;

// Private route boundaries. Decode keeps one fused router launch and warp-per-row expert kernels;
// grouped prefill loads each active expert once per call. The A4 Tensor Core arm replaces the SIMT
// arm from the extent where activation quantization is repaid.
inline constexpr std::int32_t kNvfp4MoeDecodeMaxTokens = 8;
inline constexpr std::int32_t kNvfp4MoeA4MinTokens     = 256;

enum class Nvfp4MoeRoute : std::uint8_t { Decode, GroupedA16, GroupedA4 };

[[nodiscard]] Nvfp4MoeRoute resolve_nvfp4_moe_route(std::int32_t tokens, bool allow_a4);

struct Nvfp4MoeWorkspace {
    Tensor scores;       // FP32 [513,T]: router rows then the shared-expert gate score
    Tensor ids;          // I32 [10,T]
    Tensor weights;      // FP32 [10,T] renormalized route weights
    Tensor shared_scale; // FP32 [T]
    Tensor activations;  // BF16 [640,11,T]: ten routed paths then the shared path
    Tensor arrivals;     // I32 [4]: decode router completion counter
    // Grouped routes.
    Tensor expert_counts;  // I32 [512]
    Tensor expert_offsets; // I32 [513]
    Tensor active_experts; // I32 [512]
    Tensor active_count;   // I32 [1]
    Tensor grouped_tokens; // I32 [10T]
    Tensor grouped_paths;  // I32 [10T]
    Tensor token_to_pos;   // I32 [10T]
    Tensor shared_gemm;    // BF16 [640,2T]: shared gate then shared up / product
    // GroupedA16.
    Tensor down_partials; // FP32 [10,2560,T]
    // GroupedA4.
    Tensor input_codes;       // U8 [1280,T]
    Tensor input_scales;      // U8 [160,T]
    Tensor product_codes;     // U8 [320,11T]
    Tensor product_scales;    // U8 [40,11T]
    Tensor weighted_products; // BF16 [2560,10T]
};

template <class Arena>
Nvfp4MoeWorkspace allocate_nvfp4_moe_workspace(Arena& arena, std::int32_t tokens,
                                               Nvfp4MoeRoute route) {
    Nvfp4MoeWorkspace out;
    out.scores       = arena.alloc(DType::FP32, {kNvfp4MoeExperts + 1, tokens});
    out.ids          = arena.alloc(DType::I32, {kNvfp4MoeTopK, tokens});
    out.weights      = arena.alloc(DType::FP32, {kNvfp4MoeTopK, tokens});
    out.shared_scale = arena.alloc(DType::FP32, {tokens});
    out.activations  = arena.alloc(DType::BF16, {kNvfp4MoeIntermediate, kNvfp4MoePaths, tokens});
    out.arrivals     = arena.alloc(DType::I32, {4});
    if (route == Nvfp4MoeRoute::Decode) { return out; }

    const std::int32_t items = kNvfp4MoeTopK * tokens;
    out.expert_counts        = arena.alloc(DType::I32, {kNvfp4MoeExperts});
    out.expert_offsets       = arena.alloc(DType::I32, {kNvfp4MoeExperts + 1});
    out.active_experts       = arena.alloc(DType::I32, {kNvfp4MoeExperts});
    out.active_count         = arena.alloc(DType::I32, {1});
    out.grouped_tokens       = arena.alloc(DType::I32, {items});
    out.grouped_paths        = arena.alloc(DType::I32, {items});
    out.token_to_pos         = arena.alloc(DType::I32, {items});
    out.shared_gemm          = arena.alloc(DType::BF16, {kNvfp4MoeIntermediate, 2 * tokens});
    if (route == Nvfp4MoeRoute::GroupedA16) {
        out.down_partials = arena.alloc(DType::FP32, {kNvfp4MoeTopK, kNvfp4MoeHidden, tokens});
        return out;
    }
    out.input_codes  = arena.alloc(DType::U8, {kNvfp4MoeHidden / 2, tokens});
    out.input_scales = arena.alloc(DType::U8, {kNvfp4MoeHidden / 16, tokens});
    out.product_codes =
        arena.alloc(DType::U8, {kNvfp4MoeIntermediate / 2, kNvfp4MoePaths * tokens});
    out.product_scales =
        arena.alloc(DType::U8, {kNvfp4MoeIntermediate / 16, kNvfp4MoePaths * tokens});
    out.weighted_products = arena.alloc(DType::BF16, {kNvfp4MoeHidden, items});
    return out;
}

[[nodiscard]] std::size_t nvfp4_moe_workspace_bytes(std::int32_t tokens, Nvfp4MoeRoute route);

// Router projection, top-10 selection, renormalization and shared-expert gate.
void nvfp4_moe_route(const Tensor& x, const SparseMoeNvfp4BankWeights& weights,
                     const Nvfp4MoeWorkspace& workspace, cudaStream_t stream);

void nvfp4_moe_decode(const Tensor& x, const SparseMoeNvfp4BankWeights& weights,
                      const Nvfp4MoeWorkspace& workspace, Tensor& destination, cudaStream_t stream);

void nvfp4_moe_grouped(const Tensor& x, const SparseMoeNvfp4BankWeights& weights,
                       const Nvfp4MoeWorkspace& workspace, Nvfp4MoeRoute route, Tensor& destination,
                       cudaStream_t stream);

} // namespace ninfer::ops::detail
