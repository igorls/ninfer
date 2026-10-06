#pragma once

// Phase allocations shared by Text execution and startup sizing. Every stage helper is called with
// the execution WorkspaceArena or the sizing WorkspaceLayoutBuilder in the same order, so the
// planned capacity and the executed high-water mark agree by construction.

#include "core/arena.h"
#include "core/layout.h"
#include "models/qwen4_exp/execution/parameters.h"

#include <cstdint>

namespace ninfer::models::qwen4_exp::execution::workspace {

template <class Allocator>
Tensor matrix(Allocator& allocator, DType dtype, std::int32_t rows, std::int32_t tokens) {
    return allocator.alloc(dtype, {rows, tokens});
}

template <class Allocator>
Tensor vector(Allocator& allocator, DType dtype, std::int32_t elements) {
    return allocator.alloc(dtype, {elements});
}

// Prefill-only control inputs: token ids, cache positions and planar [T,3] MRoPE positions.
struct PrefillControls {
    Tensor ids;
    Tensor positions;
    Tensor rope_positions;
};

template <class Allocator>
PrefillControls prefill_controls(Allocator& allocator, std::int32_t tokens) {
    return {vector(allocator, DType::I32, tokens), vector(allocator, DType::I32, tokens),
            matrix(allocator, DType::I32, tokens, 3)};
}

// Live across every layer beside the four-stream residual (which prefill allocates with
// stream_hidden and decode receives from its round buffers): the current block's input and stream
// injection, and the block output awaiting injection.
struct LayerRoots {
    Tensor hidden;
    Tensor block_input;
    Tensor injection;
    Tensor block_output;
};

template <class Allocator>
LayerRoots layer_roots(Allocator& allocator, const TextConfig& config, std::int32_t tokens,
                       const Tensor& hidden) {
    return {
        hidden,
        matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens),
        matrix(allocator, DType::FP32, dimension(config.hyper_connection.streams), tokens),
        matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens),
    };
}

// Prefill PLE row staging (decode rows arrive in the round ingress).
struct PleRows {
    Tensor codes;
    Tensor scales;
};

template <class Allocator>
PleRows ple_rows(Allocator& allocator, const TextConfig& config, std::int32_t tokens) {
    const std::int32_t rows = dimension(std::uint64_t(config.ple.heads()) * std::uint64_t(tokens));
    const std::int32_t width = dimension(config.ple.head_width());
    return {matrix(allocator, DType::U8, width / 2, rows),
            matrix(allocator, DType::FP16, width / 16, rows)};
}

struct PleStage {
    Tensor embedding;
    Tensor key;
    Tensor value;
    Tensor output;
};

template <class Allocator>
PleStage ple_stage(Allocator& allocator, const TextConfig& config, std::int32_t tokens) {
    return {
        matrix(allocator, DType::BF16, dimension(config.ple.embed_dim), tokens),
        matrix(allocator, DType::BF16, dimension(config.stream_width()), tokens),
        matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens),
        matrix(allocator, DType::BF16, dimension(config.stream_width()), tokens),
    };
}

struct AttentionStage {
    Tensor projection;
    Tensor query;
    Tensor gate;
    Tensor key;
    Tensor value;
    Tensor indexer;
    Tensor selections;
    Tensor counts;
    Tensor attended;
};

template <class Allocator>
AttentionStage attention_stage(Allocator& allocator, const TextConfig& config,
                               std::int32_t tokens) {
    const auto& a   = config.attention;
    const auto& idx = config.indexer;
    const std::int32_t projection_rows =
        dimension(2ULL * a.query_width() + 2ULL * a.key_width());
    const std::int32_t indexer_rows =
        dimension(std::uint64_t(idx.num_heads + idx.num_key_value_heads) * idx.head_dim);
    const std::int32_t selected = dimension(idx.budget / idx.compress_ratio);
    return {
        matrix(allocator, DType::BF16, projection_rows, tokens),
        matrix(allocator, DType::BF16, dimension(a.query_width()), tokens),
        matrix(allocator, DType::BF16, dimension(a.query_width()), tokens),
        matrix(allocator, DType::BF16, dimension(a.key_width()), tokens),
        matrix(allocator, DType::BF16, dimension(a.key_width()), tokens),
        matrix(allocator, DType::BF16, indexer_rows, tokens),
        matrix(allocator, DType::I32, selected, tokens),
        vector(allocator, DType::I32, tokens),
        matrix(allocator, DType::BF16, dimension(a.query_width()), tokens),
    };
}

struct GdnStage {
    Tensor qkv; // prefill only: projected [q,k,v] before the convolution
    Tensor z;
    Tensor query;
    Tensor key;
    Tensor value;
    Tensor g;
    Tensor beta;
    Tensor recurrent;
    Tensor normalized;
};

template <class Allocator>
GdnStage gdn_stage(Allocator& allocator, const TextConfig& config, std::int32_t tokens,
                   bool prefill) {
    const auto& g = config.gdn;
    GdnStage out;
    if (prefill) {
        out.qkv = matrix(allocator, DType::BF16, dimension(g.conv_channels()), tokens);
    }
    out.z          = matrix(allocator, DType::BF16, dimension(g.value_width()), tokens);
    out.query      = matrix(allocator, DType::BF16, dimension(g.key_width()), tokens);
    out.key        = matrix(allocator, DType::BF16, dimension(g.key_width()), tokens);
    out.value      = matrix(allocator, DType::BF16, dimension(g.value_width()), tokens);
    out.g          = matrix(allocator, DType::FP32, dimension(g.linear_num_value_heads), tokens);
    out.beta       = matrix(allocator, DType::FP32, dimension(g.linear_num_value_heads), tokens);
    out.recurrent  = matrix(allocator, DType::BF16, dimension(g.value_width()), tokens);
    out.normalized = matrix(allocator, DType::BF16, dimension(g.value_width()), tokens);
    return out;
}

template <class Allocator>
Tensor mixed_hidden(Allocator& allocator, const TextConfig& config, std::int32_t tokens) {
    return matrix(allocator, DType::BF16, dimension(config.hidden_size), tokens);
}

template <class Allocator>
Tensor stream_hidden(Allocator& allocator, const TextConfig& config, std::int32_t tokens) {
    return matrix(allocator, DType::BF16, dimension(config.stream_width()), tokens);
}

} // namespace ninfer::models::qwen4_exp::execution::workspace
