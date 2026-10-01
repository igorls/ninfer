#pragma once

#include "models/qwen4_exp/model.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/sparse_moe.h"
#include "ninfer/ops/weight_input.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <variant>
#include <vector>

namespace ninfer::models::qwen4_exp::execution {

using LinearParameters = ops::SingleProjectionWeight;

[[nodiscard]] inline std::int32_t dimension(std::uint64_t value) {
    if (value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("model dimension exceeds the Tensor integer domain");
    }
    return static_cast<std::int32_t>(value);
}

// One four-stream transition around a block: the input mixing and the injection of the block
// output back into the streams.
struct HyperParameters {
    ops::HyperConnectionWeights mixing;
    Weight block_inject;
};

// QSA attention: the packed [query, gate] x heads, key, value projection, the indexer projection
// of four 128-wide queries and one raw key, and the gated output projection.
struct AttentionParameters {
    LinearParameters projection;
    Tensor query_norm, key_norm;
    LinearParameters indexer_projection;
    Tensor indexer_query_norm, indexer_key_norm;
    LinearParameters output;
};

struct GdnParameters {
    LinearParameters projection; // [q,k,v,z] parent
    Weight control;              // [a,b] parent
    Tensor a_log, dt_bias, convolution, norm;
    LinearParameters output;
};

struct BlockParameters {
    HyperParameters attention_hyper, mlp_hyper;
    std::variant<AttentionParameters, GdnParameters> mixer;
    ops::SparseMoeNvfp4BankWeights moe;
};

// Device half of the per-layer n-gram embedding; the gathered rows come from Model::ple_table().
struct PleParameters {
    std::uint32_t layer = 0;
    LinearParameters key_projection, value_projection;
    Tensor query_norm, key_norm, conv_norm, convolution;
};

struct TextParameters {
    Weight token_embedding;
    LinearParameters output_head;
    std::vector<BlockParameters> layers;
    PleParameters ple;
    ops::HyperConnectionWeights final_mixer;
};

// Cold native preparation for the fixed model implementation. All weight addresses borrow the
// source Model; shape-dependent kernel selection and scratch remain with the calling Op.
class Parameters {
public:
    explicit Parameters(const Model& source);
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;
    Parameters(Parameters&&)                 = delete;
    Parameters& operator=(Parameters&&)      = delete;

    const Model& model;
    TextParameters text;
};

} // namespace ninfer::models::qwen4_exp::execution
