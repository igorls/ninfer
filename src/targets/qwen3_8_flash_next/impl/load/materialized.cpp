#include "targets/qwen3_8_flash_next/impl/load/materialized.h"

#include "artifact/typed_binding.h"

#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

using artifact::NumericFormat;

Tensor bf16_tensor(const artifact::MaterializedArtifact& backing, artifact::ObjectHandle handle,
                   std::initializer_list<std::int32_t> shape) {
    return artifact::materialized_tensor(backing, handle, NumericFormat::BF16, shape);
}

Weight bf16_weight(const artifact::MaterializedArtifact& backing, artifact::ObjectHandle handle,
                   std::int32_t rows, std::int32_t columns) {
    return artifact::materialized_weight(backing, handle, NumericFormat::BF16, rows, columns);
}

Weight fp8_weight(const artifact::MaterializedArtifact& backing, artifact::ObjectHandle handle,
                  std::int32_t rows, std::int32_t columns) {
    return artifact::materialized_weight(backing, handle, NumericFormat::FP8_E4M3FN_ROW_F32S, rows,
                                         columns);
}

HyperConnectionWeights load_hyper(const HyperConnectionPlan& plan,
                                  const artifact::MaterializedArtifact& backing) {
    return {
        .block_inject   = bf16_weight(backing, plan.block_inject, 4, 10'240),
        .norm           = bf16_tensor(backing, plan.norm, {10'240}),
        .input_mix_down = bf16_weight(backing, plan.input_mix_down, 320, 10'240),
        .input_mix_up   = bf16_weight(backing, plan.input_mix_up, 10'240, 320),
    };
}

HyperMixerWeights load_mixer(const HyperMixerPlan& plan,
                             const artifact::MaterializedArtifact& backing) {
    return {
        .norm           = bf16_tensor(backing, plan.norm, {10'240}),
        .input_mix_down = bf16_weight(backing, plan.input_mix_down, 320, 10'240),
        .input_mix_up   = bf16_weight(backing, plan.input_mix_up, 10'240, 320),
    };
}

MoeWeights load_moe(const MoePlan& plan, const artifact::MaterializedArtifact& backing) {
    return {
        .router             = bf16_weight(backing, plan.router, 512, 2'560),
        .shared_down        = bf16_weight(backing, plan.shared_down, 2'560, 640),
        .shared_gate        = bf16_weight(backing, plan.shared_gate, 640, 2'560),
        .shared_up          = bf16_weight(backing, plan.shared_up, 640, 2'560),
        .shared_gate_weight = bf16_weight(backing, plan.shared_gate_weight, 1, 2'560),
        .expert_gate_up =
            materialized_nvfp4_expert_bank_view(backing, plan.expert_gate_up, 512, 1'280, 2'560),
        .expert_down =
            materialized_nvfp4_expert_bank_view(backing, plan.expert_down, 512, 2'560, 640),
    };
}

GdnWeights load_gdn(const GdnPlan& plan, const artifact::MaterializedArtifact& backing) {
    return {
        .a_log             = bf16_tensor(backing, plan.a_log, {48}),
        .convolution       = bf16_tensor(backing, plan.convolution, {10'240, 4}),
        .dt_bias           = bf16_tensor(backing, plan.dt_bias, {48}),
        .a_b_projection    = bf16_weight(backing, plan.a_b_projection, 96, 2'560),
        .norm              = bf16_tensor(backing, plan.norm, {128}),
        .query_key_value_z = fp8_weight(backing, plan.query_key_value_z, 16'384, 2'560),
        .output            = fp8_weight(backing, plan.output, 2'560, 6'144),
    };
}

AttentionWeights load_attention(const AttentionPlan& plan,
                                const artifact::MaterializedArtifact& backing) {
    return {
        .indexer_query_key    = bf16_weight(backing, plan.indexer_query_key, 640, 2'560),
        .indexer_key_norm     = bf16_tensor(backing, plan.indexer_key_norm, {128}),
        .indexer_query_norm   = bf16_tensor(backing, plan.indexer_query_norm, {128}),
        .key_norm             = bf16_tensor(backing, plan.key_norm, {256}),
        .query_norm           = bf16_tensor(backing, plan.query_norm, {256}),
        .query_gate_key_value = fp8_weight(backing, plan.query_gate_key_value, 13'312, 2'560),
        .output               = fp8_weight(backing, plan.output, 2'560, 6'144),
    };
}

PleWeights load_ple(const PlePlan& plan, const artifact::MaterializedArtifact& backing) {
    PleWeights out{
        .convolution      = bf16_tensor(backing, plan.convolution, {10'240, 4}),
        .key_projection   = bf16_weight(backing, plan.key_projection, 10'240, 2'560),
        .conv_norm        = bf16_tensor(backing, plan.conv_norm, {10'240}),
        .key_norm         = bf16_tensor(backing, plan.key_norm, {10'240}),
        .query_norm       = bf16_tensor(backing, plan.query_norm, {10'240}),
        .value_projection = bf16_weight(backing, plan.value_projection, 2'560, 2'560),
    };
    for (std::size_t shard = 0; shard < out.table.shards.size(); ++shard) {
        out.table.shards[shard] =
            make_ple_shard_view(backing.mapped_tensor_bytes(plan.shards[shard]));
    }
    return out;
}

} // namespace

LoadedTextModelData::LoadedTextModelData(BindingPlan plan,
                                         artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    if (plan.features.mtp || plan.features.vision) {
        throw std::invalid_argument(
            "text-only Flash-Next materialization received optional weights");
    }
    for (std::size_t index = 0; index < frontend.size(); ++index) {
        frontend[index] = backing.take_resource_bytes(plan.frontend.resources[index]);
    }

    runtime.weights_arena   = &backing.device_arena();
    runtime.token_embedding = bf16_weight(backing, plan.token_embedding, 248'320, 2'560);
    std::size_t full_index  = 0;
    std::size_t gdn_index   = 0;
    for (std::size_t layer = 0; layer < plan.text_layers.size(); ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        TextLayerWeights& target    = runtime.layers[layer];
        target.attention_hyper      = load_hyper(source.attention_hyper, backing);
        target.moe                  = load_moe(source.moe, backing);
        target.mlp_hyper            = load_hyper(source.mlp_hyper, backing);
        if (source.is_full_attention) {
            runtime.full_attention.at(full_index++) = load_attention(source.attention, backing);
        } else {
            runtime.gdn.at(gdn_index++) = load_gdn(source.gdn, backing);
        }
    }
    if (full_index != runtime.full_attention.size() || gdn_index != runtime.gdn.size()) {
        throw std::logic_error("Flash-Next Text topology materialization is incomplete");
    }
    runtime.ple         = load_ple(plan.ple, backing);
    runtime.output_head = bf16_weight(backing, plan.output_head, 248'320, 2'560);
    runtime.final_mixer = load_mixer(plan.final_mixer, backing);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
