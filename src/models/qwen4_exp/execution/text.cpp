#include "models/qwen4_exp/execution/text.h"

#include "core/nvtx.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/candidate_logprobs.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/ple_ngram.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/qsa_indexer.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm_rope.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/selected_block_attention.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/sparse_moe.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen4_exp::execution {
namespace {

void require_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                   const char* label) {
    bool ok = t.dtype == dtype && t.is_contiguous() && t.data != nullptr;
    int i   = 0;
    for (const std::int32_t extent : shape) { ok = ok && t.ne[i++] == extent; }
    for (; i < 4; ++i) { ok = ok && t.ne[i] == 1; }
    if (!ok) { throw std::invalid_argument(std::string(label) + " has an invalid view"); }
}

Tensor columns(const Tensor& t, std::int32_t count) {
    if (count <= 0 || t.ne[1] < count || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("column window is outside its buffer");
    }
    return t.slice(1, 0, count);
}

float inverse_sqrt(std::uint32_t value) {
    return static_cast<float>(1.0 / std::sqrt(static_cast<double>(value)));
}

} // namespace

void enqueue_first_token_readout(const FirstTokenReadout& readout, const Tensor& logits,
                                 const Tensor& sampled, std::int32_t token_domain,
                                 cudaStream_t stream) {
    Tensor sampled_out                   = readout.sampled_out;
    std::optional<Tensor> candidates_out = readout.candidates_out;
    ops::candidate_logprobs(logits, token_domain, sampled,
                            readout.candidate_ids ? &*readout.candidate_ids : nullptr,
                            readout.allowed ? &*readout.allowed : nullptr, sampled_out,
                            candidates_out ? &*candidates_out : nullptr, stream);
    CUDA_CHECK(cudaMemcpyAsync(readout.host, readout.sampled_out.data, readout.bytes,
                               cudaMemcpyDeviceToHost, stream));
}

TextContext::TextContext(DeviceContext& ctx, const Parameters& parameters, WorkspaceArena& work,
                         PleGather& ple, qwen4_exp::PagedKVCacheView kv,
                         StateImageDevicePool& state, qwen4_exp::RoundState& io,
                         Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                         std::uint32_t text_kv_base, const qwen4_exp::PagedKVCache* batch_kv)
    : ctx_(ctx), parameters_(parameters), config_(parameters.model.config().text), work_(work),
      ple_(ple), kv_(kv), batch_kv_(batch_kv), state_(state), io_(io),
      prefill_hidden_(prefill_hidden), prefill_chunk_(prefill_chunk), text_kv_base_(text_kv_base) {
    if (prefill_chunk_ == 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TextContext prefill chunk must fit positive int32");
    }
}

void TextContext::set_state_slots(std::int32_t source_slot, std::int32_t destination_slot) {
    if (source_slot < 0 || source_slot >= state_.slot_count() || destination_slot < 0 ||
        destination_slot >= state_.slot_count()) {
        throw std::out_of_range("TextContext state slots are out of range");
    }
    state_source_slot_      = source_slot;
    state_destination_slot_ = destination_slot;
}

void TextContext::project(const Tensor& x, const LinearParameters& parameters, Tensor& out) {
    auto scope = work_.scope();
    // Verification is target decode even when its W*B physical extent exceeds a prefill
    // kernel threshold. It must retain the ordinary decode activation representation.
    const auto policy = decode_ ? ops::LinearPolicy::A16Only : parameters.policy;
    ops::linear(x, parameters.weight, out, policy, work_, ctx_.stream);
}

void TextContext::inject_ple(workspace::LayerRoots& roots, const Tensor& codes,
                             const Tensor& scales, bool prefill, std::int32_t tokens) {
    const PleParameters& p  = parameters_.text.ple;
    cudaStream_t s          = ctx_.stream;
    auto scope              = work_.scope();
    auto stage              = workspace::ple_stage(work_, config_, tokens);
    const std::int32_t rows = dimension(std::uint64_t(config_.ple.heads()) * std::uint64_t(tokens));
    Tensor decoded          = stage.embedding.view({dimension(config_.ple.head_width()), rows});
    ops::ple_ngram_decode(codes, scales, decoded, s);
    project(stage.embedding, p.key_projection, stage.key);
    project(stage.embedding, p.value_projection, stage.value);
    {
        auto ple_scope = work_.scope();
        WorkspaceArena scratch(work_.alloc_bytes(
            std::max<std::size_t>(1, ops::ple_ngram_workspace_capacity_bytes(tokens, tokens))));
        if (prefill) {
            const Tensor history_in = state_.ple_history_slot(state_source_slot_);
            Tensor history_out      = state_.ple_history_slot(state_destination_slot_);
            ops::ple_ngram(roots.hidden, stage.key, stage.value, p.query_norm, p.key_norm,
                           p.conv_norm, p.convolution, history_in, history_out, stage.output,
                           scratch, s);
        } else {
            Tensor histories           = state_.ple_history();
            const std::int32_t streams = dimension(config_.stream_width());
            const auto width           = decode_->width;
            const auto batch           = tokens / width;
            Tensor hidden3             = roots.hidden.view({streams, width, batch});
            Tensor key3                = stage.key.view({streams, width, batch});
            Tensor value3  = stage.value.view({dimension(config_.hidden_size), width, batch});
            Tensor output3 = stage.output.view({streams, width, batch});
            ops::ple_ngram_snapshot(hidden3, key3, value3, p.query_norm, p.key_norm, p.conv_norm,
                                    p.convolution, histories, decode_->valid_columns,
                                    decode_->state_source_slots, decode_->state_destination_slots,
                                    output3, scratch, s);
        }
    }
    ops::residual_add(stage.output, roots.hidden, s);
}

void TextContext::attention(const AttentionParameters& p, std::uint32_t layer,
                            workspace::LayerRoots& roots, bool prefill, std::int32_t tokens) {
    cudaStream_t s       = ctx_.stream;
    const auto& a        = config_.attention;
    const std::int32_t d = dimension(a.head_dim);
    auto scope           = work_.scope();
    auto stage           = workspace::attention_stage(work_, config_, tokens);
    project(roots.block_input, p.projection, stage.projection);
    Tensor q    = stage.query.view({d, dimension(a.num_attention_heads), tokens});
    Tensor gate = stage.gate.view({d, dimension(a.num_attention_heads), tokens});
    Tensor k    = stage.key.view({d, dimension(a.num_key_value_heads), tokens});
    Tensor v    = stage.value.view({d, dimension(a.num_key_value_heads), tokens});
    ops::rmsnorm_rope(stage.projection, *rope_positions_, p.query_norm, p.key_norm, q, gate, k, v,
                      s);
    project(roots.block_input, p.indexer_projection, stage.indexer);
    const IndexerStateView indexer_state =
        state_.indexer(mtp_active_ ? config_.full_attention_layers : layer);
    const ops::QsaIndexerKeyState key_state{indexer_state.raw_keys, indexer_state.raw_positions};
    Tensor attended = stage.attended.view({d, dimension(a.num_attention_heads), tokens});
    if (prefill) {
        const PagedKVLayerView cache = (mtp_active_ ? mtp_kv_ : kv_).layer_view(layer);
        ops::kv_cache_append(k, v, *positions_, cache, s);
        const ops::QsaIndexerBlockKeys blocks = (mtp_active_ ? mtp_kv_ : kv_).indexer_blocks(layer);
        ops::qsa_indexer_append(stage.indexer, *positions_, *rope_positions_, p.indexer_key_norm,
                                state_source_slot_, state_destination_slot_, key_state, blocks, s);
        if (mtp_teacher_) { return; }
        ops::qsa_indexer_select(stage.indexer, *positions_, *rope_positions_, p.indexer_query_norm,
                                blocks, indexer_envelope_, work_, stage.selections, stage.counts,
                                s);
        ops::selected_block_attention(q, *positions_, stage.selections, stage.counts, cache,
                                      attended, s);
    } else {
        const auto* active_cache          = mtp_active_ ? mtp_batch_kv_ : batch_kv_;
        const PagedKVBatchLayerView cache = active_cache->batch_layer_view(layer);
        const Tensor& rows                = decode_->kv_table_rows;
        const auto width                  = decode_->width;
        const auto batch                  = tokens / width;
        const Tensor& sequence_rows       = width == 1 ? rows : decode_->sequence_rows;
        Tensor k4         = k.view({d, dimension(a.num_key_value_heads), width, batch});
        Tensor v4         = v.view({d, dimension(a.num_key_value_heads), width, batch});
        Tensor positions2 = positions_->view({width, batch});
        ops::kv_cache_append(k4, v4, positions2, decode_->valid_columns, sequence_rows, cache, s);
        const ops::QsaIndexerBlockKeys blocks = active_cache->batch_indexer_blocks(layer);
        Tensor indexer3 = stage.indexer.view({stage.indexer.ne[0], width, batch});
        ops::qsa_indexer_append(indexer3, positions2, *rope_positions_, sequence_rows,
                                decode_->state_source_slots, decode_->state_destination_slots,
                                decode_->valid_columns, p.indexer_key_norm, key_state, blocks, s);
        if (mtp_teacher_) { return; }
        ops::qsa_indexer_select(stage.indexer, *positions_, *rope_positions_, rows,
                                p.indexer_query_norm, blocks, indexer_envelope_, work_,
                                stage.selections, stage.counts, s);
        ops::selected_block_attention(q, *positions_, rows, stage.selections, stage.counts, cache,
                                      work_, attended, s);
    }
    ops::sigmoid_mul(gate, attended, s);
    project(stage.attended, p.output, roots.block_output);
}

void TextContext::gdn(const GdnParameters& p, std::uint32_t layer, workspace::LayerRoots& roots,
                      bool prefill, std::int32_t tokens) {
    cudaStream_t s                   = ctx_.stream;
    const auto& g                    = config_.gdn;
    auto scope                       = work_.scope();
    auto stage                       = workspace::gdn_stage(work_, config_, tokens, prefill);
    const std::int32_t qk_heads      = dimension(g.linear_num_key_heads);
    const std::int32_t value_heads   = dimension(g.linear_num_value_heads);
    const std::int32_t head_dim      = dimension(g.linear_value_head_dim);
    LinearAttentionStatePool& linear = state_.linear();
    if (prefill) {
        {
            auto projection_scope = work_.scope();
            ops::gdn_input_proj(roots.block_input, p.projection.weight, stage.qkv, stage.z,
                                p.projection.policy, work_, s);
        }
        const Tensor conv_in = linear.conv_slot(layer, state_source_slot_);
        Tensor conv_out      = linear.conv_slot(layer, state_destination_slot_);
        ops::causal_conv1d_silu_split(stage.qkv, p.convolution, conv_in, conv_out, stage.query,
                                      stage.key, stage.value, s);
    } else {
        const std::int32_t hidden = dimension(config_.hidden_size);
        const auto width          = decode_->width;
        const auto batch          = tokens / width;
        Tensor input3             = roots.block_input.view({hidden, width, batch});
        Tensor query3             = stage.query.view({stage.query.ne[0], width, batch});
        Tensor key3               = stage.key.view({stage.key.ne[0], width, batch});
        Tensor value3             = stage.value.view({stage.value.ne[0], width, batch});
        Tensor z3                 = stage.z.view({stage.z.ne[0], width, batch});
        Tensor conv_states        = linear.layer_view(layer).conv;
        auto projection_scope     = work_.scope();
        const auto& w             = p.projection.weight;
        // The fused route's overlap contract takes a disjoint span even when it needs no scratch.
        WorkspaceArena scratch(work_.alloc_bytes(std::max<std::size_t>(
            1, ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                   w.qtype, w.n, w.k, ops::LinearPolicy::A16Only, batch, width, width))));
        if (decode_->replay != nullptr) {
            auto records = decode_->replay->layer(static_cast<std::int32_t>(layer), batch);
            ops::gdn_input_proj_conv_record(input3, w, p.convolution, conv_states,
                                            decode_->valid_columns, decode_->state_source_slots,
                                            records.conv, query3, key3, value3, z3,
                                            ops::LinearPolicy::A16Only, scratch, s);
        } else {
            ops::gdn_input_proj_conv_snapshot(input3, w, p.convolution, conv_states,
                                              decode_->valid_columns, decode_->state_source_slots,
                                              decode_->state_destination_slots, query3, key3,
                                              value3, z3, ops::LinearPolicy::A16Only, scratch, s);
        }
    }
    {
        auto control_scope = work_.scope();
        ops::gdn_gating_proj(roots.block_input, p.control, p.a_log, p.dt_bias, work_, stage.g,
                             stage.beta, ctx_.execution_view());
    }
    const float scale = inverse_sqrt(g.linear_key_head_dim);
    if (prefill) {
        Tensor q              = stage.query.view({head_dim, qk_heads, tokens});
        Tensor k              = stage.key.view({head_dim, qk_heads, tokens});
        Tensor v              = stage.value.view({head_dim, value_heads, tokens});
        Tensor out            = stage.recurrent.view({head_dim, value_heads, tokens});
        const Tensor state_in = linear.recurrent_slot(layer, state_source_slot_);
        Tensor state_out      = linear.recurrent_slot(layer, state_destination_slot_);
        auto recurrence_scope = work_.scope();
        ops::gated_delta_net(q, k, v, stage.g, stage.beta, scale, true, work_, state_in, state_out,
                             out, ctx_.execution_view());
    } else {
        const auto width        = decode_->width;
        const auto batch        = tokens / width;
        Tensor q                = stage.query.view({head_dim, qk_heads, width, batch});
        Tensor k                = stage.key.view({head_dim, qk_heads, width, batch});
        Tensor v                = stage.value.view({head_dim, value_heads, width, batch});
        Tensor out              = stage.recurrent.view({head_dim, value_heads, width, batch});
        Tensor gate             = stage.g.view({value_heads, width, batch});
        Tensor beta             = stage.beta.view({value_heads, width, batch});
        Tensor recurrent_states = linear.layer_view(layer).recurrent;
        if (decode_->replay != nullptr) {
            auto records = decode_->replay->layer(static_cast<std::int32_t>(layer), batch);
            ops::gated_delta_net_replay_record(q, k, v, gate, beta, scale, recurrent_states,
                                               decode_->valid_columns, decode_->state_source_slots,
                                               records.key, records.value, records.gate, out, s);
        } else {
            ops::gated_delta_net_batch_update(q, k, v, gate, beta, scale, true, recurrent_states,
                                              decode_->state_source_slots,
                                              decode_->state_destination_slots, out, s);
        }
    }
    Tensor recurrent  = stage.recurrent.view({head_dim, value_heads * tokens});
    Tensor z          = stage.z.view({head_dim, value_heads * tokens});
    Tensor normalized = stage.normalized.view({head_dim, value_heads * tokens});
    ops::gated_rmsnorm_sigmoid(recurrent, p.norm, z, config_.rms_norm_eps, normalized, s);
    project(stage.normalized, p.output, roots.block_output);
}

void TextContext::run_layers(workspace::LayerRoots& roots, const Tensor& ple_codes,
                             const Tensor& ple_scales, bool prefill, std::int32_t tokens) {
    cudaStream_t s     = ctx_.stream;
    const auto& layers = parameters_.text.layers;
    for (std::uint32_t layer = 0; layer < layers.size(); ++layer) {
        const BlockParameters& block = layers[layer];
        try {
            if (layer == parameters_.text.ple.layer) {
                inject_ple(roots, ple_codes, ple_scales, prefill, tokens);
            }
            {
                auto scope = work_.scope();
                ops::hyper_connection_prepare(roots.hidden, block.attention_hyper.mixing,
                                              block.attention_hyper.block_inject, roots.block_input,
                                              roots.injection, work_, s);
            }
            const std::uint32_t compact = config_.compact_layer_indices[layer];
            if (const auto* a = std::get_if<AttentionParameters>(&block.mixer)) {
                attention(*a, compact, roots, prefill, tokens);
            } else {
                gdn(std::get<GdnParameters>(block.mixer), compact, roots, prefill, tokens);
            }
            ops::hyper_connection_inject(roots.block_output, roots.injection, roots.hidden, s);
            {
                auto scope = work_.scope();
                ops::hyper_connection_prepare(roots.hidden, block.mlp_hyper.mixing,
                                              block.mlp_hyper.block_inject, roots.block_input,
                                              roots.injection, work_, s);
            }
            {
                // The MoE requires its workspace disjoint from every operand, so it gets an
                // exact sub-arena rather than the arena that also holds the layer roots.
                auto scope = work_.scope();
                WorkspaceArena scratch(work_.alloc_bytes(ops::sparse_moe_workspace_capacity_bytes(
                    block.moe.gate_up.policy, block.moe.down.policy, tokens, tokens)));
                ops::sparse_moe(roots.block_input, block.moe, ops::SparseMoeEpilogue::Store,
                                roots.block_output, scratch, s);
            }
            ops::hyper_connection_inject(roots.block_output, roots.injection, roots.hidden, s);
        } catch (const std::exception& error) {
            throw std::runtime_error("text/layers/" + std::to_string(layer) +
                                     (prefill ? " prefill" : " decode") +
                                     " columns=" + std::to_string(tokens) + ": " + error.what());
        }
    }
}

void TextContext::project_streams(const Tensor& streams, Tensor& logits) {
    const std::int32_t count = streams.ne[1];
    auto scope               = work_.scope();
    Tensor mixed             = workspace::mixed_hidden(work_, config_, count);
    {
        auto mix_scope = work_.scope();
        ops::hyper_connection_mix(streams, parameters_.text.final_mixer, mixed, work_, ctx_.stream);
    }
    project(mixed, parameters_.text.output_head, logits);
}

void TextContext::mtp_forward(const Tensor& embedding, const Tensor& previous_streams,
                              const Tensor& positions, const Tensor& rope_positions,
                              Tensor& streams, Tensor* logits, const OrdinaryDecodeInputs* batch,
                              bool teacher) {
    if (!parameters_.mtp || (batch ? mtp_batch_kv_ == nullptr : !mtp_kv_.valid())) {
        throw std::logic_error("MTP execution has no selected weights or cache");
    }
    const auto& m     = *parameters_.mtp;
    const auto tokens = embedding.ne[1];
    require_shape(embedding, DType::BF16, {dimension(config_.hidden_size), tokens},
                  "MTP embedding");
    require_shape(previous_streams, DType::BF16, {dimension(config_.stream_width()), tokens},
                  "MTP previous streams");
    require_shape(streams, DType::BF16, {dimension(config_.stream_width()), tokens}, "MTP streams");
    const auto* saved_positions = positions_;
    const auto* saved_rope      = rope_positions_;
    const auto* saved_decode    = decode_;
    positions_                  = &positions;
    rope_positions_             = &rope_positions;
    decode_                     = batch;
    if (batch != nullptr) { indexer_envelope_ = batch->indexer; }
    mtp_active_        = true;
    mtp_teacher_       = teacher;
    const auto restore = [&] {
        positions_      = saved_positions;
        rope_positions_ = saved_rope;
        decode_         = saved_decode;
        mtp_active_ = mtp_teacher_ = false;
    };
    try {
        auto scope        = work_.scope();
        auto roots        = workspace::layer_roots(work_, config_, tokens, streams);
        const auto stream = ctx_.stream;
        {
            auto stem                  = work_.scope();
            Tensor norm_embedding      = workspace::mixed_hidden(work_, config_, tokens);
            Tensor projected_embedding = workspace::mixed_hidden(work_, config_, tokens);
            Tensor norm_hidden         = workspace::stream_hidden(work_, config_, tokens);
            Tensor projected_hidden    = workspace::stream_hidden(work_, config_, tokens);
            ops::rmsnorm(embedding, m.embedding_norm, config_.rms_norm_eps, true, norm_embedding,
                         stream);
            project(norm_embedding, m.embedding_projection, projected_embedding);
            ops::rmsnorm(previous_streams, m.hidden_norm, config_.rms_norm_eps, true, norm_hidden,
                         stream);
            Tensor flat_in  = norm_hidden.view({dimension(config_.hidden_size), 4 * tokens});
            Tensor flat_out = projected_hidden.view({dimension(config_.hidden_size), 4 * tokens});
            project(flat_in, m.hidden_projection, flat_out);
            ops::hyper_connection_expand(projected_embedding, roots.hidden, stream);
            ops::residual_add(projected_hidden, roots.hidden, stream);
        }
        const auto& block = m.layer;
        {
            auto scratch = work_.scope();
            ops::hyper_connection_prepare(roots.hidden, block.attention_hyper.mixing,
                                          block.attention_hyper.block_inject, roots.block_input,
                                          roots.injection, work_, stream);
        }
        attention(std::get<AttentionParameters>(block.mixer), 0, roots, batch == nullptr, tokens);
        if (!teacher) {
            ops::hyper_connection_inject(roots.block_output, roots.injection, roots.hidden, stream);
            {
                auto scratch = work_.scope();
                ops::hyper_connection_prepare(roots.hidden, block.mlp_hyper.mixing,
                                              block.mlp_hyper.block_inject, roots.block_input,
                                              roots.injection, work_, stream);
            }
            {
                auto stage = work_.scope();
                WorkspaceArena scratch(work_.alloc_bytes(ops::sparse_moe_workspace_capacity_bytes(
                    block.moe.gate_up.policy, block.moe.down.policy, tokens, tokens)));
                ops::sparse_moe(roots.block_input, block.moe, ops::SparseMoeEpilogue::Store,
                                roots.block_output, scratch, stream);
            }
            ops::hyper_connection_inject(roots.block_output, roots.injection, roots.hidden, stream);
            if (logits != nullptr) {
                Tensor mixed = workspace::mixed_hidden(work_, config_, tokens);
                {
                    auto scratch = work_.scope();
                    ops::hyper_connection_mix(roots.hidden, m.final_mixer, mixed, work_, stream);
                }
                project(mixed, m.output_head, *logits);
            }
        }
    } catch (...) {
        restore();
        throw;
    }
    restore();
}

void TextContext::run_prompt_readout(const Tensor& streams, std::int64_t begin,
                                     std::int32_t length) {
    cudaStream_t s               = ctx_.stream;
    const PromptReadout& readout = *prompt_readout_;
    const std::int32_t width     = dimension(config_.stream_width());
    const std::int32_t hidden    = dimension(config_.hidden_size);
    if (readout.feature_position && static_cast<std::int64_t>(*readout.feature_position) >= begin &&
        static_cast<std::int64_t>(*readout.feature_position) < begin + length) {
        const auto local = static_cast<std::int32_t>(*readout.feature_position - begin);
        auto scope       = work_.scope();
        Tensor mixed     = workspace::mixed_hidden(work_, config_, 1);
        {
            auto mix_scope = work_.scope();
            ops::hyper_connection_mix(streams.slice(1, local, 1), parameters_.text.final_mixer,
                                      mixed, work_, s);
        }
        CUDA_CHECK(cudaMemcpyAsync(readout.feature_host, mixed.data,
                                   static_cast<std::size_t>(hidden) * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToHost, s));
    }
    const auto lo = std::lower_bound(readout.positions.begin(), readout.positions.end(),
                                     static_cast<std::uint32_t>(begin));
    const auto hi =
        std::lower_bound(lo, readout.positions.end(), static_cast<std::uint32_t>(begin + length));
    if (lo == hi) { return; }
    const std::int32_t domain      = dimension(parameters_.model.resources().public_token_count);
    const std::size_t first        = static_cast<std::size_t>(lo - readout.positions.begin());
    const std::size_t count        = static_cast<std::size_t>(hi - lo);
    const std::int32_t tile_width  = io_.logits.ne[1];
    const std::size_t block_floats = readout.block_floats();
    const auto candidates          = static_cast<std::int32_t>(readout.candidates);
    for (std::size_t tile = 0; tile < count; tile += static_cast<std::size_t>(tile_width)) {
        const auto m = static_cast<std::int32_t>(
            std::min<std::size_t>(static_cast<std::size_t>(tile_width), count - tile));
        auto scope      = work_.scope();
        Tensor gathered = workspace::stream_hidden(work_, config_, m);
        for (std::int32_t j = 0; j < m; ++j) {
            const std::uint32_t position = readout.positions[first + tile + j];
            const auto local =
                static_cast<std::int32_t>(static_cast<std::int64_t>(position) - begin);
            CUDA_CHECK(cudaMemcpyAsync(gathered.slice(1, j, 1).data,
                                       streams.slice(1, local, 1).data,
                                       static_cast<std::size_t>(width) * sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToDevice, s));
        }
        Tensor logits = columns(io_.logits, m);
        project_streams(gathered, logits);
        for (std::int32_t j = 0; j < m; ++j) {
            const std::size_t index = first + tile + static_cast<std::size_t>(j);
            float* block            = readout.readout + index * block_floats;
            const Tensor column     = logits.slice(1, j, 1);
            const Tensor sampled(const_cast<std::int32_t*>(readout.next_ids) + index, DType::I32,
                                 {1});
            Tensor sampled_out(block, DType::FP32, {1, 2});
            std::optional<Tensor> candidates_out;
            if (candidates > 0) {
                candidates_out.emplace(block + 2, DType::FP32,
                                       std::initializer_list<std::int32_t>{1, candidates, 2});
            }
            ops::candidate_logprobs(
                column, domain, sampled, readout.candidate_ids ? &*readout.candidate_ids : nullptr,
                nullptr, sampled_out, candidates_out ? &*candidates_out : nullptr, s);
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(readout.host + first * block_floats,
                               readout.readout + first * block_floats,
                               count * block_floats * sizeof(float), cudaMemcpyDeviceToHost, s));
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end,
                                              const PreparedPromptData* prompt,
                                              VisionPrefillSession* vision) {
    runtime::ExecutionTimingRecorder timing;
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin || begin != text_kv_base_) {
        throw std::invalid_argument("text prefill chunk is outside the prompt or its cache base");
    }
    if (static_cast<std::uint64_t>(begin) + nominal_length >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("text prefill absolute position exceeds int32");
    }
    cudaStream_t s  = ctx_.stream;
    const auto base = static_cast<std::int32_t>(begin);
    std::int32_t len =
        static_cast<std::int32_t>(std::min<std::uint32_t>(prefill_chunk_, nominal_length));
    const std::int64_t split = prefill_split_frontier_;
    if (split > base && split < static_cast<std::int64_t>(base) + len) {
        len = static_cast<std::int32_t>(split - base);
    }
    work_.reset();
    VisionChunk vision_chunk;
    if (vision != nullptr) {
        vision_chunk = vision->prepare_chunk(begin, static_cast<std::uint32_t>(len));
        len          = static_cast<std::int32_t>(vision_chunk.length);
    }
    std::vector<std::int32_t> scatter_indices;
    std::int32_t visual_begin = 0;
    if (vision_chunk.control != nullptr) {
        const auto& scatter = vision_chunk.control->scatter_indices;
        const auto first    = std::lower_bound(scatter.begin(), scatter.end(), base);
        const auto last     = std::lower_bound(first, scatter.end(), base + len);
        visual_begin        = static_cast<std::int32_t>(first - scatter.begin());
        for (auto it = first; it != last; ++it) { scatter_indices.push_back(*it - base); }
    }
    if (prompt != nullptr && (prompt->token_ids.size() != full_ids.size() ||
                              prompt->positions.size() != 3 * full_ids.size())) {
        throw std::invalid_argument("prepared prompt positions do not cover the prefill ledger");
    }
    const bool is_last = finalize_at_end && static_cast<std::uint32_t>(len) == nominal_length;
    nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                  static_cast<std::uint64_t>(len));
    work_.reset();
    {
        auto controls = workspace::prefill_controls(work_, len);
        CUDA_CHECK(cudaMemcpyAsync(controls.ids.data, full_ids.data() + begin,
                                   static_cast<std::size_t>(len) * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, s));
        ops::fill_i32_positions(controls.positions, base, s);
        for (std::int32_t axis = 0; axis < 3; ++axis) {
            Tensor axis_positions = controls.rope_positions.slice(1, axis, 1).view({len});
            if (prompt != nullptr) {
                CUDA_CHECK(cudaMemcpyAsync(axis_positions.data,
                                           prompt->positions.data() +
                                               static_cast<std::size_t>(axis) * full_ids.size() +
                                               begin,
                                           static_cast<std::size_t>(len) * sizeof(std::int32_t),
                                           cudaMemcpyHostToDevice, s));
            } else {
                ops::fill_i32_positions(axis_positions, base + rope_delta_, s);
            }
        }
        const Tensor hidden = workspace::stream_hidden(work_, config_, len);
        auto roots          = workspace::layer_roots(work_, config_, len, hidden);
        auto ple_rows       = workspace::ple_rows(work_, config_, len);
        positions_          = &controls.positions;
        rope_positions_     = &controls.rope_positions;
        decode_             = nullptr;
        indexer_envelope_   = {.max_complete_blocks = static_cast<std::int32_t>(
                                 (static_cast<std::int64_t>(base) + len) /
                                 static_cast<std::int64_t>(config_.indexer.compress_ratio))};
        {
            auto scope      = work_.scope();
            Tensor embedded = workspace::mixed_hidden(work_, config_, len);
            ops::embedding(controls.ids, parameters_.text.token_embedding, embedded, s);
            if (!scatter_indices.empty()) {
                const auto count = static_cast<std::int32_t>(scatter_indices.size());
                Tensor indices   = work_.alloc(DType::I32, {count});
                CUDA_CHECK(cudaMemcpyAsync(indices.data, scatter_indices.data(), indices.bytes(),
                                           cudaMemcpyHostToDevice, s));
                ops::scatter(vision_chunk.embeddings.slice(1, visual_begin, count), indices,
                             embedded, s);
            }
            ops::hyper_connection_expand(embedded, roots.hidden, s);
        }
        // The PLE rows are gathered on the host while the device runs the blocks ahead of the
        // injection layer; the copy is ordered on the stream before the injection consumes it.
        const PleRowStaging staged = ple_.gather(full_ids, begin, static_cast<std::uint32_t>(len));
        CUDA_CHECK(cudaMemcpyAsync(ple_rows.codes.data, staged.codes, ple_rows.codes.bytes(),
                                   cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(ple_rows.scales.data, staged.scales, ple_rows.scales.bytes(),
                                   cudaMemcpyHostToDevice, s));
        run_layers(roots, ple_rows.codes, ple_rows.scales, true, len);

        if (mtp_kv_.valid()) {
            const std::int32_t offset = begin == 0 ? 1 : 0;
            const std::int32_t count  = len - offset;
            if (count > 0) {
                auto teacher    = work_.scope();
                Tensor embedded = workspace::mixed_hidden(work_, config_, len);
                ops::embedding(controls.ids, parameters_.mtp->token_embedding, embedded, s);
                if (!scatter_indices.empty()) {
                    Tensor indices = work_.alloc(
                        DType::I32, {static_cast<std::int32_t>(scatter_indices.size())});
                    CUDA_CHECK(cudaMemcpyAsync(indices.data, scatter_indices.data(),
                                               indices.bytes(), cudaMemcpyHostToDevice, s));
                    ops::scatter(vision_chunk.embeddings.slice(1, visual_begin, indices.ne[0]),
                                 indices, embedded, s);
                }
                Tensor previous  = workspace::stream_hidden(work_, config_, count);
                Tensor output    = workspace::stream_hidden(work_, config_, count);
                Tensor positions = work_.alloc(DType::I32, {count});
                Tensor rope      = work_.alloc(DType::I32, {count, 3});
                ops::fill_i32_positions(positions, base + offset - 1, s);
                const auto row_bytes =
                    static_cast<std::size_t>(config_.stream_width()) * sizeof(std::uint16_t);
                if (offset == 0) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        previous.data, state_.continuation_hidden_slot(state_source_slot_).data,
                        row_bytes, cudaMemcpyDeviceToDevice, s));
                }
                const auto chain = count - (offset == 0 ? 1 : 0);
                if (chain > 0) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        previous.slice(1, offset == 0 ? 1 : 0, chain).data, roots.hidden.data,
                        static_cast<std::size_t>(chain) * row_bytes, cudaMemcpyDeviceToDevice, s));
                }
                for (std::int32_t axis = 0; axis < 3; ++axis) {
                    auto* dst = static_cast<std::int32_t*>(rope.data) + axis * count;
                    if (offset == 0) {
                        CUDA_CHECK(cudaMemcpyAsync(
                            dst,
                            static_cast<const std::int32_t*>(
                                state_.continuation_positions_slot(state_source_slot_).data) +
                                axis,
                            sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
                    }
                    if (chain > 0) {
                        CUDA_CHECK(cudaMemcpyAsync(
                            dst + (offset == 0 ? 1 : 0),
                            static_cast<const std::int32_t*>(controls.rope_positions.data) +
                                axis * len,
                            static_cast<std::size_t>(chain) * sizeof(std::int32_t),
                            cudaMemcpyDeviceToDevice, s));
                    }
                }
                mtp_forward(embedded.slice(1, offset, count), previous, positions, rope, output,
                            nullptr, nullptr, true);
            }
        }
        for (std::int32_t axis = 0; axis < 3; ++axis) {
            CUDA_CHECK(cudaMemcpyAsync(
                static_cast<std::int32_t*>(
                    state_.continuation_positions_slot(state_destination_slot_).data) +
                    axis,
                static_cast<const std::int32_t*>(controls.rope_positions.data) + axis * len + len -
                    1,
                sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
        }

        if (prompt_readout_ != nullptr) { run_prompt_readout(roots.hidden, base, len); }
        if (prefill_hidden_.data != nullptr) {
            Tensor mixed = columns(prefill_hidden_, len);
            auto scope   = work_.scope();
            ops::hyper_connection_mix(roots.hidden, parameters_.text.final_mixer, mixed, work_, s);
        }
        const Tensor tail = roots.hidden.slice(1, len - 1, 1);
        CUDA_CHECK(cudaMemcpyAsync(io_.prefill_tail.data, tail.data, io_.prefill_tail.bytes(),
                                   cudaMemcpyDeviceToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(state_.continuation_hidden_slot(state_destination_slot_).data,
                                   tail.data, tail.bytes(), cudaMemcpyDeviceToDevice, s));
        if (split == static_cast<std::int64_t>(base) + len &&
            rewrite_checkpoint_hidden_output_ != nullptr) {
            require_shape(*rewrite_checkpoint_hidden_output_, DType::BF16,
                          {dimension(config_.stream_width())}, "rewrite checkpoint hidden");
            CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data, tail.data,
                                       tail.bytes(), cudaMemcpyDeviceToDevice, s));
        }
        if (is_last) {
            Tensor logits = columns(io_.logits, 1);
            project_streams(tail, logits);
            // The bonus token's absolute position keys the sampler RNG (prefill purpose).
            ops::set_i32_scalar(io_.pos, base + len, s);
            const std::int32_t domain = dimension(parameters_.model.resources().public_token_count);
            if (sampling_config_ != nullptr) {
                ops::sample(logits, io_.token, domain, sampling_config_, io_.pos,
                            ops::kSamplePurposePrefill, work_, s);
            } else {
                ops::argmax(logits, io_.token, domain, s);
            }
            if (first_token_logits_host_ != nullptr) {
                CUDA_CHECK(cudaMemcpyAsync(first_token_logits_host_, logits.data,
                                           static_cast<std::size_t>(domain) * sizeof(std::uint16_t),
                                           cudaMemcpyDeviceToHost, s));
            }
            if (first_token_readout_ != nullptr) {
                enqueue_first_token_readout(*first_token_readout_, logits, io_.token, domain, s);
            }
        }
        positions_      = nullptr;
        rope_positions_ = nullptr;
    }
    prefill_split_frontier_ = -1;
    timing.begin_wait();
    ctx_.synchronize();
    timing.end_wait();
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(len),
                              .finalized        = is_last,
                              .timing           = timing.finish()};
}

void TextContext::ordinary_decode_batch(const OrdinaryDecodeInputs& inputs, Tensor& hidden,
                                        Tensor& logits) {
    if (inputs.width != 1 || inputs.replay != nullptr) {
        throw std::invalid_argument("ordinary decode requires width one without replay records");
    }
    decode_batch(inputs, hidden, logits);
}

void TextContext::verify_batch(const OrdinaryDecodeInputs& inputs, Tensor& hidden, Tensor& logits) {
    if (inputs.width < 2 || inputs.width > 6 || inputs.replay == nullptr) {
        throw std::invalid_argument("MTP verification requires width 2..6 and replay records");
    }
    decode_batch(inputs, hidden, logits);
}

void TextContext::decode_batch(const OrdinaryDecodeInputs& inputs, Tensor& hidden, Tensor& logits) {
    const std::int32_t batch = inputs.ids.ne[0];
    if (batch <= 0 || batch % inputs.width != 0 ||
        batch / inputs.width > static_cast<std::int32_t>(kMaximumConcurrency) ||
        batch_kv_ == nullptr) {
        throw std::invalid_argument("ordinary decode batch is outside [1,8] or has no cache");
    }
    const std::int32_t rows = dimension(std::uint64_t(config_.ple.heads()) * std::uint64_t(batch));
    require_shape(inputs.ids, DType::I32, {batch}, "ordinary decode ids");
    require_shape(inputs.cache_positions, DType::I32, {batch}, "ordinary decode positions");
    require_shape(inputs.rope_positions, DType::I32, {batch, 3}, "ordinary decode RoPE positions");
    require_shape(inputs.kv_table_rows, DType::I32, {batch}, "ordinary decode KV rows");
    require_shape(inputs.state_source_slots, DType::I32, {batch / inputs.width},
                  "ordinary decode source slots");
    require_shape(inputs.state_destination_slots, DType::I32, {batch / inputs.width},
                  "ordinary decode destination slots");
    require_shape(inputs.ple_codes, DType::U8, {dimension(config_.ple.head_width() / 2), rows},
                  "ordinary decode PLE codes");
    require_shape(inputs.ple_scales, DType::FP16, {dimension(config_.ple.head_width() / 16), rows},
                  "ordinary decode PLE scales");
    require_shape(hidden, DType::BF16, {dimension(config_.stream_width()), batch},
                  "ordinary decode hidden");
    require_shape(logits, DType::BF16, {dimension(config_.vocab_size), batch},
                  "ordinary decode logits");

    cudaStream_t s = ctx_.stream;
    work_.reset();
    positions_        = &inputs.cache_positions;
    rope_positions_   = &inputs.rope_positions;
    decode_           = &inputs;
    indexer_envelope_ = inputs.indexer;
    {
        auto roots = workspace::layer_roots(work_, config_, batch, hidden);
        {
            auto scope      = work_.scope();
            Tensor embedded = workspace::mixed_hidden(work_, config_, batch);
            ops::embedding(inputs.ids, parameters_.text.token_embedding, embedded, s);
            ops::hyper_connection_expand(embedded, roots.hidden, s);
        }
        run_layers(roots, inputs.ple_codes, inputs.ple_scales, false, batch);
        project_streams(roots.hidden, logits);
    }
    positions_      = nullptr;
    rope_positions_ = nullptr;
    decode_         = nullptr;
    work_.reset();
}

} // namespace ninfer::models::qwen4_exp::execution
