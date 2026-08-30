#include "targets/qwen3_8_flash_next/impl/text_decode.h"

#include "core/device.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"

#include "core/layout.h"
#include "targets/qwen3_8_flash_next/impl/gdn.h"
#include "targets/qwen3_8_flash_next/impl/hyper_connection.h"
#include "targets/qwen3_8_flash_next/impl/moe.h"
#include "targets/qwen3_8_flash_next/impl/ple_decode.h"
#include "targets/qwen3_8_flash_next/impl/qsa_attention.h"
#include "targets/qwen3_8_flash_next/impl/qsa_indexer.h"
#include "targets/qwen3_8_flash_next/impl/qsa_indexer_kernels.h"
#include "targets/qwen3_8_flash_next/impl/text_decode_kernels.h"
#include "targets/qwen3_8_flash_next/impl/text_decode_workspace.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool exact_tensor(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1 = 1,
                  std::int32_t n2 = 1, std::int32_t n3 = 1) {
    return tensor.dtype == dtype && tensor.ne[0] == n0 && tensor.ne[1] == n1 &&
           tensor.ne[2] == n2 && tensor.ne[3] == n3 && tensor.is_contiguous() &&
           aligned_to(tensor.data, 16);
}

bool exact_bf16_weight(const Weight& weight, std::int32_t rows, std::int32_t columns) {
    return weight.qtype == QType::BF16_CTRL && weight.layout == QuantLayout::Contiguous &&
           weight.n == rows && weight.k == columns && weight.ndim == 2 && weight.shape[0] == rows &&
           weight.shape[1] == columns && weight.padded_shape[0] == rows &&
           weight.padded_shape[1] == columns && weight.qdata == weight.payload &&
           weight.payload_bytes >= static_cast<std::uint64_t>(rows) * columns * 2 &&
           aligned_to(weight.qdata, 16);
}

} // namespace

void validate_flash_next_decode_state(const FlashNextDecodeStateView& state,
                                      std::int32_t state_slots) {
    if (state_slots <= 0) {
        throw std::invalid_argument("Flash-Next state validation requires state_slots > 0");
    }
    if (!exact_tensor(state.ple_convolution_states, DType::BF16, 10'240, 9, state_slots)) {
        throw std::invalid_argument("Flash-Next PLE convolution state view is invalid");
    }
    for (std::size_t i = 0; i < kGdnLayers; ++i) {
        if (!exact_tensor(state.gdn_convolution_states[i], DType::BF16, 10'240, 3, state_slots) ||
            !exact_tensor(state.gdn_ssm_states[i], DType::FP32, 128, 128, 48, state_slots)) {
            throw std::invalid_argument("Flash-Next GDN state view is invalid");
        }
    }
    for (std::size_t i = 0; i < kFullAttentionLayers; ++i) {
        const auto& idx = state.qsa_indexer_caches[i];
        if (idx.block_keys.dtype != DType::BF16 || idx.block_keys.ne[0] != 128 ||
            idx.block_keys.ne[1] != 64 || idx.block_keys.ne[2] <= 0 || idx.block_keys.ne[3] != 1 ||
            !idx.block_keys.is_contiguous() || !aligned_to(idx.block_keys.data, 16) ||
            idx.block_tables.dtype != DType::I32 || idx.block_tables.ne[0] <= 0 ||
            idx.block_tables.ne[1] <= 0 || idx.block_tables.ne[2] != 1 ||
            idx.block_tables.ne[3] != 1 || !idx.block_tables.is_contiguous() ||
            !aligned_to(idx.block_tables.data, 16) ||
            !exact_tensor(idx.raw_keys, DType::BF16, 128, 4, state_slots) ||
            !exact_tensor(idx.raw_positions, DType::I32, 3, 4, state_slots)) {
            throw std::invalid_argument("Flash-Next QSA indexer cache view is invalid");
        }
        const auto& att = state.qsa_attention_caches[i];
        if (att.key_pages.dtype != DType::BF16 || att.key_pages.ne[0] != 256 ||
            att.key_pages.ne[1] != 64 || att.key_pages.ne[2] != 2 || att.key_pages.ne[3] <= 0 ||
            !att.key_pages.is_contiguous() || !aligned_to(att.key_pages.data, 16) ||
            att.value_pages.dtype != DType::BF16 || att.value_pages.ne[0] != 256 ||
            att.value_pages.ne[1] != 64 || att.value_pages.ne[2] != 2 ||
            att.value_pages.ne[3] <= 0 || !att.value_pages.is_contiguous() ||
            !aligned_to(att.value_pages.data, 16) || att.block_tables.dtype != DType::I32 ||
            att.block_tables.ne[0] <= 0 || att.block_tables.ne[1] <= 0 ||
            att.block_tables.ne[2] != 1 || att.block_tables.ne[3] != 1 ||
            !att.block_tables.is_contiguous() || !aligned_to(att.block_tables.data, 16)) {
            throw std::invalid_argument("Flash-Next QSA attention cache view is invalid");
        }
    }
}

std::size_t flash_next_text_decode_workspace_capacity_bytes(std::int32_t maximum_blocks,
                                                            std::int32_t batch) {
    if (maximum_blocks <= 0 || maximum_blocks > 65'536 || batch <= 0 || batch > 8) {
        throw std::invalid_argument("Flash-Next text decode received an invalid envelope");
    }
    WorkspaceLayoutBuilder layout;
    layout.alloc(DType::BF16, {2'560, batch}, 256);
    (void)allocate_flash_next_text_decode_workspace(layout, batch);
    const std::size_t sort_temp = flash_next_qsa_indexer_sort_temp_bytes(maximum_blocks, batch);
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_ple_workspace(layout, batch);
    }
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_gdn_workspace(layout, batch);
    }
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_qsa_indexer_workspace(layout, maximum_blocks, batch, sort_temp);
    }
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_qsa_attention_workspace(layout, batch);
    }
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_moe_workspace(layout, batch);
    }
    return layout.peak_bytes(256);
}

std::size_t flash_next_text_prefill_workspace_capacity_bytes(std::int32_t maximum_blocks,
                                                             std::int32_t tokens) {
    if (maximum_blocks <= 0 || maximum_blocks > 65'536 || tokens <= 0) {
        throw std::invalid_argument("Flash-Next text prefill received an invalid envelope");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_flash_next_prefill_chunk_staging(layout, tokens);
    (void)allocate_flash_next_text_decode_workspace(layout, tokens);
    const std::size_t sort_temp = flash_next_qsa_indexer_sort_temp_bytes(maximum_blocks, 1);
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_ple_workspace(layout, tokens);
    }
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_gdn_workspace(layout, tokens);
        const std::size_t gdn_op_ws =
            ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, tokens, tokens);
        layout.alloc_bytes(gdn_op_ws, 256);
    }
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_qsa_indexer_workspace(layout, maximum_blocks, 1, sort_temp);
    }
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_qsa_attention_workspace(layout, 1);
    }
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_moe_workspace(layout, tokens);
    }
    return layout.peak_bytes(256);
}

void flash_next_text_decode_core(const TextModelView& model, const Tensor& embedding,
                                 const Tensor& token_indices, const Tensor& mrope_positions,
                                 const Tensor& table_rows, const Tensor& source_slots,
                                 const Tensor& destination_slots,
                                 const Tensor& gathered_ple_embedding, std::int32_t maximum_blocks,
                                 std::int32_t active_blocks, FlashNextDecodeStateView state,
                                 WorkspaceArena& workspace, Tensor& final_hidden, Tensor& logits,
                                 cudaStream_t stream, const FlashNextDecodeStateSink* sink) {
    const std::int32_t batch       = embedding.ne[1];
    const std::int32_t state_slots = state.ple_convolution_states.ne[2];
    if (batch <= 0 || batch > 8 || maximum_blocks <= 0 || maximum_blocks > 65'536 ||
        active_blocks < 0 || active_blocks > maximum_blocks ||
        !exact_tensor(embedding, DType::BF16, 2'560, batch) ||
        !exact_tensor(token_indices, DType::I32, batch) ||
        !exact_tensor(mrope_positions, DType::I32, batch, 3) ||
        !exact_tensor(table_rows, DType::I32, batch) ||
        !exact_tensor(source_slots, DType::I32, batch) ||
        !exact_tensor(destination_slots, DType::I32, batch) ||
        !exact_tensor(gathered_ple_embedding, DType::BF16, 2'560, batch) ||
        !exact_tensor(final_hidden, DType::BF16, 2'560, batch) ||
        !exact_tensor(logits, DType::BF16, 248'320, batch) ||
        !exact_bf16_weight(model.output_head, 248'320, 2'560) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next text decode core received an invalid input view");
    }
    validate_flash_next_decode_state(state, state_slots);

    auto emit_state = [&](std::string_view name, const Tensor& tensor) {
        if (sink && sink->on_state) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            sink->on_state(name, tensor);
        }
    };

    emit_state("embedding", embedding);

    const auto round_scope = workspace.scope();
    FlashNextTextDecodeWorkspace round_ws =
        allocate_flash_next_text_decode_workspace(workspace, batch);

    // 1. Repeat embedding into 4 hyperconnection streams
    repeat_embedding_to_hyper_streams(embedding, round_ws.hyper_hidden, stream);
    emit_state("hyper_init", round_ws.hyper_hidden);

    // 2. 48-layer execution loop
    for (std::size_t layer = 0; layer < 48; ++layer) {
        char prefix_buf[32];
        std::snprintf(prefix_buf, sizeof(prefix_buf), "L%02zu_", layer);
        const std::string prefix(prefix_buf);

        // At layer 1: evaluate PLE neural injection and add residual
        if (layer == 1) {
            emit_state("ple_gathered", gathered_ple_embedding);
            flash_next_ple_decode(round_ws.hyper_hidden, gathered_ple_embedding, model.ple,
                                  source_slots, destination_slots, state.ple_convolution_states,
                                  workspace, round_ws.ple_injection, stream);
            emit_state("ple_injection", round_ws.ple_injection);
            ops::residual_add(round_ws.ple_injection, round_ws.hyper_hidden, stream);
            emit_state("hyper_after_ple", round_ws.hyper_hidden);
        }

        // Attention hyper prepare -> block_input [2560, B]
        flash_next_hyper_prepare(round_ws.hyper_hidden, model.layers[layer].attention_hyper,
                                 round_ws.hyper_scratch, round_ws.block_input, stream);
        emit_state(prefix + "attn_block_input", round_ws.block_input);

        // Execute QSA or GDN attention
        if (is_qsa_layer(layer)) {
            const std::size_t qsa_idx = qsa_ordinal(layer);
            flash_next_qsa_indexer_decode(
                round_ws.block_input, model.full_attention[qsa_idx], token_indices, mrope_positions,
                table_rows, source_slots, destination_slots, state.qsa_indexer_caches[qsa_idx],
                maximum_blocks, active_blocks, workspace, round_ws.selected_blocks,
                round_ws.selected_counts, stream);
            emit_state(prefix + "selected_counts", round_ws.selected_counts);
            flash_next_qsa_attention_decode(
                round_ws.block_input, model.full_attention[qsa_idx], token_indices, mrope_positions,
                table_rows, round_ws.selected_blocks, round_ws.selected_counts,
                state.qsa_attention_caches[qsa_idx], workspace, round_ws.block_output, stream);
        } else {
            const std::size_t gdn_idx = gdn_ordinal(layer);
            flash_next_gdn_decode(round_ws.block_input, model.gdn[gdn_idx], source_slots,
                                  destination_slots, state.gdn_convolution_states[gdn_idx],
                                  state.gdn_ssm_states[gdn_idx], workspace, round_ws.block_output,
                                  stream);
        }
        emit_state(prefix + "attn_block_output", round_ws.block_output);

        // Attention hyper inject
        flash_next_hyper_inject(round_ws.block_output, round_ws.hyper_scratch.injection,
                                round_ws.hyper_hidden, stream);
        emit_state(prefix + "hyper_after_attn", round_ws.hyper_hidden);

        // MLP hyper prepare -> block_input [2560, B]
        flash_next_hyper_prepare(round_ws.hyper_hidden, model.layers[layer].mlp_hyper,
                                 round_ws.hyper_scratch, round_ws.block_input, stream);
        emit_state(prefix + "mlp_block_input", round_ws.block_input);

        // MoE
        flash_next_moe(round_ws.block_input, model.layers[layer].moe, round_ws.block_output,
                       workspace, stream);
        emit_state(prefix + "mlp_block_output", round_ws.block_output);

        // MLP hyper inject
        flash_next_hyper_inject(round_ws.block_output, round_ws.hyper_scratch.injection,
                                round_ws.hyper_hidden, stream);
        emit_state(prefix + "hyper_after_mlp", round_ws.hyper_hidden);
    }

    // 3. Final hyper mixer -> final_hidden [2560, B]
    flash_next_hyper_mix(round_ws.hyper_hidden, model.final_mixer, round_ws.hyper_scratch,
                         final_hidden, stream);
    emit_state("final_hidden", final_hidden);

    // 4. Output head linear projection -> logits [248320, B]
    ops::linear(final_hidden, model.output_head, logits, ops::LinearPolicy::A16Only, workspace,
                stream);
    emit_state("logits", logits);
}

void flash_next_text_decode(const TextModelView& model, const Tensor& token_ids,
                            const Tensor& token_indices, const Tensor& mrope_positions,
                            const Tensor& table_rows, const Tensor& source_slots,
                            const Tensor& destination_slots, const Tensor& gathered_ple_embedding,
                            std::int32_t maximum_blocks, std::int32_t active_blocks,
                            FlashNextDecodeStateView state, WorkspaceArena& workspace,
                            Tensor& final_hidden, Tensor& logits, cudaStream_t stream,
                            const FlashNextDecodeStateSink* sink) {
    const std::int32_t batch = token_ids.ne[0];
    if (batch <= 0 || batch > 8 || !exact_tensor(token_ids, DType::I32, batch) ||
        !exact_bf16_weight(model.token_embedding, 248'320, 2'560)) {
        throw std::invalid_argument("Flash-Next text decode token embedding input is invalid");
    }
    const auto round_scope = workspace.scope();
    Tensor embedding       = workspace.alloc(DType::BF16, {2'560, batch}, 256);
    ops::embedding(token_ids, model.token_embedding, embedding, stream);
    flash_next_text_decode_core(model, embedding, token_indices, mrope_positions, table_rows,
                                source_slots, destination_slots, gathered_ple_embedding,
                                maximum_blocks, active_blocks, state, workspace, final_hidden,
                                logits, stream, sink);
}

void flash_next_text_prefill_chunk(const TextModelView& model, const Tensor& embedding,
                                   const Tensor& token_indices, const Tensor& mrope_positions,
                                   std::int32_t table_row, std::int32_t source_slot,
                                   std::int32_t destination_slot,
                                   const Tensor& gathered_ple_embedding, std::int32_t maximum_blocks,
                                   FlashNextDecodeStateView state, WorkspaceArena& workspace,
                                   Tensor& final_hidden, Tensor& logits, cudaStream_t stream,
                                   const FlashNextDecodeStateSink* sink) {
    const std::int32_t tokens      = embedding.ne[1];
    const std::int32_t state_slots = state.ple_convolution_states.ne[2];
    if (tokens <= 0 || maximum_blocks <= 0 || maximum_blocks > 65'536 || table_row < 0 ||
        source_slot < 0 || source_slot >= state_slots || destination_slot < 0 ||
        destination_slot >= state_slots ||
        !exact_tensor(embedding, DType::BF16, 2'560, tokens) ||
        !exact_tensor(token_indices, DType::I32, tokens) ||
        mrope_positions.dtype != DType::I32 || !mrope_positions.is_contiguous() ||
        !aligned_to(mrope_positions.data, 16) ||
        !((mrope_positions.ne[0] == 3 && mrope_positions.ne[1] == tokens) ||
          (mrope_positions.ne[0] == tokens && mrope_positions.ne[1] == 3) ||
          (mrope_positions.ne[0] == 1 && mrope_positions.ne[1] == 3 && mrope_positions.ne[2] == tokens)) ||
        !exact_tensor(gathered_ple_embedding, DType::BF16, 2'560, tokens) ||
        !exact_tensor(final_hidden, DType::BF16, 2'560, 1) ||
        !exact_tensor(logits, DType::BF16, 248'320, 1) ||
        !exact_bf16_weight(model.output_head, 248'320, 2'560) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next text prefill chunk received an invalid input view");
    }
    validate_flash_next_decode_state(state, state_slots);

    auto emit_state = [&](std::string_view name, const Tensor& tensor) {
        if (sink && sink->on_state) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            sink->on_state(name, tensor);
        }
    };

    emit_state("embedding", embedding);

    const auto round_scope = workspace.scope();
    FlashNextTextDecodeWorkspace round_ws =
        allocate_flash_next_text_decode_workspace(workspace, tokens);

    // 1. Repeat embedding into 4 hyperconnection streams
    repeat_embedding_to_hyper_streams(embedding, round_ws.hyper_hidden, stream);
    emit_state("hyper_init", round_ws.hyper_hidden);

    // 2. 48-layer execution loop
    for (std::size_t layer = 0; layer < 48; ++layer) {
        char prefix_buf[32];
        std::snprintf(prefix_buf, sizeof(prefix_buf), "L%02zu_", layer);
        const std::string prefix(prefix_buf);

        // At layer 1: evaluate PLE neural injection and add residual
        if (layer == 1) {
            emit_state("ple_gathered", gathered_ple_embedding);
            flash_next_ple_prefill_chunk(round_ws.hyper_hidden, gathered_ple_embedding, model.ple,
                                         source_slot, destination_slot,
                                         state.ple_convolution_states, workspace,
                                         round_ws.ple_injection, stream);
            emit_state("ple_injection", round_ws.ple_injection);
            ops::residual_add(round_ws.ple_injection, round_ws.hyper_hidden, stream);
            emit_state("hyper_after_ple", round_ws.hyper_hidden);
        }

        // Attention hyper prepare -> block_input [2560, T]
        flash_next_hyper_prepare(round_ws.hyper_hidden, model.layers[layer].attention_hyper,
                                 round_ws.hyper_scratch, round_ws.block_input, stream);
        emit_state(prefix + "attn_block_input", round_ws.block_input);

        // Execute QSA or GDN attention
        if (is_qsa_layer(layer)) {
            const std::size_t qsa_idx = qsa_ordinal(layer);
            for (std::int32_t t = 0; t < tokens; ++t) {
                const std::int32_t qsa_src = (t == 0) ? source_slot : destination_slot;
                const std::int32_t qsa_dst = destination_slot;

                set_qsa_step_metadata(token_indices, mrope_positions, t, table_row, qsa_src,
                                      qsa_dst, round_ws.qsa_token_indices,
                                      round_ws.qsa_mrope_positions, round_ws.qsa_table_rows,
                                      round_ws.qsa_source_slots, round_ws.qsa_destination_slots,
                                      stream);

                Tensor col_input  = round_ws.block_input.slice(1, t, 1);
                Tensor col_output = round_ws.block_output.slice(1, t, 1);

                flash_next_qsa_indexer_decode(
                    col_input, model.full_attention[qsa_idx], round_ws.qsa_token_indices,
                    round_ws.qsa_mrope_positions, round_ws.qsa_table_rows,
                    round_ws.qsa_source_slots, round_ws.qsa_destination_slots,
                    state.qsa_indexer_caches[qsa_idx], maximum_blocks, maximum_blocks,
                    workspace, round_ws.qsa_selected_blocks, round_ws.qsa_selected_counts,
                    stream);

                flash_next_qsa_attention_decode(
                    col_input, model.full_attention[qsa_idx], round_ws.qsa_token_indices,
                    round_ws.qsa_mrope_positions, round_ws.qsa_table_rows,
                    round_ws.qsa_selected_blocks, round_ws.qsa_selected_counts,
                    state.qsa_attention_caches[qsa_idx], workspace, col_output, stream);
            }
        } else {
            const std::size_t gdn_idx = gdn_ordinal(layer);
            flash_next_gdn_prefill_chunk(round_ws.block_input, model.gdn[gdn_idx], source_slot,
                                         destination_slot, state.gdn_convolution_states[gdn_idx],
                                         state.gdn_ssm_states[gdn_idx], workspace,
                                         round_ws.block_output, stream);
        }
        emit_state(prefix + "attn_block_output", round_ws.block_output);

        // Attention hyper inject
        flash_next_hyper_inject(round_ws.block_output, round_ws.hyper_scratch.injection,
                                round_ws.hyper_hidden, stream);
        emit_state(prefix + "hyper_after_attn", round_ws.hyper_hidden);

        // MLP hyper prepare -> block_input [2560, T]
        flash_next_hyper_prepare(round_ws.hyper_hidden, model.layers[layer].mlp_hyper,
                                 round_ws.hyper_scratch, round_ws.block_input, stream);
        emit_state(prefix + "mlp_block_input", round_ws.block_input);

        // MoE
        flash_next_moe(round_ws.block_input, model.layers[layer].moe, round_ws.block_output,
                       workspace, stream);
        emit_state(prefix + "mlp_block_output", round_ws.block_output);

        // MLP hyper inject
        flash_next_hyper_inject(round_ws.block_output, round_ws.hyper_scratch.injection,
                                round_ws.hyper_hidden, stream);
        emit_state(prefix + "hyper_after_mlp", round_ws.hyper_hidden);
    }

    // 3. Final hyper mixer on last token only -> final_hidden [2560, 1]
    Tensor last_hidden = round_ws.hyper_hidden.slice(1, tokens - 1, 1);
    flash_next_hyper_mix(last_hidden, model.final_mixer, round_ws.single_token_hyper_scratch,
                         final_hidden, stream);
    emit_state("final_hidden", final_hidden);

    // 4. Output head linear projection -> logits [248320, 1]
    ops::linear(final_hidden, model.output_head, logits, ops::LinearPolicy::A16Only, workspace,
                stream);
    emit_state("logits", logits);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
