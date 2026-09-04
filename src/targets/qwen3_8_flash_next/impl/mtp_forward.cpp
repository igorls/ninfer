#include "targets/qwen3_8_flash_next/impl/mtp_forward.h"

#include "core/device.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/speculative_round.h"

#include "core/layout.h"
#include "targets/qwen3_8_flash_next/impl/hyper_connection.h"
#include "targets/qwen3_8_flash_next/impl/hyper_workspace.h"
#include "targets/qwen3_8_flash_next/impl/moe.h"
#include "targets/qwen3_8_flash_next/impl/moe_workspace.h"
#include "targets/qwen3_8_flash_next/impl/mtp_forward_kernels.h"
#include "targets/qwen3_8_flash_next/impl/qsa_attention.h"
#include "targets/qwen3_8_flash_next/impl/qsa_attention_workspace.h"

#include <iostream>
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

} // namespace

std::size_t flash_next_mtp_workspace_capacity_bytes(std::int32_t batch) {
    if (batch <= 0 || batch > 8) {
        throw std::invalid_argument("Flash-Next MTP received an invalid batch size");
    }
    WorkspaceLayoutBuilder layout;
    // Primary MTP activation tensors
    (void)layout.alloc(DType::BF16, {2'560, batch}, 256);  // emb_norm
    (void)layout.alloc(DType::BF16, {2'560, batch}, 256);  // emb_proj
    (void)layout.alloc(DType::BF16, {2'560, batch}, 256);  // hid_mix
    (void)layout.alloc(DType::BF16, {2'560, batch}, 256);  // hid_proj
    (void)layout.alloc(DType::BF16, {2'560, batch}, 256);  // trunk_sum
    (void)layout.alloc(DType::BF16, {10'240, batch}, 256); // mtp_hyper_hidden
    (void)layout.alloc(DType::BF16, {2'560, batch}, 256);  // attn_in
    (void)layout.alloc(DType::BF16, {2'560, batch}, 256);  // attn_out
    (void)layout.alloc(DType::BF16, {2'560, batch}, 256);  // mlp_in
    (void)layout.alloc(DType::BF16, {2'560, batch}, 256);  // mlp_out
    (void)layout.alloc(DType::BF16, {2'560, batch}, 256);  // mtp_final_hidden

    // Hyper workspace
    (void)allocate_flash_next_hyper_workspace(layout, batch);

    // Attention workspace
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_qsa_attention_workspace(layout, batch);
        const std::size_t qgkv_ws = ops::linear_workspace_capacity_bytes(
            QType::BF16_CTRL, 13'312, 2'560, ops::LinearPolicy::A16Only, 1, batch);
        const std::size_t out_ws = ops::linear_workspace_capacity_bytes(
            QType::BF16_CTRL, 2'560, 6'144, ops::LinearPolicy::A16Only, 1, batch);
        (void)layout.alloc_bytes(std::max(qgkv_ws, out_ws), 256);
    }

    // MoE workspace
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_moe_workspace(layout, batch);
    }

    // Head linear workspace
    {
        auto scope = layout.scope();
        const std::size_t head_ws = ops::linear_workspace_capacity_bytes(
            QType::BF16_CTRL, 248'320, 2'560, ops::LinearPolicy::A16Only, 1, batch);
        (void)layout.alloc_bytes(head_ws, 256);
    }

    return layout.peak_bytes(256);
}

void flash_next_mtp_step(const TextModelView& model, const Tensor& input_embedding,
                         const Tensor& backbone_hyper_hidden, const Tensor& token_indices,
                         const Tensor& mrope_positions, const Tensor& table_rows,
                         const Tensor& selected_blocks, const Tensor& selected_counts,
                         QsaAttentionCacheView mtp_cache, WorkspaceArena& workspace,
                         Tensor& draft_logits, Tensor& draft_tokens, cudaStream_t stream,
                         const FlashNextDecodeStateSink* sink, Tensor* out_hyper_hidden) {
    if (!model.mtp.has_value()) {
        throw std::invalid_argument("Flash-Next MTP step called but MTP weights are not materialized");
    }
    const auto& mtp = *model.mtp;
    const std::int32_t batch = input_embedding.ne[1];

    if (!exact_tensor(input_embedding, DType::BF16, 2'560, batch) || batch < 1 || batch > 8 ||
        !exact_tensor(backbone_hyper_hidden, DType::BF16, 10'240, batch) ||
        !exact_tensor(token_indices, DType::I32, batch) ||
        !exact_tensor(mrope_positions, DType::I32, batch, 3) ||
        !exact_tensor(table_rows, DType::I32, batch) ||
        !exact_tensor(selected_blocks, DType::I32, 512, batch) ||
        !exact_tensor(selected_counts, DType::I32, batch) ||
        !exact_tensor(draft_logits, DType::BF16,
                      model.proposal.has_value() ? model.proposal->head.n : 248'320, batch) ||
        !exact_tensor(draft_tokens, DType::I32, batch) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next MTP step received invalid tensor views");
    }

    auto emit_state = [&](std::string_view name, const Tensor& tensor) {
        if (sink != nullptr && sink->on_state) {
            sink->on_state(name, tensor);
        }
    };

    const auto scope = workspace.scope();

    // Allocate stage tensors
    Tensor emb_norm         = workspace.alloc(DType::BF16, {2'560, batch}, 256);
    Tensor emb_proj         = workspace.alloc(DType::BF16, {2'560, batch}, 256);
    Tensor hid_mix          = workspace.alloc(DType::BF16, {2'560, batch}, 256);
    Tensor hid_proj         = workspace.alloc(DType::BF16, {2'560, batch}, 256);
    Tensor trunk_sum        = workspace.alloc(DType::BF16, {2'560, batch}, 256);
    Tensor mtp_hyper_hidden = workspace.alloc(DType::BF16, {10'240, batch}, 256);
    Tensor attn_in          = workspace.alloc(DType::BF16, {2'560, batch}, 256);
    Tensor attn_out         = workspace.alloc(DType::BF16, {2'560, batch}, 256);
    Tensor mlp_in           = workspace.alloc(DType::BF16, {2'560, batch}, 256);
    Tensor mlp_out          = workspace.alloc(DType::BF16, {2'560, batch}, 256);
    Tensor mtp_final_hidden = workspace.alloc(DType::BF16, {2'560, batch}, 256);

    FlashNextHyperWorkspace hyper_scratch = allocate_flash_next_hyper_workspace(workspace, batch);

    // 1. Stem: Embedding projection
    ops::rmsnorm(input_embedding, mtp.embedding_norm, 1e-6F, true, emb_norm, stream);
    emit_state("mtp_embedding_norm", emb_norm);
    ops::linear(emb_norm, mtp.embedding_projection, emb_proj, ops::LinearPolicy::A16Only, workspace,
                stream);
    emit_state("mtp_embedding_proj", emb_proj);

    // 2. Stem: Backbone hidden mixing & projection
    flash_next_hyper_mix(backbone_hyper_hidden, mtp.mixer, hyper_scratch, hid_mix, stream);
    emit_state("mtp_hidden_mix", hid_mix);
    ops::linear(hid_mix, mtp.hidden_projection, hid_proj, ops::LinearPolicy::A16Only, workspace,
                stream);
    emit_state("mtp_hidden_proj", hid_proj);

    // 3. Stem: Combine & broadcast into 4 MTP hyper streams
    flash_next_mtp_stem_combine_and_repeat_launch(emb_proj, hid_proj, &trunk_sum, mtp_hyper_hidden,
                                                  stream);
    emit_state("mtp_trunk_input", trunk_sum);
    emit_state("mtp_hyper_init", mtp_hyper_hidden);

    // 4. Attention hyper prepare -> attn_in
    flash_next_hyper_prepare(mtp_hyper_hidden, mtp.attention_hyper, hyper_scratch, attn_in, stream);
    emit_state("mtp_attn_block_input", attn_in);

    // 5. QSA Attention decode
    flash_next_qsa_attention_decode(attn_in, mtp.attention, token_indices, mrope_positions,
                                    table_rows, selected_blocks, selected_counts, mtp_cache,
                                    workspace, attn_out, stream);
    emit_state("mtp_attn_block_output", attn_out);

    // 6. Attention hyper inject
    flash_next_hyper_inject(attn_out, hyper_scratch.injection, mtp_hyper_hidden, stream);
    emit_state("mtp_hyper_after_attn", mtp_hyper_hidden);

    // 7. MLP hyper prepare -> mlp_in
    flash_next_hyper_prepare(mtp_hyper_hidden, mtp.mlp_hyper, hyper_scratch, mlp_in, stream);
    emit_state("mtp_mlp_block_input", mlp_in);

    // 8. NVFP4 MoE evaluation
    flash_next_moe(mlp_in, mtp.moe, mlp_out, workspace, stream);
    emit_state("mtp_mlp_block_output", mlp_out);

    // 9. MLP hyper inject
    flash_next_hyper_inject(mlp_out, hyper_scratch.injection, mtp_hyper_hidden, stream);
    emit_state("mtp_hyper_after_mlp", mtp_hyper_hidden);

    if (out_hyper_hidden != nullptr && out_hyper_hidden->data != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(out_hyper_hidden->data, mtp_hyper_hidden.data,
                                   10'240ULL * batch * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToDevice, stream));
    }

    // 10. Final Hyper Mixer -> mtp_final_hidden
    flash_next_hyper_mix(mtp_hyper_hidden, mtp.mixer, hyper_scratch, mtp_final_hidden, stream);
    emit_state("mtp_final_hidden", mtp_final_hidden);

    // 11. Draft Head Linear Projection -> draft_logits
    if (model.proposal.has_value()) {
        const auto& proposal = *model.proposal;
        ops::linear(mtp_final_hidden, proposal.head, draft_logits, ops::LinearPolicy::A16Only,
                    workspace, stream);
        emit_state("mtp_draft_logits", draft_logits);

        // 12. Greedy Argmax -> draft_tokens (within subset)
        ops::argmax(draft_logits, draft_tokens, draft_logits.ne[0], stream);

        // 13. Remap from subset index to true vocabulary ID
        ops::proposal_remap_token_ids(
            draft_tokens, static_cast<const std::int32_t*>(proposal.token_ids.data),
            draft_logits.ne[0], stream);
        emit_state("mtp_draft_tokens", draft_tokens);
    } else {
        ops::linear(mtp_final_hidden, model.output_head, draft_logits, ops::LinearPolicy::A16Only,
                    workspace, stream);
        emit_state("mtp_draft_logits", draft_logits);

        // 12. Greedy Argmax -> draft_tokens
        ops::argmax(draft_logits, draft_tokens, draft_logits.ne[0], stream);
        emit_state("mtp_draft_tokens", draft_tokens);
    }
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
