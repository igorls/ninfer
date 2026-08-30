#include "targets/qwen3_8_flash_next/impl/qsa_attention.h"

#include "ninfer/ops/linear.h"

#include "core/layout.h"
#include "targets/qwen3_8_flash_next/impl/qsa_attention_kernels.h"
#include "targets/qwen3_8_flash_next/impl/qsa_attention_workspace.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

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

bool exact_fp8_f32_weight(const Weight& weight, std::int32_t rows, std::int32_t columns) {
    const std::uint64_t codes        = static_cast<std::uint64_t>(rows) * columns;
    const std::uint64_t scale_offset = (codes + 255U) & ~std::uint64_t{255U};
    const auto* payload              = static_cast<const std::byte*>(weight.payload);
    const std::int64_t scale_stride  = static_cast<std::int64_t>(rows) * 4;
    return weight.qtype == QType::FP8_E4M3FN_ROW_F32S && weight.layout == QuantLayout::RowScale &&
           weight.scale_dtype == DType::FP32 && weight.group_size == columns &&
           weight.group == columns && weight.n == rows && weight.k == columns && weight.ndim == 2 &&
           weight.shape[0] == rows && weight.shape[1] == columns && weight.shape[2] == 1 &&
           weight.shape[3] == 1 && weight.padded_shape[0] == rows &&
           weight.padded_shape[1] == columns && weight.padded_shape[2] == 1 &&
           weight.padded_shape[3] == 1 && weight.scale_ne[0] == rows && weight.scale_ne[1] == 1 &&
           weight.scale_ne[2] == 1 && weight.scale_ne[3] == 1 && weight.scale_nb[0] == 4 &&
           weight.scale_nb[1] == scale_stride && weight.scale_nb[2] == scale_stride &&
           weight.scale_nb[3] == scale_stride && payload != nullptr && weight.qdata == payload &&
           weight.scales == payload + scale_offset && weight.qhigh == nullptr &&
           weight.high_plane_bytes == 0 &&
           weight.payload_bytes >= scale_offset + static_cast<std::uint64_t>(rows) * 4 &&
           aligned_to(weight.qdata, 16) && aligned_to(weight.scales, 16);
}

} // namespace

std::size_t flash_next_qsa_attention_workspace_capacity_bytes(std::int32_t batch) {
    if (batch <= 0 || batch > 262'144) {
        throw std::invalid_argument("Flash-Next QSA attention received an invalid batch/tokens size");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_flash_next_qsa_attention_workspace(layout, batch);
    return layout.peak_bytes(256);
}

void flash_next_qsa_attention_decode(const Tensor& input, const AttentionWeights& weights,
                                     const Tensor& token_indices, const Tensor& mrope_positions,
                                     const Tensor& table_rows, const Tensor& selected_blocks,
                                     const Tensor& selected_counts, QsaAttentionCacheView cache,
                                     WorkspaceArena& workspace, Tensor& output,
                                     cudaStream_t stream) {
    const std::int32_t batch = input.ne[1];
    if (!exact_tensor(input, DType::BF16, 2'560, batch) || batch < 1 || batch > 8 ||
        !exact_tensor(output, DType::BF16, 2'560, batch) ||
        !exact_tensor(weights.query_norm, DType::BF16, 256) ||
        !exact_tensor(weights.key_norm, DType::BF16, 256) ||
        !exact_fp8_f32_weight(weights.query_gate_key_value, 13'312, 2'560) ||
        !exact_fp8_f32_weight(weights.output, 2'560, 6'144) ||
        !exact_tensor(token_indices, DType::I32, batch) ||
        !exact_tensor(mrope_positions, DType::I32, batch, 3) ||
        !exact_tensor(table_rows, DType::I32, batch) ||
        !exact_tensor(selected_blocks, DType::I32, 512, batch) ||
        !exact_tensor(selected_counts, DType::I32, batch) || cache.key_pages.dtype != DType::BF16 ||
        cache.key_pages.ne[0] != 256 || cache.key_pages.ne[1] != 64 || cache.key_pages.ne[2] != 2 ||
        cache.key_pages.ne[3] <= 0 || !cache.key_pages.is_contiguous() ||
        !aligned_to(cache.key_pages.data, 16) || cache.value_pages.dtype != DType::BF16 ||
        cache.value_pages.ne[0] != 256 || cache.value_pages.ne[1] != 64 ||
        cache.value_pages.ne[2] != 2 || cache.value_pages.ne[3] != cache.key_pages.ne[3] ||
        !cache.value_pages.is_contiguous() || !aligned_to(cache.value_pages.data, 16) ||
        cache.block_tables.dtype != DType::I32 || cache.block_tables.ne[0] <= 0 ||
        cache.block_tables.ne[1] <= 0 || cache.block_tables.ne[2] != 1 ||
        cache.block_tables.ne[3] != 1 || !cache.block_tables.is_contiguous() ||
        !aligned_to(cache.block_tables.data, 16) || stream == nullptr) {
        throw std::invalid_argument(
            "Flash-Next QSA attention received an invalid exact target view");
    }

    const auto scope = workspace.scope();
    FlashNextQsaAttentionWorkspace scratch =
        allocate_flash_next_qsa_attention_workspace(workspace, batch);
    ops::linear(input, weights.query_gate_key_value, scratch.projected, ops::LinearPolicy::A16Only,
                workspace, stream);
    flash_next_qsa_attention_launch(token_indices, mrope_positions, table_rows, selected_blocks,
                                    selected_counts, weights.query_norm, weights.key_norm, cache,
                                    scratch, stream);
    ops::linear(scratch.gated, weights.output, output, ops::LinearPolicy::A16Only, workspace,
                stream);
}

void flash_next_qsa_attention_prefill_chunk(
    const Tensor& input, const AttentionWeights& weights, const Tensor& token_indices,
    const Tensor& mrope_positions, std::int32_t table_row, const Tensor& selected_blocks,
    const Tensor& selected_counts, QsaAttentionCacheView cache, WorkspaceArena& workspace,
    Tensor& output, cudaStream_t stream, const QsaStageEmitter& emit) {
    const std::int32_t tokens = input.ne[1];
    if (tokens <= 0 || !exact_tensor(input, DType::BF16, 2'560, tokens) ||
        !exact_tensor(output, DType::BF16, 2'560, tokens) ||
        !exact_tensor(weights.query_norm, DType::BF16, 256) ||
        !exact_tensor(weights.key_norm, DType::BF16, 256) ||
        !exact_fp8_f32_weight(weights.query_gate_key_value, 13'312, 2'560) ||
        !exact_fp8_f32_weight(weights.output, 2'560, 6'144) ||
        !exact_tensor(token_indices, DType::I32, tokens) ||
        table_row < 0 || table_row >= cache.block_tables.ne[1] ||
        !exact_tensor(selected_blocks, DType::I32, 512, tokens) ||
        !exact_tensor(selected_counts, DType::I32, tokens) || cache.key_pages.dtype != DType::BF16 ||
        cache.key_pages.ne[0] != 256 || cache.key_pages.ne[1] != 64 || cache.key_pages.ne[2] != 2 ||
        cache.key_pages.ne[3] <= 0 || !cache.key_pages.is_contiguous() ||
        !aligned_to(cache.key_pages.data, 16) || cache.value_pages.dtype != DType::BF16 ||
        cache.value_pages.ne[0] != 256 || cache.value_pages.ne[1] != 64 ||
        cache.value_pages.ne[2] != 2 || cache.value_pages.ne[3] != cache.key_pages.ne[3] ||
        !cache.value_pages.is_contiguous() || !aligned_to(cache.value_pages.data, 16) ||
        cache.block_tables.dtype != DType::I32 || cache.block_tables.ne[0] <= 0 ||
        cache.block_tables.ne[1] <= 0 || cache.block_tables.ne[2] != 1 ||
        cache.block_tables.ne[3] != 1 || !cache.block_tables.is_contiguous() ||
        !aligned_to(cache.block_tables.data, 16) || stream == nullptr) {
        throw std::invalid_argument(
            "Flash-Next QSA attention prefill chunk received an invalid exact target view");
    }

    const auto scope = workspace.scope();
    FlashNextQsaAttentionWorkspace scratch =
        allocate_flash_next_qsa_attention_workspace(workspace, tokens);
    ops::linear(input, weights.query_gate_key_value, scratch.projected, ops::LinearPolicy::A16Only,
                workspace, stream);
    if (emit) { emit("qsa_projected", scratch.projected); }
    flash_next_qsa_attention_prefill_launch(token_indices, mrope_positions, table_row, selected_blocks,
                                            selected_counts, weights.query_norm, weights.key_norm,
                                            cache, scratch, stream);
    if (emit) {
        emit("qsa_query", scratch.query);
        emit("qsa_gate", scratch.gate);
        emit("qsa_key", scratch.key);
        emit("qsa_value", scratch.value);
        emit("qsa_attended", scratch.attended);
        emit("qsa_gated", scratch.gated);
    }
    ops::linear(scratch.gated, weights.output, output, ops::LinearPolicy::A16Only, workspace,
                stream);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
