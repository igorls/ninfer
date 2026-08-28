#include "targets/qwen3_8_flash_next/impl/runtime_state.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_8_flash_next::detail {

namespace {

inline std::size_t align_up_256(std::size_t bytes) noexcept { return (bytes + 255ULL) & ~255ULL; }

void validate_plan_match(const FlashNextRuntimePlan& actual) {
    const auto expected = finalize_flash_next_runtime_plan(actual.config, actual.main_page_groups);

    if (actual.resolved_tokens != expected.resolved_tokens ||
        actual.attention_physical_pages != expected.attention_physical_pages ||
        actual.indexer_physical_pages != expected.indexer_physical_pages ||
        actual.attention_logical_pages != expected.attention_logical_pages ||
        actual.indexer_logical_pages != expected.indexer_logical_pages ||
        actual.state_slots != expected.state_slots ||
        actual.maximum_blocks != expected.maximum_blocks ||
        actual.attention_kv_bytes != expected.attention_kv_bytes ||
        actual.indexer_block_keys_bytes != expected.indexer_block_keys_bytes ||
        actual.block_tables_bytes != expected.block_tables_bytes ||
        actual.recurrent_state_bytes != expected.recurrent_state_bytes ||
        actual.round_tensors_bytes != expected.round_tensors_bytes ||
        actual.workspace_bytes != expected.workspace_bytes ||
        actual.total_device_bytes != expected.total_device_bytes ||
        actual.capacity_curve.main_page_tokens != expected.capacity_curve.main_page_tokens ||
        actual.capacity_curve.minimum_main_page_groups !=
            expected.capacity_curve.minimum_main_page_groups ||
        actual.capacity_curve.maximum_main_page_groups !=
            expected.capacity_curve.maximum_main_page_groups ||
        actual.capacity_curve.bytes_per_additional_main_page_group !=
            expected.capacity_curve.bytes_per_additional_main_page_group ||
        actual.capacity_curve.minimum_device_reservation_bytes !=
            expected.capacity_curve.minimum_device_reservation_bytes) {
        throw std::invalid_argument(
            "FlashNextRuntimeAllocation: supplied plan does not match canonical layout");
    }
}

} // namespace

FlashNextRuntimeAllocation::FlashNextRuntimeAllocation(FlashNextRuntimePlan plan)
    : plan_(std::move(plan)) {
    validate_plan_match(plan_);

    const std::uint32_t concurrency = plan_.config.max_concurrency;
    host_active_slots_.resize(concurrency);
    host_standby_slots_.resize(concurrency);
    for (std::uint32_t b = 0; b < concurrency; ++b) {
        host_active_slots_[b]  = static_cast<std::int32_t>(2U * b);
        host_standby_slots_[b] = static_cast<std::int32_t>(2U * b + 1U);
    }

    const std::size_t persistent_bytes = plan_.total_device_bytes - plan_.workspace_bytes;
    storage_                           = std::make_unique<DeviceBuffer>(persistent_bytes);
    workspace_                         = std::make_unique<WorkspaceArena>(plan_.workspace_bytes);

    materialize_views();
    validate_flash_next_decode_state(state_view_, plan_.state_slots);
}

void FlashNextRuntimeAllocation::materialize_views() {
    auto* cur                       = static_cast<std::byte*>(storage_->p);
    const std::uint32_t concurrency = plan_.config.max_concurrency;

    // 1. Attention KV pages (12 layers, Key & Value)
    // Key: [256, 64, 2, attention_physical_pages] BF16
    // Value: [256, 64, 2, attention_physical_pages] BF16
    const std::size_t att_kv_bytes = align_up_256(
        256ULL * 64ULL * 2ULL * plan_.attention_physical_pages * sizeof(std::uint16_t));
    for (std::size_t i = 0; i < kFullAttentionLayers; ++i) {
        state_view_.qsa_attention_caches[i].key_pages =
            Tensor(cur, DType::BF16,
                   {256, 64, 2, static_cast<std::int32_t>(plan_.attention_physical_pages)});
        cur += att_kv_bytes;
        state_view_.qsa_attention_caches[i].value_pages =
            Tensor(cur, DType::BF16,
                   {256, 64, 2, static_cast<std::int32_t>(plan_.attention_physical_pages)});
        cur += att_kv_bytes;
    }

    // 2. Indexer compressed block keys (12 layers)
    // [128, 64, indexer_physical_pages] BF16
    const std::size_t idx_keys_bytes =
        align_up_256(128ULL * 64ULL * plan_.indexer_physical_pages * sizeof(std::uint16_t));
    for (std::size_t i = 0; i < kFullAttentionLayers; ++i) {
        state_view_.qsa_indexer_caches[i].block_keys = Tensor(
            cur, DType::BF16, {128, 64, static_cast<std::int32_t>(plan_.indexer_physical_pages)});
        cur += idx_keys_bytes;
    }

    // 3. Shared Single Block tables (1 attention block table + 1 indexer block table)
    const std::size_t att_table_bytes =
        align_up_256(static_cast<std::size_t>(plan_.attention_logical_pages) * concurrency *
                     sizeof(std::int32_t));
    Tensor shared_att_table(cur, DType::I32,
                            {static_cast<std::int32_t>(plan_.attention_logical_pages),
                             static_cast<std::int32_t>(concurrency)});
    cur += att_table_bytes;

    const std::size_t idx_table_bytes = align_up_256(
        static_cast<std::size_t>(plan_.indexer_logical_pages) * concurrency * sizeof(std::int32_t));
    Tensor shared_idx_table(cur, DType::I32,
                            {static_cast<std::int32_t>(plan_.indexer_logical_pages),
                             static_cast<std::int32_t>(concurrency)});
    cur += idx_table_bytes;

    for (std::size_t i = 0; i < kFullAttentionLayers; ++i) {
        state_view_.qsa_attention_caches[i].block_tables = shared_att_table;
        state_view_.qsa_indexer_caches[i].block_tables   = shared_idx_table;
    }

    // 4. Recurrent states
    // 36 GDN conv: [10240, 3, state_slots] BF16
    const std::size_t gdn_conv_bytes =
        align_up_256(10'240ULL * 3ULL * plan_.state_slots * sizeof(std::uint16_t));
    for (std::size_t i = 0; i < kGdnLayers; ++i) {
        state_view_.gdn_convolution_states[i] =
            Tensor(cur, DType::BF16, {10'240, 3, static_cast<std::int32_t>(plan_.state_slots)});
        cur += gdn_conv_bytes;
    }

    // 36 GDN SSM: [128, 128, 48, state_slots] FP32
    const std::size_t gdn_ssm_bytes =
        align_up_256(128ULL * 128ULL * 48ULL * plan_.state_slots * sizeof(float));
    for (std::size_t i = 0; i < kGdnLayers; ++i) {
        state_view_.gdn_ssm_states[i] =
            Tensor(cur, DType::FP32, {128, 128, 48, static_cast<std::int32_t>(plan_.state_slots)});
        cur += gdn_ssm_bytes;
    }

    // 1 PLE conv: [10240, 9, state_slots] BF16
    const std::size_t ple_conv_bytes =
        align_up_256(10'240ULL * 9ULL * plan_.state_slots * sizeof(std::uint16_t));
    state_view_.ple_convolution_states =
        Tensor(cur, DType::BF16, {10'240, 9, static_cast<std::int32_t>(plan_.state_slots)});
    cur += ple_conv_bytes;

    // 12 QSA raw keys: [128, 4, state_slots] BF16
    const std::size_t raw_keys_bytes =
        align_up_256(128ULL * 4ULL * plan_.state_slots * sizeof(std::uint16_t));
    for (std::size_t i = 0; i < kFullAttentionLayers; ++i) {
        state_view_.qsa_indexer_caches[i].raw_keys =
            Tensor(cur, DType::BF16, {128, 4, static_cast<std::int32_t>(plan_.state_slots)});
        cur += raw_keys_bytes;
    }

    // 12 QSA raw positions: [3, 4, state_slots] I32
    const std::size_t raw_pos_bytes =
        align_up_256(3ULL * 4ULL * plan_.state_slots * sizeof(std::int32_t));
    for (std::size_t i = 0; i < kFullAttentionLayers; ++i) {
        state_view_.qsa_indexer_caches[i].raw_positions =
            Tensor(cur, DType::I32, {3, 4, static_cast<std::int32_t>(plan_.state_slots)});
        cur += raw_pos_bytes;
    }

    // 5. Round buffers
    round_tensors_.token_ids = Tensor(cur, DType::I32, {static_cast<std::int32_t>(concurrency)});
    cur += align_up_256(concurrency * sizeof(std::int32_t));

    round_tensors_.token_indices =
        Tensor(cur, DType::I32, {static_cast<std::int32_t>(concurrency)});
    cur += align_up_256(concurrency * sizeof(std::int32_t));

    round_tensors_.mrope_positions =
        Tensor(cur, DType::I32, {static_cast<std::int32_t>(concurrency), 3});
    cur += align_up_256(concurrency * 3 * sizeof(std::int32_t));

    round_tensors_.table_rows = Tensor(cur, DType::I32, {static_cast<std::int32_t>(concurrency)});
    cur += align_up_256(concurrency * sizeof(std::int32_t));

    round_tensors_.source_slots = Tensor(cur, DType::I32, {static_cast<std::int32_t>(concurrency)});
    cur += align_up_256(concurrency * sizeof(std::int32_t));

    round_tensors_.destination_slots =
        Tensor(cur, DType::I32, {static_cast<std::int32_t>(concurrency)});
    cur += align_up_256(concurrency * sizeof(std::int32_t));

    round_tensors_.gathered_ple_embedding =
        Tensor(cur, DType::BF16, {2'560, static_cast<std::int32_t>(concurrency)});
    cur += align_up_256(2'560ULL * concurrency * sizeof(std::uint16_t));

    round_tensors_.final_hidden =
        Tensor(cur, DType::BF16, {2'560, static_cast<std::int32_t>(concurrency)});
    cur += align_up_256(2'560ULL * concurrency * sizeof(std::uint16_t));

    round_tensors_.logits =
        Tensor(cur, DType::BF16, {248'320, static_cast<std::int32_t>(concurrency)});
    cur += align_up_256(248'320ULL * concurrency * sizeof(std::uint16_t));

    const auto used_bytes = static_cast<std::size_t>(cur - static_cast<std::byte*>(storage_->p));
    if (used_bytes > storage_->bytes) {
        throw std::overflow_error("Flash-Next runtime layout exceeded allocated device buffer");
    }
}

void FlashNextRuntimeAllocation::initialize(cudaStream_t stream) {
    // Deterministically zero all persistent / recurrent state device storage
    CUDA_CHECK(cudaMemsetAsync(storage_->p, 0, storage_->bytes, stream));
    sync_slots_to_device(stream);
}

void FlashNextRuntimeAllocation::commit_row_slot(std::uint32_t row_index, cudaStream_t stream) {
    if (row_index >= plan_.config.max_concurrency) {
        throw std::out_of_range("row_index exceeds max_concurrency");
    }
    std::swap(host_active_slots_[row_index], host_standby_slots_[row_index]);
    sync_slots_to_device(stream);
}

void FlashNextRuntimeAllocation::sync_slots_to_device(cudaStream_t stream) {
    const std::size_t bytes = plan_.config.max_concurrency * sizeof(std::int32_t);
    CUDA_CHECK(cudaMemcpyAsync(round_tensors_.source_slots.data, host_active_slots_.data(), bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(round_tensors_.destination_slots.data, host_standby_slots_.data(),
                               bytes, cudaMemcpyHostToDevice, stream));
}

std::int32_t FlashNextRuntimeAllocation::current_source_slot(std::uint32_t row_index) const {
    if (row_index >= plan_.config.max_concurrency) {
        throw std::out_of_range("row_index exceeds max_concurrency");
    }
    return host_active_slots_[row_index];
}

std::int32_t FlashNextRuntimeAllocation::current_destination_slot(std::uint32_t row_index) const {
    if (row_index >= plan_.config.max_concurrency) {
        throw std::out_of_range("row_index exceeds max_concurrency");
    }
    return host_standby_slots_[row_index];
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
