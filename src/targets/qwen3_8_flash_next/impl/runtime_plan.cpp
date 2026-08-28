#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"

#include "targets/qwen3_8_flash_next/impl/text_decode.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next::detail {

namespace {

template <typename T>
T checked_add(T a, T b) {
    if (b > 0 && a > std::numeric_limits<T>::max() - b) {
        throw std::overflow_error("runtime plan: integer addition overflow");
    }
    return a + b;
}

template <typename T>
T checked_mul(T a, T b) {
    if (a > 0 && b > 0 && a > std::numeric_limits<T>::max() / b) {
        throw std::overflow_error("runtime plan: integer multiplication overflow");
    }
    return a * b;
}

std::size_t checked_align_up_256(std::size_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() - 255ULL) {
        throw std::overflow_error("runtime plan: alignment overflow");
    }
    return (bytes + 255ULL) & ~255ULL;
}

void validate_config_invariants(const FlashNextRuntimeConfig& config,
                                std::uint32_t& resolved_state_slots) {
    if (config.max_concurrency < 1 || config.max_concurrency > 8) {
        throw std::invalid_argument("Flash-Next max_concurrency must be in [1, 8]");
    }
    if (config.max_context < 1 || config.max_context > 262'144) {
        throw std::invalid_argument("Flash-Next max_context must be in [1, 262144]");
    }

    resolved_state_slots = config.state_slot_capacity;
    if (resolved_state_slots == 0) {
        resolved_state_slots = 2U * config.max_concurrency;
    } else if (resolved_state_slots < 2U * config.max_concurrency || resolved_state_slots > 64) {
        throw std::invalid_argument(
            "Flash-Next state_slot_capacity must be in [2 * max_concurrency, 64]");
    }
}

std::size_t
compute_fixed_base_bytes(const FlashNextRuntimeConfig& config, std::uint32_t resolved_state_slots,
                         std::uint32_t attention_logical_pages, std::uint32_t indexer_logical_pages,
                         std::uint32_t maximum_blocks, std::size_t& block_tables_bytes,
                         std::size_t& recurrent_state_bytes, std::size_t& round_tensors_bytes,
                         std::size_t& workspace_bytes) {
    // 1. Shared Single Block tables
    const std::size_t single_att_table_plane = checked_align_up_256(checked_mul<std::size_t>(
        static_cast<std::size_t>(attention_logical_pages) * config.max_concurrency,
        sizeof(std::int32_t)));
    const std::size_t single_idx_table_plane = checked_align_up_256(checked_mul<std::size_t>(
        static_cast<std::size_t>(indexer_logical_pages) * config.max_concurrency,
        sizeof(std::int32_t)));
    block_tables_bytes = checked_add(single_att_table_plane, single_idx_table_plane);

    // 2. Recurrent states
    const std::size_t single_gdn_conv = checked_align_up_256(
        checked_mul<std::size_t>(10'240ULL * 3ULL * sizeof(std::uint16_t), resolved_state_slots));
    const std::size_t single_gdn_ssm = checked_align_up_256(
        checked_mul<std::size_t>(128ULL * 128ULL * 48ULL * sizeof(float), resolved_state_slots));
    const std::size_t ple_conv = checked_align_up_256(
        checked_mul<std::size_t>(10'240ULL * 9ULL * sizeof(std::uint16_t), resolved_state_slots));
    const std::size_t single_raw_keys = checked_align_up_256(
        checked_mul<std::size_t>(128ULL * 4ULL * sizeof(std::uint16_t), resolved_state_slots));
    const std::size_t single_raw_pos = checked_align_up_256(
        checked_mul<std::size_t>(3ULL * 4ULL * sizeof(std::int32_t), resolved_state_slots));

    recurrent_state_bytes = checked_add(
        checked_add(checked_mul(36ULL, single_gdn_conv), checked_mul(36ULL, single_gdn_ssm)),
        checked_add(ple_conv, checked_add(checked_mul(12ULL, single_raw_keys),
                                          checked_mul(12ULL, single_raw_pos))));

    // 3. Round buffers
    round_tensors_bytes = checked_add(
        checked_align_up_256(config.max_concurrency * sizeof(std::int32_t)),
        checked_add(
            checked_align_up_256(config.max_concurrency * sizeof(std::int32_t)),
            checked_add(
                checked_align_up_256(config.max_concurrency * 3 * sizeof(std::int32_t)),
                checked_add(
                    checked_align_up_256(config.max_concurrency * sizeof(std::int32_t)),
                    checked_add(
                        checked_align_up_256(config.max_concurrency * sizeof(std::int32_t)),
                        checked_add(
                            checked_align_up_256(config.max_concurrency * sizeof(std::int32_t)),
                            checked_add(
                                checked_align_up_256(2'560ULL * config.max_concurrency *
                                                     sizeof(std::uint16_t)),
                                checked_add(
                                    checked_align_up_256(2'560ULL * config.max_concurrency *
                                                         sizeof(std::uint16_t)),
                                    checked_align_up_256(248'320ULL * config.max_concurrency *
                                                         sizeof(std::uint16_t))))))))));

    // 4. Text decode workspace peak
    workspace_bytes =
        flash_next_text_decode_workspace_capacity_bytes(maximum_blocks, config.max_concurrency);

    return checked_add(
        block_tables_bytes,
        checked_add(recurrent_state_bytes, checked_add(round_tensors_bytes, workspace_bytes)));
}

} // namespace

ninfer::runtime::SequenceCapacityCurve
flash_next_capacity_curve(const FlashNextRuntimeConfig& config) {
    std::uint32_t resolved_state_slots = 0;
    validate_config_invariants(config, resolved_state_slots);

    const std::uint32_t groups_per_seq =
        (config.max_context + kMainPageGroupTokens - 1U) / kMainPageGroupTokens;
    const std::uint32_t min_groups =
        std::max<std::uint32_t>(groups_per_seq, config.max_concurrency);
    const std::uint32_t max_groups =
        checked_mul<std::uint32_t>(config.max_concurrency, groups_per_seq);

    const std::uint32_t attention_logical_pages =
        (config.max_context + kPageTokens - 1U) / kPageTokens;
    const std::uint32_t indexer_logical_pages =
        (config.max_context + kIndexerPageTokens - 1U) / kIndexerPageTokens;
    const std::uint32_t maximum_blocks =
        std::min<std::uint32_t>(65'536U, (config.max_context + kBlockTokens - 1U) / kBlockTokens);

    std::size_t block_tables_bytes    = 0;
    std::size_t recurrent_state_bytes = 0;
    std::size_t round_tensors_bytes   = 0;
    std::size_t workspace_bytes       = 0;
    const std::size_t fixed_base_bytes =
        compute_fixed_base_bytes(config, resolved_state_slots, attention_logical_pages,
                                 indexer_logical_pages, maximum_blocks, block_tables_bytes,
                                 recurrent_state_bytes, round_tensors_bytes, workspace_bytes);

    ninfer::runtime::SequenceCapacityCurve curve{};
    curve.main_page_tokens                     = kMainPageGroupTokens;
    curve.minimum_main_page_groups             = min_groups;
    curve.maximum_main_page_groups             = max_groups;
    curve.bytes_per_additional_main_page_group = kPhysicalStrideBytesPerGroup;
    curve.minimum_device_reservation_bytes     = checked_add(
        fixed_base_bytes, checked_mul<std::size_t>(min_groups, kPhysicalStrideBytesPerGroup));

    return curve;
}

FlashNextRuntimePlan finalize_flash_next_runtime_plan(const FlashNextRuntimeConfig& config,
                                                      std::uint32_t selected_main_page_groups) {
    std::uint32_t resolved_state_slots = 0;
    validate_config_invariants(config, resolved_state_slots);

    const auto curve = flash_next_capacity_curve(config);
    if (selected_main_page_groups < curve.minimum_main_page_groups ||
        selected_main_page_groups > curve.maximum_main_page_groups) {
        throw std::invalid_argument(
            "Flash-Next selected_main_page_groups outside capacity curve [min_groups, max_groups]");
    }

    FlashNextRuntimePlan plan{};
    plan.config                     = config;
    plan.config.state_slot_capacity = resolved_state_slots;
    plan.main_page_groups           = selected_main_page_groups;
    plan.resolved_tokens =
        checked_mul<std::uint32_t>(selected_main_page_groups, kMainPageGroupTokens);
    plan.attention_physical_pages = checked_mul<std::uint32_t>(4U, selected_main_page_groups);
    plan.indexer_physical_pages   = selected_main_page_groups;
    plan.attention_logical_pages  = (config.max_context + kPageTokens - 1U) / kPageTokens;
    plan.indexer_logical_pages =
        (config.max_context + kIndexerPageTokens - 1U) / kIndexerPageTokens;
    plan.state_slots = resolved_state_slots;
    plan.maximum_blocks =
        std::min<std::uint32_t>(65'536U, (config.max_context + kBlockTokens - 1U) / kBlockTokens);

    const std::size_t single_att_kv_plane = checked_align_up_256(checked_mul<std::size_t>(
        256ULL * 64ULL * 2ULL * sizeof(std::uint16_t), plan.attention_physical_pages));
    plan.attention_kv_bytes               = checked_mul<std::size_t>(24ULL, single_att_kv_plane);

    const std::size_t single_indexer_keys_plane = checked_align_up_256(checked_mul<std::size_t>(
        128ULL * 64ULL * sizeof(std::uint16_t), plan.indexer_physical_pages));
    plan.indexer_block_keys_bytes = checked_mul<std::size_t>(12ULL, single_indexer_keys_plane);

    const std::size_t fixed_base_bytes = compute_fixed_base_bytes(
        config, resolved_state_slots, plan.attention_logical_pages, plan.indexer_logical_pages,
        plan.maximum_blocks, plan.block_tables_bytes, plan.recurrent_state_bytes,
        plan.round_tensors_bytes, plan.workspace_bytes);

    plan.total_device_bytes = checked_add(
        checked_add(plan.attention_kv_bytes, plan.indexer_block_keys_bytes), fixed_base_bytes);
    plan.capacity_curve = curve;

    return plan;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
