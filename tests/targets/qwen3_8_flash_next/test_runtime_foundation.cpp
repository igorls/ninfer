#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"
#include "targets/qwen3_8_flash_next/impl/runtime_state.h"
#include "targets/qwen3_8_flash_next/impl/text_decode_state.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

int test_constants_and_math() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    if (kAttentionKvBytesPerGroup != 6'291'456ULL) {
        std::cerr << "kAttentionKvBytesPerGroup mismatch: " << kAttentionKvBytesPerGroup << "\n";
        return 1;
    }
    if (kIndexerBlockKeysBytesPerGroup != 196'608ULL) {
        std::cerr << "kIndexerBlockKeysBytesPerGroup mismatch: " << kIndexerBlockKeysBytesPerGroup
                  << "\n";
        return 1;
    }
    if (kPhysicalStrideBytesPerGroup != 6'488'064ULL) {
        std::cerr << "kPhysicalStrideBytesPerGroup mismatch: " << kPhysicalStrideBytesPerGroup
                  << "\n";
        return 1;
    }

    if (kGdnConvBytesPerSlot != 2'211'840ULL) {
        std::cerr << "kGdnConvBytesPerSlot mismatch: " << kGdnConvBytesPerSlot << "\n";
        return 1;
    }
    if (kGdnSsmBytesPerSlot != 113'246'208ULL) {
        std::cerr << "kGdnSsmBytesPerSlot mismatch: " << kGdnSsmBytesPerSlot << "\n";
        return 1;
    }
    if (kPleConvBytesPerSlot != 184'320ULL) {
        std::cerr << "kPleConvBytesPerSlot mismatch: " << kPleConvBytesPerSlot << "\n";
        return 1;
    }
    if (kQsaRawKeysBytesPerSlot != 12'288ULL) {
        std::cerr << "kQsaRawKeysBytesPerSlot mismatch: " << kQsaRawKeysBytesPerSlot << "\n";
        return 1;
    }
    if (kQsaRawPositionsBytesPerSlot != 576ULL) {
        std::cerr << "kQsaRawPositionsBytesPerSlot mismatch: " << kQsaRawPositionsBytesPerSlot
                  << "\n";
        return 1;
    }
    if (kRecurrentStateBytesPerSlot != 115'655'232ULL) {
        std::cerr << "kRecurrentStateBytesPerSlot mismatch: " << kRecurrentStateBytesPerSlot
                  << "\n";
        return 1;
    }

    std::cout << "PASS: test_constants_and_math\n";
    return 0;
}

int test_capacity_curve_and_finalize() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    // 1. Boundary: context 128 -> 1 group per sequence
    FlashNextRuntimeConfig cfg1{
        .max_concurrency     = 1,
        .max_context         = 128,
        .state_slot_capacity = 0,
        .prefill_chunk       = 128,
    };
    auto curve1 = flash_next_capacity_curve(cfg1);
    if (curve1.minimum_main_page_groups != 1 || curve1.maximum_main_page_groups != 1 ||
        curve1.main_page_tokens != 256 ||
        curve1.bytes_per_additional_main_page_group != kPhysicalStrideBytesPerGroup) {
        std::cerr << "curve1 mismatch\n";
        return 1;
    }

    auto plan1 = finalize_flash_next_runtime_plan(cfg1, 1);
    if (plan1.main_page_groups != 1 || plan1.attention_physical_pages != 4 ||
        plan1.indexer_physical_pages != 1 || plan1.attention_logical_pages != 2 ||
        plan1.indexer_logical_pages != 1 || plan1.state_slots != 2 ||
        plan1.resolved_tokens != 256) {
        std::cerr << "plan1 (context 128) mismatch\n";
        return 1;
    }

    // 2. Boundary: context 262144 -> 1024 groups per sequence
    FlashNextRuntimeConfig cfg_max{
        .max_concurrency     = 1,
        .max_context         = 262'144,
        .state_slot_capacity = 0,
        .prefill_chunk       = 1024,
    };
    auto curve_max = flash_next_capacity_curve(cfg_max);
    if (curve_max.minimum_main_page_groups != 1024 || curve_max.maximum_main_page_groups != 1024) {
        std::cerr << "curve_max mismatch\n";
        return 1;
    }

    auto plan_max = finalize_flash_next_runtime_plan(cfg_max, 1024);
    if (plan_max.main_page_groups != 1024 || plan_max.attention_physical_pages != 4096 ||
        plan_max.indexer_physical_pages != 1024 || plan_max.attention_logical_pages != 4096 ||
        plan_max.indexer_logical_pages != 1024 || plan_max.maximum_blocks != 65'536 ||
        plan_max.resolved_tokens != 262'144) {
        std::cerr << "plan_max (context 262144) mismatch\n";
        return 1;
    }

    // 3. Concurrency 8 envelope with capacity resolution
    FlashNextRuntimeConfig cfg8{
        .max_concurrency     = 8,
        .max_context         = 2048,
        .state_slot_capacity = 0,
        .prefill_chunk       = 1024,
    };
    auto curve8 = flash_next_capacity_curve(cfg8);
    if (curve8.minimum_main_page_groups != 8 || curve8.maximum_main_page_groups != 64) {
        std::cerr << "curve8 concurrency mismatch\n";
        return 1;
    }

    auto plan8_max = finalize_flash_next_runtime_plan(cfg8, curve8.maximum_main_page_groups);
    if (plan8_max.main_page_groups != 64 || plan8_max.state_slots != 16 ||
        plan8_max.resolved_tokens != 64 * 256) {
        std::cerr << "plan8_max mismatch\n";
        return 1;
    }

    auto plan8_min = finalize_flash_next_runtime_plan(cfg8, curve8.minimum_main_page_groups);
    if (plan8_min.main_page_groups != 8 || plan8_min.resolved_tokens != 2048) {
        std::cerr << "plan8_min resolution mismatch\n";
        return 1;
    }
    if (plan8_min.total_device_bytes >= plan8_max.total_device_bytes) {
        std::cerr << "Resolution at min groups did not reduce allocation bytes\n";
        return 1;
    }

    // 4. Invalid configuration rejection
    auto reject_curve = [](FlashNextRuntimeConfig c) {
        try {
            (void)flash_next_capacity_curve(c);
            return false;
        } catch (const std::invalid_argument&) { return true; }
    };

    if (!reject_curve({.max_concurrency = 0, .max_context = 256, .state_slot_capacity = 0, .prefill_chunk = 128})) {
        std::cerr << "Failed to reject max_concurrency = 0\n";
        return 1;
    }
    if (!reject_curve({.max_concurrency = 9, .max_context = 256, .state_slot_capacity = 0, .prefill_chunk = 128})) {
        std::cerr << "Failed to reject max_concurrency = 9\n";
        return 1;
    }
    if (!reject_curve({.max_concurrency = 1, .max_context = 0, .state_slot_capacity = 0, .prefill_chunk = 128})) {
        std::cerr << "Failed to reject max_context = 0\n";
        return 1;
    }
    if (!reject_curve({.max_concurrency = 1, .max_context = 262'145, .state_slot_capacity = 0, .prefill_chunk = 128})) {
        std::cerr << "Failed to reject max_context > 262144\n";
        return 1;
    }
    if (!reject_curve({.max_concurrency = 4, .max_context = 256, .state_slot_capacity = 7, .continuation_capacity = 0, .prefill_chunk = 128})) {
        std::cerr << "Failed to reject state_slot_capacity < 2 * max_concurrency\n";
        return 1;
    }
    if (!reject_curve({.max_concurrency = 2, .max_context = 256, .state_slot_capacity = 10, .continuation_capacity = 8, .prefill_chunk = 128})) {
        std::cerr << "Failed to reject state_slot_capacity < 2 * max_concurrency + continuation_capacity\n";
        return 1;
    }
    if (!reject_curve({.max_concurrency = 4, .max_context = 256, .state_slot_capacity = 65, .prefill_chunk = 128})) {
        std::cerr << "Failed to reject state_slot_capacity > 64\n";
        return 1;
    }
    if (!reject_curve({.max_concurrency = 1, .max_context = 256, .state_slot_capacity = 0, .prefill_chunk = 0})) {
        std::cerr << "Failed to reject prefill_chunk = 0\n";
        return 1;
    }
    if (!reject_curve({.max_concurrency = 1, .max_context = 256, .state_slot_capacity = 0, .prefill_chunk = 64})) {
        std::cerr << "Failed to reject prefill_chunk not aligned to 128\n";
        return 1;
    }
    if (!reject_curve({.max_concurrency = 1, .max_context = 256, .state_slot_capacity = 0, .prefill_chunk = 512})) {
        std::cerr << "Failed to reject prefill_chunk > max_context\n";
        return 1;
    }

    // Reject selected_main_page_groups out of range
    try {
        (void)finalize_flash_next_runtime_plan(cfg8, 7); // < min_groups 8
        std::cerr << "Failed to reject selected_groups < min_groups\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    try {
        (void)finalize_flash_next_runtime_plan(cfg8, 65); // > max_groups 64
        std::cerr << "Failed to reject selected_groups > max_groups\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    std::cout << "PASS: test_capacity_curve_and_finalize\n";
    return 0;
}

int test_runtime_allocation_and_slots(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    FlashNextRuntimeConfig cfg{
        .max_concurrency     = 2,
        .max_context         = 256,
        .state_slot_capacity = 4,
        .prefill_chunk       = 128,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups);

    // 1. Test plan tamper rejection (byte mismatch and curve mismatch)
    auto forged_plan_bytes = plan;
    forged_plan_bytes.total_device_bytes -= 1024;
    try {
        FlashNextRuntimeAllocation bad_alloc(forged_plan_bytes);
        std::cerr << "Failed to reject tampered bytes in plan\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    auto forged_plan_curve = plan;
    forged_plan_curve.capacity_curve.minimum_main_page_groups += 1;
    try {
        FlashNextRuntimeAllocation bad_alloc(forged_plan_curve);
        std::cerr << "Failed to reject tampered curve in plan\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    FlashNextRuntimeAllocation alloc(plan);

    // 2. Initialize device slots and zero persistent recurrent state
    alloc.initialize(device.stream);
    device.synchronize();

    // Verify representative state values on device are zeroed
    std::vector<std::uint16_t> ple_host(10'240 * 9 * 4, 0x1234);
    CUDA_CHECK(cudaMemcpy(ple_host.data(), alloc.state_view().ple_convolution_states.data,
                          ple_host.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < ple_host.size(); ++i) {
        if (ple_host[i] != 0) {
            std::cerr << "PLE state not zeroed on initialization at index " << i << ": "
                      << ple_host[i] << "\n";
            return 1;
        }
    }

    // 3. Verify state view passes validation
    try {
        validate_flash_next_decode_state(alloc.state_view(), plan.state_slots);
    } catch (const std::exception& e) {
        std::cerr << "State view failed validation: " << e.what() << "\n";
        return 1;
    }

    // 4. Verify shared block tables across all 12 layers
    for (std::size_t i = 1; i < kFullAttentionLayers; ++i) {
        if (alloc.state_view().qsa_attention_caches[i].block_tables.data !=
            alloc.state_view().qsa_attention_caches[0].block_tables.data) {
            std::cerr << "Attention block tables not shared across layers\n";
            return 1;
        }
        if (alloc.state_view().qsa_indexer_caches[i].block_tables.data !=
            alloc.state_view().qsa_indexer_caches[0].block_tables.data) {
            std::cerr << "Indexer block tables not shared across layers\n";
            return 1;
        }
    }

    // 5. Verify initial device slots match host (source=0, destination=1 for row 0)
    std::vector<std::int32_t> dev_src(2), dev_dst(2);
    CUDA_CHECK(cudaMemcpy(dev_src.data(), alloc.round_tensors().source_slots.data,
                          2 * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dev_dst.data(), alloc.round_tensors().destination_slots.data,
                          2 * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
    if (dev_src[0] != 0 || dev_dst[0] != 1 || dev_src[1] != 2 || dev_dst[1] != 3) {
        std::cerr << "Initial device slot tensors mismatch\n";
        return 1;
    }

    // 6. Test slot transactional commit and address stability
    const void* initial_ple_data    = alloc.state_view().ple_convolution_states.data;
    const void* initial_logits_data = alloc.round_tensors().logits.data;

    alloc.commit_row_slot(0, device.stream);
    device.synchronize();

    const auto new_src = alloc.current_source_slot(0);
    const auto new_dst = alloc.current_destination_slot(0);
    if (new_src != 1 || new_dst != 0) {
        std::cerr << "Commit row slot failed host swap: expected src=1 dst=0, got src=" << new_src
                  << " dst=" << new_dst << "\n";
        return 1;
    }

    CUDA_CHECK(cudaMemcpy(dev_src.data(), alloc.round_tensors().source_slots.data,
                          2 * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dev_dst.data(), alloc.round_tensors().destination_slots.data,
                          2 * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
    if (dev_src[0] != 1 || dev_dst[0] != 0) {
        std::cerr << "Device slot tensors not updated after commit\n";
        return 1;
    }

    if (alloc.state_view().ple_convolution_states.data != initial_ple_data ||
        alloc.round_tensors().logits.data != initial_logits_data) {
        std::cerr << "Device tensor pointers changed after slot commit\n";
        return 1;
    }

    std::cout << "PASS: test_runtime_allocation_and_slots\n";
    return 0;
}

} // namespace

int main() {
    // 1. CPU tests run unconditionally without requiring a CUDA device
    if (test_constants_and_math() != 0) return 1;
    if (test_capacity_curve_and_finalize() != 0) return 1;

    // 2. CUDA device tests run only when a CUDA device is available
    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || device_count == 0) {
        std::cout << "SKIP: CUDA device tests (no usable device)\n";
        return 0;
    }
    CUDA_CHECK(count_error);

    ninfer::DeviceContext device(0);
    if (test_runtime_allocation_and_slots(device) != 0) return 1;

    std::cout << "OK Flash-Next Runtime Foundation\n";
    return 0;
}
