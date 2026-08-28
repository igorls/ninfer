#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/lane_ledger.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"
#include "targets/qwen3_8_flash_next/impl/ple_index.h"
#include "targets/qwen3_8_flash_next/impl/ple_table.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"
#include "targets/qwen3_8_flash_next/impl/runtime_state.h"
#include "targets/qwen3_8_flash_next/impl/text_executor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

int test_ledger_cpu() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    FlashNextRuntimeConfig cfg{
        .max_concurrency     = 2,
        .max_context         = 512,
        .state_slot_capacity = 4,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    FlashNextLaneLedger ledger(plan);

    if (ledger.available_physical_groups() != plan.main_page_groups) {
        std::cerr << "Initial available physical groups mismatch\n";
        return 1;
    }

    // 1. Allocate two lanes
    auto lane0 = ledger.allocate_lane();
    auto lane1 = ledger.allocate_lane();

    if (lane0.lane_index() != 0 || lane1.lane_index() != 1) {
        std::cerr << "Lane indices mismatch\n";
        return 1;
    }
    if (ledger.active_lanes_count() != 2) {
        std::cerr << "Active lanes count mismatch\n";
        return 1;
    }

    // 2. Reject 3rd lane allocation
    try {
        (void)ledger.allocate_lane();
        std::cerr << "Failed to reject 3rd lane allocation\n";
        return 1;
    } catch (const std::runtime_error&) {}

    // 3. Stale handle rejection
    auto stale_handle = lane0;
    ledger.release_lane(lane0);
    try {
        (void)ledger.committed_frontier(stale_handle);
        std::cerr << "Failed to reject released/stale lane handle\n";
        return 1;
    } catch (const std::invalid_argument&) {}
    lane0 = ledger.allocate_lane();

    // 4. CPU owner-isolation: two fresh ledgers with the same lane index and epoch
    FlashNextLaneLedger owner_ledger_a(plan);
    FlashNextLaneLedger owner_ledger_b(plan);
    auto owner_lane_a = owner_ledger_a.allocate_lane();
    auto owner_lane_b = owner_ledger_b.allocate_lane();
    if (owner_lane_a.lane_index() != owner_lane_b.lane_index() ||
        owner_lane_a.epoch() != owner_lane_b.epoch()) {
        std::cerr << "Owner-isolation setup did not produce matching lane/epoch\n";
        return 1;
    }
    try {
        (void)owner_ledger_a.committed_frontier(owner_lane_b);
        std::cerr << "Failed to reject cross-ledger handle with same lane/epoch\n";
        return 1;
    } catch (const std::invalid_argument&) {}
    owner_ledger_a.release_lane(owner_lane_a);
    owner_ledger_b.release_lane(owner_lane_b);

    // 5. Batch validation: empty, duplicate, non-monotonic
    PleIndexMetadata ple_meta{};
    ple_meta.multipliers = {1, 2, 3};
    ple_meta.head_offsets.fill(0);
    ple_meta.head_vocab_sizes.fill(100);

    std::vector<LaneStepRequest> empty_reqs;
    try {
        (void)ledger.begin_round(empty_reqs, ple_meta);
        std::cerr << "Failed to reject empty batch\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    std::vector<LaneStepRequest> dup_reqs = {
        {.handle = lane0, .token_id = 42, .token_index = 0, .mrope_positions = {0, 0, 0}},
        {.handle = lane0, .token_id = 43, .token_index = 0, .mrope_positions = {0, 0, 0}},
    };
    try {
        (void)ledger.begin_round(dup_reqs, ple_meta);
        std::cerr << "Failed to reject duplicate lane\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    std::vector<LaneStepRequest> non_mono_reqs = {
        {.handle = lane0, .token_id = 42, .token_index = 5, .mrope_positions = {5, 5, 5}},
    };
    try {
        (void)ledger.begin_round(non_mono_reqs, ple_meta);
        std::cerr << "Failed to reject non-monotonic token index\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    // 6. Active blocks exact calculation: token_index=0 -> blocks=0
    std::vector<LaneStepRequest> req_token0 = {
        {.handle = lane0, .token_id = 1, .token_index = 0, .mrope_positions = {0, 0, 0}},
    };
    auto prep0 = ledger.begin_round(req_token0, ple_meta);
    if (prep0.max_active_blocks != 0) {
        std::cerr << "Active blocks for token 0 expected 0 got " << prep0.max_active_blocks << "\n";
        return 1;
    }
    ledger.abort_round(prep0.transaction_id);

    // 7. Valid round, exact table indexing, and full rollback verification
    const auto groups_before = ledger.available_physical_groups();
    std::vector<LaneStepRequest> valid_reqs = {
        {.handle = lane0, .token_id = 10, .token_index = 0, .mrope_positions = {0, 0, 0}},
        {.handle = lane1, .token_id = 20, .token_index = 0, .mrope_positions = {0, 0, 0}},
    };
    auto prep = ledger.begin_round(valid_reqs, ple_meta);

    // Check lane 1 exact indexer and attention table entries
    const auto att_tab = ledger.host_attention_table();
    const auto idx_tab = ledger.host_indexer_table();

    const auto l1_idx_group = idx_tab[1ULL * plan.indexer_logical_pages + 0];
    if (l1_idx_group < 0) {
        std::cerr << "Host indexer table for lane 1 not mapped\n";
        return 1;
    }
    for (std::uint32_t s = 0; s < 4; ++s) {
        const auto att_val = att_tab[1ULL * plan.attention_logical_pages + s];
        if (att_val != static_cast<std::int32_t>(l1_idx_group * 4 + s)) {
            std::cerr << "Host attention table for lane 1 mismatch at subpage " << s << "\n";
            return 1;
        }
    }

    // Rollback: verify ALL attention+indexer entries cleared AND groups restored
    ledger.rollback_prepared_round(prep.transaction_id);
    if (ledger.has_pending_transaction()) {
        std::cerr << "Pending transaction still active after rollback\n";
        return 1;
    }
    if (ledger.available_physical_groups() != groups_before) {
        std::cerr << "Physical groups not restored after rollback: "
                  << ledger.available_physical_groups() << " expected " << groups_before << "\n";
        return 1;
    }
    // All attention entries for lane 1 should be -1
    for (std::uint32_t s = 0; s < 4; ++s) {
        if (att_tab[1ULL * plan.attention_logical_pages + s] != -1) {
            std::cerr << "Attention table entry not cleared after rollback at subpage " << s
                      << "\n";
            return 1;
        }
    }
    if (idx_tab[1ULL * plan.indexer_logical_pages + 0] != -1) {
        std::cerr << "Indexer table entry not cleared after rollback\n";
        return 1;
    }

    // 8. Re-run round and test commit accept/reject
    prep = ledger.begin_round(valid_reqs, ple_meta);
    FlashNextRuntimeAllocation dummy_alloc(plan);
    std::vector<LaneCommitDecision> decisions = {
        {.accept = true},
        {.accept = false},
    };
    ledger.commit_round(prep.transaction_id, decisions, dummy_alloc, nullptr);

    if (ledger.committed_frontier(lane0) != 1) {
        std::cerr << "Lane 0 frontier mismatch on accept\n";
        return 1;
    }
    if (ledger.committed_frontier(lane1) != 0) {
        std::cerr << "Lane 1 frontier mismatch on reject\n";
        return 1;
    }

    ledger.release_lane(lane0);
    ledger.release_lane(lane1);

    if (ledger.available_physical_groups() != plan.main_page_groups) {
        std::cerr << "Physical groups not fully reclaimed after release\n";
        return 1;
    }

    std::cout << "PASS: test_ledger_cpu\n";
    return 0;
}

int test_cuda_ledger_and_executor(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    PleIndexMetadata ple_meta{};
    ple_meta.multipliers = {1, 2, 3};
    ple_meta.head_offsets.fill(0);
    ple_meta.head_vocab_sizes.fill(1);

    constexpr std::uint64_t rows         = 1;
    constexpr std::uint64_t width        = 160;
    constexpr std::uint64_t scale_offset = 256;
    std::vector<std::byte> encoded(scale_offset + (width / 16) * 2, std::byte{0});
    std::fill_n(encoded.begin(), width / 2, std::byte{0x88});
    for (std::uint8_t index = 0; index < 8; ++index) {
        encoded[index] = static_cast<std::byte>(index * 2 | ((index * 2 + 1) << 4));
    }
    constexpr std::uint16_t half_point_five = 0x3800;
    for (std::size_t offset = scale_offset; offset < encoded.size(); offset += 2) {
        std::memcpy(encoded.data() + offset, &half_point_five, sizeof(half_point_five));
    }
    PleTableView ple_table;
    for (PleShardView& shard : ple_table.shards) {
        shard = make_ple_shard_view(encoded, rows, width);
    }

    TextModelView mock_model{};
    mock_model.ple.table = ple_table;

    FlashNextRuntimeConfig cfg{
        .max_concurrency     = 2,
        .max_context         = 512,
        .state_slot_capacity = 4,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    FlashNextRuntimeAllocation alloc(plan);
    alloc.initialize(device.stream);
    device.synchronize();

    // 1. Bounded CUDA ledger: table upload and slot commit
    FlashNextLaneLedger ledger(plan);
    auto lane0 = ledger.allocate_lane();
    auto lane1 = ledger.allocate_lane();

    std::vector<LaneStepRequest> reqs = {
        {.handle = lane0, .token_id = 0, .token_index = 0, .mrope_positions = {0, 0, 0}},
        {.handle = lane1, .token_id = 0, .token_index = 0, .mrope_positions = {0, 0, 0}},
    };
    auto prep = ledger.begin_round(reqs, ple_meta);
    ledger.sync_tables_if_dirty(alloc, device.stream);
    device.synchronize();

    // Verify transposed device table for lane 1
    std::vector<std::int32_t> dev_att(plan.attention_logical_pages * cfg.max_concurrency);
    CUDA_CHECK(cudaMemcpy(dev_att.data(),
                          alloc.state_view().qsa_attention_caches[0].block_tables.data,
                          dev_att.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));

    std::vector<std::int32_t> dev_idx(plan.indexer_logical_pages * cfg.max_concurrency);
    CUDA_CHECK(cudaMemcpy(dev_idx.data(),
                          alloc.state_view().qsa_indexer_caches[0].block_tables.data,
                          dev_idx.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));

    const auto l1_phys = dev_idx[1ULL * plan.indexer_logical_pages + 0];
    if (l1_phys < 0) {
        std::cerr << "Device indexer table for lane 1 not mapped\n";
        return 1;
    }
    for (std::uint32_t s = 0; s < 4; ++s) {
        const auto val = dev_att[1ULL * plan.attention_logical_pages + s];
        if (val != static_cast<std::int32_t>(l1_phys * 4 + s)) {
            std::cerr << "Device attention table for lane 1 mismatch\n";
            return 1;
        }
    }

    std::vector<LaneCommitDecision> decisions = {{.accept = true}, {.accept = false}};
    ledger.commit_round(prep.transaction_id, decisions, alloc, device.stream);
    device.synchronize();

    if (alloc.current_source_slot(0) != 1 || alloc.current_source_slot(1) != 2) {
        std::cerr << "Slots after commit mismatch\n";
        return 1;
    }

    ledger.release_lane(lane0);
    ledger.release_lane(lane1);

    // 2. Executor: cross-executor rejection, invalid-model execute_round failure rollback
    FlashNextTextExecutor executor(mock_model, ple_meta, device, alloc);
    auto elane0 = executor.allocate_lane();
    auto elane1 = executor.allocate_lane();

    // Cross-executor rejection
    FlashNextRuntimeAllocation alloc2(plan);
    alloc2.initialize(device.stream);
    FlashNextTextExecutor executor2(mock_model, ple_meta, device, alloc2);
    try {
        (void)executor2.committed_frontier(elane0);
        std::cerr << "Failed to reject cross-executor handle\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    // 3. Invalid-model execute_round must throw (flash_next_text_decode with null weights),
    //    must leave no pending transaction, and must restore physical groups.
    const auto groups_before_exec = executor.available_physical_groups();
    std::vector<LaneStepRequest> exec_reqs = {
        {.handle = elane0, .token_id = 0, .token_index = 0, .mrope_positions = {0, 0, 0}},
    };
    bool decode_threw = false;
    try {
        auto round = executor.execute_round(exec_reqs);
        round.abort();
    } catch (...) { decode_threw = true; }
    if (!decode_threw) {
        std::cerr << "Invalid model unexpectedly bypassed the real decode path\n";
        return 1;
    }
    if (executor.has_pending_round()) {
        std::cerr << "Pending round leaked after execute_round failure\n";
        return 1;
    }
    if (executor.available_physical_groups() != groups_before_exec) {
        std::cerr << "Physical groups leaked after execute_round failure: "
                  << executor.available_physical_groups() << " expected " << groups_before_exec
                  << "\n";
        return 1;
    }

    // 4. Recurrent state slot zeroing on allocation/reuse
    std::vector<std::uint16_t> dirty_ple(10'240 * 9, 0xABCD);
    CUDA_CHECK(cudaMemcpy(alloc.state_view().ple_convolution_states.data, dirty_ple.data(),
                          dirty_ple.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));

    executor.release_lane(elane0);
    auto reallocated = executor.allocate_lane();
    device.synchronize();

    std::vector<std::uint16_t> clean_ple(10'240 * 9, 0x1234);
    CUDA_CHECK(cudaMemcpy(clean_ple.data(), alloc.state_view().ple_convolution_states.data,
                          clean_ple.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < clean_ple.size(); ++i) {
        if (clean_ple[i] != 0) {
            std::cerr << "Recurrent state not zeroed on lane reallocation\n";
            return 1;
        }
    }

    executor.release_lane(reallocated);
    executor.release_lane(elane1);

    std::cout << "PASS: test_cuda_ledger_and_executor\n";
    return 0;
}

} // namespace

int main() {
    if (test_ledger_cpu() != 0) return 1;

    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || device_count == 0) {
        std::cout << "SKIP: CUDA device tests (no usable device)\n";
        return 0;
    }
    CUDA_CHECK(count_error);

    ninfer::DeviceContext device(0);

    if (test_cuda_ledger_and_executor(device) != 0) return 1;

    std::cout << "OK Flash-Next Text Executor\n";
    return 0;
}
