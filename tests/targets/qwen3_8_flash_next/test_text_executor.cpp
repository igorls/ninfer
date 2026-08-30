#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/lane_ledger.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"
#include "targets/qwen3_8_flash_next/impl/ple_index.h"
#include "targets/qwen3_8_flash_next/impl/ple_table.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"
#include "targets/qwen3_8_flash_next/impl/runtime_state.h"
#include "targets/qwen3_8_flash_next/impl/text_decode.h"
#include "targets/qwen3_8_flash_next/impl/text_decode_workspace.h"
#include "targets/qwen3_8_flash_next/impl/text_executor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
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
        .prefill_chunk       = 512,
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

int test_ple_boundary_lifecycle_cpu() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    FlashNextRuntimeConfig cfg{
        .max_concurrency     = 2,
        .max_context         = 512,
        .state_slot_capacity = 4,
        .prefill_chunk       = 512,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    FlashNextLaneLedger ledger(plan);
    PleIndexMetadata ple_meta{};
    ple_meta.multipliers = {1, 2, 3};
    ple_meta.head_offsets.fill(0);
    ple_meta.head_vocab_sizes.fill(100);

    // 1. Initial allocation: exact default history (248044, 248044)
    auto lane = ledger.allocate_lane();
    const auto& hist0 = ledger.lane_history(lane);
    if (hist0.previous_token() != kPleBoundaryTokenId ||
        hist0.second_previous_token() != kPleBoundaryTokenId) {
        std::cerr << "Initial lane history not seeded with kPleBoundaryTokenId (248044)\n";
        return 1;
    }

    // 2. Abort unchanged
    std::vector<LaneStepRequest> req0 = {
        {.handle = lane, .token_id = 1234, .token_index = 0, .mrope_positions = {0, 0, 0}},
    };
    auto prep0 = ledger.begin_round(req0, ple_meta);
    ledger.abort_round(prep0.transaction_id);
    const auto& hist_after_abort = ledger.lane_history(lane);
    if (hist_after_abort.previous_token() != kPleBoundaryTokenId ||
        hist_after_abort.second_previous_token() != kPleBoundaryTokenId) {
        std::cerr << "Lane history changed after abort_round\n";
        return 1;
    }

    // 3. Commit token 500 -> (500, 248044)
    FlashNextRuntimeAllocation dummy_alloc(plan);
    std::vector<LaneCommitDecision> accept = {{.accept = true}};
    prep0 = ledger.begin_round(req0, ple_meta);
    ledger.commit_round(prep0.transaction_id, accept, dummy_alloc, nullptr);
    const auto& hist1 = ledger.lane_history(lane);
    if (hist1.previous_token() != 1234 || hist1.second_previous_token() != kPleBoundaryTokenId) {
        std::cerr << "Lane history mismatch after first commit\n";
        return 1;
    }

    // 4. Token 248046: ordinary advance (not reset) -> (248046, 1234)
    std::vector<LaneStepRequest> req1 = {
        {.handle = lane, .token_id = 248'046, .token_index = 1, .mrope_positions = {1, 1, 1}},
    };
    auto prep1 = ledger.begin_round(req1, ple_meta);
    ledger.commit_round(prep1.transaction_id, accept, dummy_alloc, nullptr);
    const auto& hist2 = ledger.lane_history(lane);
    if (hist2.previous_token() != 248'046 || hist2.second_previous_token() != 1234) {
        std::cerr << "Token 248046 did not advance history normally\n";
        return 1;
    }

    // 5. Token 248044: boundary reset -> (248044, 248044)
    std::vector<LaneStepRequest> req2 = {
        {.handle = lane, .token_id = 248'044, .token_index = 2, .mrope_positions = {2, 2, 2}},
    };
    auto prep2 = ledger.begin_round(req2, ple_meta);
    ledger.commit_round(prep2.transaction_id, accept, dummy_alloc, nullptr);
    const auto& hist3 = ledger.lane_history(lane);
    if (hist3.previous_token() != kPleBoundaryTokenId ||
        hist3.second_previous_token() != kPleBoundaryTokenId) {
        std::cerr << "Token 248044 did not reset boundary history\n";
        return 1;
    }

    // 6. Next token 777 after reset -> (777, 248044)
    std::vector<LaneStepRequest> req3 = {
        {.handle = lane, .token_id = 777, .token_index = 3, .mrope_positions = {3, 3, 3}},
    };
    auto prep3 = ledger.begin_round(req3, ple_meta);
    ledger.commit_round(prep3.transaction_id, accept, dummy_alloc, nullptr);
    const auto& hist4 = ledger.lane_history(lane);
    if (hist4.previous_token() != 777 || hist4.second_previous_token() != kPleBoundaryTokenId) {
        std::cerr << "Token after boundary reset mismatch\n";
        return 1;
    }

    // 7. Release and reallocate reseed -> (248044, 248044)
    ledger.release_lane(lane);
    auto reallocated_lane = ledger.allocate_lane();
    const auto& hist_realloc = ledger.lane_history(reallocated_lane);
    if (hist_realloc.previous_token() != kPleBoundaryTokenId ||
        hist_realloc.second_previous_token() != kPleBoundaryTokenId) {
        std::cerr << "Reallocated lane history not reseeded with kPleBoundaryTokenId\n";
        return 1;
    }
    ledger.release_lane(reallocated_lane);

    std::cout << "PASS: test_ple_boundary_lifecycle_cpu\n";
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
        .prefill_chunk       = 512,
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

    // 5. B=2 planar MRoPE positions layout on device
    auto plane0 = executor.allocate_lane();
    auto plane1 = executor.allocate_lane();
    std::vector<LaneStepRequest> b2_reqs = {
        {.handle = plane0, .token_id = 10, .token_index = 0, .mrope_positions = {100, 101, 102}},
        {.handle = plane1, .token_id = 20, .token_index = 0, .mrope_positions = {200, 201, 202}},
    };
    try {
        auto round = executor.execute_round(b2_reqs);
        round.abort();
    } catch (...) {}
    device.synchronize();

    std::vector<std::int32_t> dev_mrope(6);
    CUDA_CHECK(cudaMemcpy(dev_mrope.data(), alloc.round_tensors().mrope_positions.data,
                          6 * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
    const std::vector<std::int32_t> expected_planar_mrope = {100, 200, 101, 201, 102, 202};
    if (dev_mrope != expected_planar_mrope) {
        std::cerr << "Device MRoPE positions did not match expected planar layout for B=2: got [";
        for (auto v : dev_mrope) std::cerr << v << ", ";
        std::cerr << "]\n";
        return 1;
    }
    executor.release_lane(plane0);
    executor.release_lane(plane1);

    std::cout << "PASS: test_cuda_ledger_and_executor\n";
    return 0;
}

int test_ledger_prefill_chunk_cpu() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    FlashNextRuntimeConfig cfg{
        .max_concurrency     = 2,
        .max_context         = 512,
        .state_slot_capacity = 4,
        .prefill_chunk       = 512,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    FlashNextLaneLedger ledger(plan);
    PleIndexMetadata ple_meta{};
    ple_meta.multipliers = {1, 2, 3};
    ple_meta.head_offsets.fill(0);
    ple_meta.head_vocab_sizes.fill(100);

    auto lane0 = ledger.allocate_lane();

    // 1. Validation: empty, invalid frontier, out-of-range
    std::vector<std::int32_t> empty_tokens;
    try {
        (void)ledger.begin_prefill_chunk(lane0, empty_tokens, 0, ple_meta);
        std::cerr << "Failed to reject empty prefill chunk\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    std::vector<std::int32_t> chunk4 = {10, 11, 12, 13};
    try {
        (void)ledger.begin_prefill_chunk(lane0, chunk4, 2, ple_meta); // expected frontier 0
        std::cerr << "Failed to reject frontier mismatch in prefill chunk\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    // 2. Abort transaction: leaves frontier and history unchanged
    auto prep0 = ledger.begin_prefill_chunk(lane0, chunk4, 0, ple_meta);
    if (!ledger.has_pending_transaction()) {
        std::cerr << "Pending transaction not flagged after begin_prefill_chunk\n";
        return 1;
    }
    ledger.abort_round(prep0.transaction_id);
    if (ledger.has_pending_transaction()) {
        std::cerr << "Pending transaction still active after abort_round\n";
        return 1;
    }
    if (ledger.committed_frontier(lane0) != 0) {
        std::cerr << "Frontier modified after aborted prefill chunk\n";
        return 1;
    }
    const auto& hist0 = ledger.lane_history(lane0);
    if (hist0.previous_token() != kPleBoundaryTokenId ||
        hist0.second_previous_token() != kPleBoundaryTokenId) {
        std::cerr << "History modified after aborted prefill chunk\n";
        return 1;
    }

    // 3. Rollback prepared prefill chunk: restores groups and clears tables
    const auto groups_before = ledger.available_physical_groups();
    auto prep1 = ledger.begin_prefill_chunk(lane0, chunk4, 0, ple_meta);
    ledger.rollback_prepared_round(prep1.transaction_id);
    if (ledger.available_physical_groups() != groups_before) {
        std::cerr << "Groups not restored after rollback_prepared_round\n";
        return 1;
    }

    // 4. Commit prefill chunk (T=4): advances frontier by 4, commits all 4 tokens to history
    FlashNextRuntimeAllocation dummy_alloc(plan);
    prep1 = ledger.begin_prefill_chunk(lane0, chunk4, 0, ple_meta);
    std::vector<LaneCommitDecision> accept = {{.accept = true}};
    ledger.commit_round(prep1.transaction_id, accept, dummy_alloc, nullptr);

    if (ledger.committed_frontier(lane0) != 4) {
        std::cerr << "Committed frontier expected 4, got " << ledger.committed_frontier(lane0) << "\n";
        return 1;
    }
    const auto& hist1 = ledger.lane_history(lane0);
    if (hist1.previous_token() != 13 || hist1.second_previous_token() != 12) {
        std::cerr << "History after T=4 prefill chunk commit mismatch: got ("
                  << hist1.previous_token() << ", " << hist1.second_previous_token() << ")\n";
        return 1;
    }

    // 5. Subsequent chunk (T=3): from index 4 -> 7
    std::vector<std::int32_t> chunk3 = {20, 21, 22};
    auto prep2 = ledger.begin_prefill_chunk(lane0, chunk3, 4, ple_meta);
    ledger.commit_round(prep2.transaction_id, accept, dummy_alloc, nullptr);
    if (ledger.committed_frontier(lane0) != 7) {
        std::cerr << "Committed frontier expected 7, got " << ledger.committed_frontier(lane0) << "\n";
        return 1;
    }
    const auto& hist2 = ledger.lane_history(lane0);
    if (hist2.previous_token() != 22 || hist2.second_previous_token() != 21) {
        std::cerr << "History after second chunk commit mismatch\n";
        return 1;
    }

    ledger.release_lane(lane0);
    std::cout << "PASS: test_ledger_prefill_chunk_cpu\n";
    return 0;
}

struct SyntheticFlashNextModel {
    ninfer::DeviceBuffer big_bf16_buf;
    ninfer::DeviceBuffer big_fp8_buf;
    ninfer::DeviceBuffer big_nvfp4_gate_codes_buf;
    ninfer::DeviceBuffer big_nvfp4_gate_scales_buf;
    ninfer::DeviceBuffer big_nvfp4_down_codes_buf;
    ninfer::DeviceBuffer big_nvfp4_down_scales_buf;
    ninfer::DeviceBuffer big_divisors_buf;
    std::vector<std::byte> ple_table_data;
    ninfer::targets::qwen3_8_flash_next::detail::TextModelView view;
};

SyntheticFlashNextModel make_synthetic_model(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    SyntheticFlashNextModel model;
    constexpr std::uint64_t kOutputHeadBytes = 248'320ULL * 2'560 * 2;
    model.big_bf16_buf = ninfer::DeviceBuffer(kOutputHeadBytes);
    model.big_bf16_buf.fill(0);

    constexpr std::uint64_t kFp8Codes = 16'384ULL * 2'560;
    constexpr std::uint64_t kFp8Bytes = ((kFp8Codes + 255U) & ~255ULL) + 16'384ULL * 4;
    model.big_fp8_buf = ninfer::DeviceBuffer(kFp8Bytes);
    model.big_fp8_buf.fill(0);
    std::vector<float> fp8_scales(16'384, 1.0f);
    model.big_fp8_buf.copy_from_host(fp8_scales.data(), fp8_scales.size() * sizeof(float), ((kFp8Codes + 255U) & ~255ULL));

    constexpr std::uint64_t gate_code_bytes_per_expert  = 1'280ULL * 2'560 / 2;
    constexpr std::uint64_t gate_scale_bytes_per_expert = 1'280ULL * 2'560 / 16;
    constexpr std::uint64_t down_code_bytes_per_expert  = 2'560ULL * 640 / 2;
    constexpr std::uint64_t down_scale_bytes_per_expert = 2'560ULL * 640 / 16;

    model.big_nvfp4_gate_codes_buf  = ninfer::DeviceBuffer(512 * gate_code_bytes_per_expert);
    model.big_nvfp4_gate_scales_buf = ninfer::DeviceBuffer(512 * gate_scale_bytes_per_expert);
    model.big_nvfp4_down_codes_buf  = ninfer::DeviceBuffer(512 * down_code_bytes_per_expert);
    model.big_nvfp4_down_scales_buf = ninfer::DeviceBuffer(512 * down_scale_bytes_per_expert);
    model.big_divisors_buf          = ninfer::DeviceBuffer(512 * sizeof(float));

    model.big_nvfp4_gate_codes_buf.fill(0x22);
    model.big_nvfp4_gate_scales_buf.fill(0x38);
    model.big_nvfp4_down_codes_buf.fill(0x22);
    model.big_nvfp4_down_scales_buf.fill(0x38);

    std::vector<float> divisors(512, 1.0f);
    model.big_divisors_buf.copy_from_host(divisors.data(), sizeof(divisors));

    constexpr std::uint64_t rows         = 1;
    constexpr std::uint64_t width        = 160;
    constexpr std::uint64_t scale_offset = 256;
    model.ple_table_data = std::vector<std::byte>(scale_offset + (width / 16) * 2, std::byte{0});
    std::fill_n(model.ple_table_data.begin(), width / 2, std::byte{0x88});
    for (std::uint8_t index = 0; index < 8; ++index) {
        model.ple_table_data[index] = static_cast<std::byte>(index * 2 | ((index * 2 + 1) << 4));
    }
    constexpr std::uint16_t half_point_five = 0x3800;
    for (std::size_t offset = scale_offset; offset < model.ple_table_data.size(); offset += 2) {
        std::memcpy(model.ple_table_data.data() + offset, &half_point_five, sizeof(half_point_five));
    }
    for (PleShardView& shard : model.view.ple.table.shards) {
        shard = make_ple_shard_view(model.ple_table_data, rows, width);
    }

    auto make_bf16_weight = [&](std::int32_t rows, std::int32_t cols) {
        ninfer::Weight w{};
        w.payload         = model.big_bf16_buf.p;
        w.payload_bytes   = static_cast<std::uint64_t>(rows) * cols * 2;
        w.qdata           = model.big_bf16_buf.p;
        w.qtype           = ninfer::QType::BF16_CTRL;
        w.layout          = ninfer::QuantLayout::Contiguous;
        w.n               = rows;
        w.k               = cols;
        w.ndim            = 2;
        w.shape[0]        = rows;
        w.shape[1]        = cols;
        w.padded_shape[0] = rows;
        w.padded_shape[1] = cols;
        return w;
    };

    auto make_fp8_weight = [&](std::int32_t rows, std::int32_t cols) {
        const std::uint64_t codes = static_cast<std::uint64_t>(rows) * cols;
        const std::uint64_t scale_off = (codes + 255U) & ~255ULL;
        ninfer::Weight w{};
        w.payload           = model.big_fp8_buf.p;
        w.payload_bytes     = model.big_fp8_buf.bytes;
        w.qdata             = model.big_fp8_buf.p;
        w.scales            = static_cast<const std::byte*>(model.big_fp8_buf.p) + scale_off;
        w.qtype             = ninfer::QType::FP8_E4M3FN_ROW_F32S;
        w.layout            = ninfer::QuantLayout::RowScale;
        w.scale_dtype       = ninfer::DType::FP32;
        w.group_size        = cols;
        w.group             = cols;
        w.n                 = rows;
        w.k                 = cols;
        w.ndim              = 2;
        w.shape[0]          = rows;
        w.shape[1]          = cols;
        w.shape[2]          = 1;
        w.shape[3]          = 1;
        w.padded_shape[0]   = rows;
        w.padded_shape[1]   = cols;
        w.padded_shape[2]   = 1;
        w.padded_shape[3]   = 1;
        w.scale_ne[0]       = rows;
        w.scale_ne[1]       = 1;
        w.scale_ne[2]       = 1;
        w.scale_ne[3]       = 1;
        w.scale_nb[0]       = 4;
        w.scale_nb[1]       = static_cast<std::int64_t>(rows) * 4;
        w.scale_nb[2]       = static_cast<std::int64_t>(rows) * 4;
        w.scale_nb[3]       = static_cast<std::int64_t>(rows) * 4;
        return w;
    };

    auto make_tensor = [&](ninfer::DType dt, std::initializer_list<std::int32_t> dims) {
        if (dt == ninfer::DType::BF16) {
            return ninfer::Tensor(model.big_bf16_buf.p, dt, dims);
        } else {
            return ninfer::Tensor(model.big_fp8_buf.p, dt, dims);
        }
    };

    model.view.token_embedding = make_bf16_weight(248'320, 2'560);
    model.view.output_head     = make_bf16_weight(248'320, 2'560);

    model.view.ple.convolution      = make_tensor(ninfer::DType::BF16, {10'240, 4});
    model.view.ple.key_projection   = make_bf16_weight(10'240, 2'560);
    model.view.ple.conv_norm        = make_tensor(ninfer::DType::BF16, {10'240});
    model.view.ple.key_norm         = make_tensor(ninfer::DType::BF16, {10'240});
    model.view.ple.query_norm       = make_tensor(ninfer::DType::BF16, {10'240});
    model.view.ple.value_projection = make_bf16_weight(2'560, 2'560);

    model.view.final_mixer.norm           = make_tensor(ninfer::DType::BF16, {10'240});
    model.view.final_mixer.input_mix_down = make_bf16_weight(320, 10'240);
    model.view.final_mixer.input_mix_up   = make_bf16_weight(10'240, 320);

    for (std::size_t l = 0; l < 48; ++l) {
        auto& layer = model.view.layers[l];
        layer.attention_hyper.block_inject   = make_bf16_weight(4, 10'240);
        layer.attention_hyper.norm           = make_tensor(ninfer::DType::BF16, {10'240});
        layer.attention_hyper.input_mix_down = make_bf16_weight(320, 10'240);
        layer.attention_hyper.input_mix_up   = make_bf16_weight(10'240, 320);

        layer.mlp_hyper.block_inject   = make_bf16_weight(4, 10'240);
        layer.mlp_hyper.norm           = make_tensor(ninfer::DType::BF16, {10'240});
        layer.mlp_hyper.input_mix_down = make_bf16_weight(320, 10'240);
        layer.mlp_hyper.input_mix_up   = make_bf16_weight(10'240, 320);

        layer.moe.router             = make_bf16_weight(512, 2'560);
        layer.moe.shared_down        = make_bf16_weight(2'560, 640);
        layer.moe.shared_gate        = make_bf16_weight(640, 2'560);
        layer.moe.shared_up          = make_bf16_weight(640, 2'560);
        layer.moe.shared_gate_weight = make_bf16_weight(1, 2'560);
        layer.moe.expert_gate_up     = Nvfp4ExpertBankView{
            .codes                  = static_cast<const std::byte*>(model.big_nvfp4_gate_codes_buf.p),
            .scales                 = static_cast<const std::byte*>(model.big_nvfp4_gate_scales_buf.p),
            .weight_scale_divisors  = static_cast<const float*>(model.big_divisors_buf.p),
            .experts                = 512,
            .rows                   = 1'280,
            .columns                = 2'560,
            .code_bytes_per_expert  = gate_code_bytes_per_expert,
            .scale_bytes_per_expert = gate_scale_bytes_per_expert,
        };
        layer.moe.expert_down        = Nvfp4ExpertBankView{
            .codes                  = static_cast<const std::byte*>(model.big_nvfp4_down_codes_buf.p),
            .scales                 = static_cast<const std::byte*>(model.big_nvfp4_down_scales_buf.p),
            .weight_scale_divisors  = static_cast<const float*>(model.big_divisors_buf.p),
            .experts                = 512,
            .rows                   = 2'560,
            .columns                = 640,
            .code_bytes_per_expert  = down_code_bytes_per_expert,
            .scale_bytes_per_expert = down_scale_bytes_per_expert,
        };
    }

    for (std::size_t i = 0; i < kGdnLayers; ++i) {
        auto& gdn = model.view.gdn[i];
        gdn.a_log             = make_tensor(ninfer::DType::BF16, {48});
        gdn.convolution       = make_tensor(ninfer::DType::BF16, {10'240, 4});
        gdn.dt_bias           = make_tensor(ninfer::DType::BF16, {48});
        gdn.a_b_projection    = make_bf16_weight(96, 2'560);
        gdn.norm              = make_tensor(ninfer::DType::BF16, {128});
        gdn.query_key_value_z = make_fp8_weight(16'384, 2'560);
        gdn.output            = make_fp8_weight(2'560, 6'144);
    }

    for (std::size_t i = 0; i < kFullAttentionLayers; ++i) {
        auto& att = model.view.full_attention[i];
        att.indexer_query_key    = make_bf16_weight(640, 2'560);
        att.indexer_key_norm     = make_tensor(ninfer::DType::BF16, {128});
        att.indexer_query_norm   = make_tensor(ninfer::DType::BF16, {128});
        att.key_norm             = make_tensor(ninfer::DType::BF16, {256});
        att.query_norm           = make_tensor(ninfer::DType::BF16, {256});
        att.query_gate_key_value = make_fp8_weight(13'312, 2'560);
        att.output               = make_fp8_weight(2'560, 6'144);
    }

    return model;
}

int test_prefill_chunk_executor(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);

        auto synthetic_model = make_synthetic_model(device);

        FlashNextRuntimeConfig cfg{
            .max_concurrency     = 2,
            .max_context         = 512,
            .state_slot_capacity = 4,
            .prefill_chunk       = 512,
        };
        const auto curve = flash_next_capacity_curve(cfg);
        auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

        // 1. Sequential execution of 4 tokens on allocation 1
        FlashNextRuntimeAllocation alloc_seq(plan);
        alloc_seq.initialize(device.stream);
        FlashNextTextExecutor exec_seq(synthetic_model.view, ple_meta, device, alloc_seq);
        auto lane_seq = exec_seq.allocate_lane();

        const std::vector<std::int32_t> tokens = {100, 101, 102, 103};
        const std::vector<std::array<std::int32_t, 3>> positions = {
            {0, 0, 0}, {1, 1, 1}, {2, 2, 2}, {3, 3, 3}};

        for (std::size_t t = 0; t < 4; ++t) {
            LaneStepRequest req{
                .handle          = lane_seq,
                .token_id        = tokens[t],
                .token_index     = static_cast<std::int32_t>(t),
                .mrope_positions = positions[t],
            };
            auto round = exec_seq.execute_round(std::span(&req, 1));
            std::vector<LaneCommitDecision> decision = {{.accept = true}};
            round.commit(decision);
        }
        device.synchronize();

        const std::size_t persistent_bytes =
            plan.total_device_bytes - plan.workspace_bytes - plan.cuda_graph_allowance_bytes;
        std::vector<std::uint8_t> seq_storage(persistent_bytes);
        CUDA_CHECK(cudaMemcpy(seq_storage.data(), alloc_seq.state_view().qsa_attention_caches[0].key_pages.data,
                              persistent_bytes, cudaMemcpyDeviceToHost));

        std::vector<std::uint16_t> seq_logits(248'320);
        CUDA_CHECK(cudaMemcpy(seq_logits.data(), alloc_seq.round_tensors().logits.data,
                              seq_logits.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

        std::vector<std::uint16_t> seq_hidden(2'560);
        CUDA_CHECK(cudaMemcpy(seq_hidden.data(), alloc_seq.round_tensors().final_hidden.data,
                              seq_hidden.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

        // 2. Chunked execution of T=4 tokens on allocation 2
        FlashNextRuntimeAllocation alloc_chunk(plan);
        alloc_chunk.initialize(device.stream);
        FlashNextTextExecutor exec_chunk(synthetic_model.view, ple_meta, device, alloc_chunk);
        auto lane_chunk = exec_chunk.allocate_lane();

        auto chunk_round = exec_chunk.execute_prefill_chunk(lane_chunk, tokens, positions, 0);
        std::vector<LaneCommitDecision> decision = {{.accept = true}};
        chunk_round.commit(decision);
        device.synchronize();

        std::vector<std::uint8_t> chunk_storage(persistent_bytes);
        CUDA_CHECK(cudaMemcpy(chunk_storage.data(), alloc_chunk.state_view().qsa_attention_caches[0].key_pages.data,
                              persistent_bytes, cudaMemcpyDeviceToHost));

        std::vector<std::uint16_t> chunk_logits(248'320);
        CUDA_CHECK(cudaMemcpy(chunk_logits.data(), alloc_chunk.round_tensors().logits.data,
                              chunk_logits.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

        std::vector<std::uint16_t> chunk_hidden(2'560);
        CUDA_CHECK(cudaMemcpy(chunk_hidden.data(), alloc_chunk.round_tensors().final_hidden.data,
                              chunk_hidden.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

        // 3. Compare final hidden and logits equality within numerical tolerances
        auto bf16_to_f = [](std::uint16_t v) {
            return std::bit_cast<float>(static_cast<std::uint32_t>(v) << 16U);
        };
        double hid_diff = 0.0, hid_ref = 0.0;
        for (std::size_t i = 0; i < seq_hidden.size(); ++i) {
            float a = bf16_to_f(seq_hidden[i]);
            float b = bf16_to_f(chunk_hidden[i]);
            hid_diff += (a - b) * (a - b);
            hid_ref += a * a;
        }
        double hid_rel_l2 = std::sqrt(hid_diff) / std::max(1e-6, std::sqrt(hid_ref));
        if (hid_rel_l2 > 2e-2) {
            std::cerr << "Final hidden rel-L2 mismatch: " << hid_rel_l2 << " > 2e-2\n";
            return 1;
        }

        double log_diff = 0.0, log_ref = 0.0;
        std::size_t seq_argmax = 0, chunk_argmax = 0;
        float seq_max = -1e30F, chunk_max = -1e30F;
        for (std::size_t i = 0; i < seq_logits.size(); ++i) {
            float a = bf16_to_f(seq_logits[i]);
            float b = bf16_to_f(chunk_logits[i]);
            log_diff += (a - b) * (a - b);
            log_ref += a * a;
            if (a > seq_max) { seq_max = a; seq_argmax = i; }
            if (b > chunk_max) { chunk_max = b; chunk_argmax = i; }
        }
        double log_rel_l2 = std::sqrt(log_diff) / std::max(1e-6, std::sqrt(log_ref));
        if (log_rel_l2 > 2e-2) {
            std::cerr << "Logits rel-L2 mismatch: " << log_rel_l2 << " > 2e-2\n";
            return 1;
        }
        if (seq_argmax != chunk_argmax) {
            std::cerr << "Argmax mismatch: seq=" << seq_argmax << " chunk=" << chunk_argmax << "\n";
            return 1;
        }

        exec_seq.release_lane(lane_seq);
        exec_chunk.release_lane(lane_chunk);

        std::cout << "PASS: test_prefill_chunk_executor\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_prefill_chunk_executor exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "test_prefill_chunk_executor unknown exception\n";
        return 1;
    }
}

int test_prefill_chunk_workspace_envelope(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    // Regression: a full prefill_chunk-wide chunk exhausted the workspace arena (std::bad_alloc
    // from DeviceArena::alloc inside layer 0's GDN block) because the capacity estimate did not
    // include the executor's staging tensors. max_concurrency=1 keeps the decode estimate small so
    // it cannot mask a prefill under-estimate; two chunks cover the non-zero frontier as well.
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);

        auto synthetic_model = make_synthetic_model(device);

        constexpr std::int32_t kChunk = 128;
        FlashNextRuntimeConfig cfg{
            .max_concurrency     = 1,
            .max_context         = 512,
            .state_slot_capacity = 2,
            .prefill_chunk       = kChunk,
        };
        const auto curve = flash_next_capacity_curve(cfg);
        auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

        FlashNextRuntimeAllocation alloc(plan);
        alloc.initialize(device.stream);
        FlashNextTextExecutor exec(synthetic_model.view, ple_meta, device, alloc);
        auto lane = exec.allocate_lane();

        for (std::int32_t chunk = 0; chunk < 2; ++chunk) {
            std::vector<std::int32_t> tokens(kChunk);
            std::vector<std::array<std::int32_t, 3>> positions(kChunk);
            for (std::int32_t t = 0; t < kChunk; ++t) {
                const std::int32_t index = chunk * kChunk + t;
                tokens[t]                = 100 + index;
                positions[t]             = {index, index, index};
            }
            auto round = exec.execute_prefill_chunk(lane, tokens, positions, chunk * kChunk);
            std::vector<LaneCommitDecision> decision = {{.accept = true}};
            round.commit(decision);
        }
        device.synchronize();

        const std::size_t peak     = alloc.workspace().peak_used();
        const std::size_t capacity = alloc.workspace().capacity();
        std::cout << "prefill chunk T=" << kChunk << " workspace peak " << peak << " / capacity "
                  << capacity << " bytes\n";
        if (peak > capacity) {
            std::cerr << "Workspace peak exceeds capacity\n";
            return 1;
        }

        exec.release_lane(lane);
        std::cout << "PASS: test_prefill_chunk_workspace_envelope\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_prefill_chunk_workspace_envelope exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "test_prefill_chunk_workspace_envelope unknown exception\n";
        return 1;
    }
}

int test_cuda_graph_decode_equivalence(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);

        auto synthetic_model = make_synthetic_model(device);

        const std::vector<std::uint32_t> test_batches = {1, 2, 4, 8};
        for (std::uint32_t B : test_batches) {
            FlashNextRuntimeConfig cfg_eager{
                .max_concurrency     = B,
                .max_context         = 512,
                .state_slot_capacity = 2 * B,
                .prefill_chunk       = 512,
                .use_cuda_graph      = false,
            };
            const auto curve_eager = flash_next_capacity_curve(cfg_eager);
            auto plan_eager =
                finalize_flash_next_runtime_plan(cfg_eager, curve_eager.maximum_main_page_groups);

            FlashNextRuntimeAllocation alloc_eager(plan_eager);
            alloc_eager.initialize(device.stream);
            FlashNextTextExecutor exec_eager(synthetic_model.view, ple_meta, device, alloc_eager);

            FlashNextRuntimeConfig cfg_graph{
                .max_concurrency     = B,
                .max_context         = 512,
                .state_slot_capacity = 2 * B,
                .prefill_chunk       = 512,
                .use_cuda_graph      = true,
            };
            const auto curve_graph = flash_next_capacity_curve(cfg_graph);
            auto plan_graph =
                finalize_flash_next_runtime_plan(cfg_graph, curve_graph.maximum_main_page_groups);

            FlashNextRuntimeAllocation alloc_graph(plan_graph);
            alloc_graph.initialize(device.stream);
            FlashNextTextExecutor exec_graph(synthetic_model.view, ple_meta, device, alloc_graph);

            // DIAG: captured graphs, then eager body — isolates capture-time state pollution
            // from replay itself.
            FlashNextRuntimeAllocation alloc_mixed(plan_graph);
            alloc_mixed.initialize(device.stream);
            FlashNextTextExecutor exec_mixed(synthetic_model.view, ple_meta, device, alloc_mixed);
            exec_mixed.set_use_cuda_graph(false);

            const std::size_t expected_cap =
                flash_next_text_decode_workspace_capacity_bytes(plan_graph.maximum_blocks, B);
            const std::size_t peak_capture = alloc_graph.workspace().peak_used();
            if (((peak_capture + 255U) & ~255ULL) != expected_cap) {
                std::cerr << "Workspace peak_used after capture mismatch at B=" << B << ": got "
                          << peak_capture << " (aligned " << ((peak_capture + 255U) & ~255ULL)
                          << ") expected " << expected_cap << "\n";
                return 1;
            }

            std::vector<LaneHandle> lanes_eager;
            std::vector<LaneHandle> lanes_graph;
            std::vector<LaneHandle> lanes_mixed;
            for (std::uint32_t b = 0; b < B; ++b) {
                lanes_eager.push_back(exec_eager.allocate_lane());
                lanes_graph.push_back(exec_graph.allocate_lane());
                lanes_mixed.push_back(exec_mixed.allocate_lane());
            }

            for (std::size_t step = 0; step < 4; ++step) {
                std::vector<LaneStepRequest> reqs_eager(B);
                std::vector<LaneStepRequest> reqs_graph(B);
                for (std::uint32_t b = 0; b < B; ++b) {
                    reqs_eager[b] = LaneStepRequest{
                        .handle          = lanes_eager[b],
                        .token_id        = static_cast<std::int32_t>(100 + step * 10 + b),
                        .token_index     = static_cast<std::int32_t>(step),
                        .mrope_positions = {static_cast<std::int32_t>(step),
                                            static_cast<std::int32_t>(step),
                                            static_cast<std::int32_t>(step)},
                        .sampling        = {.temperature = 0.0F, .top_p = 1.0F},
                    };
                    reqs_graph[b] = LaneStepRequest{
                        .handle          = lanes_graph[b],
                        .token_id        = static_cast<std::int32_t>(100 + step * 10 + b),
                        .token_index     = static_cast<std::int32_t>(step),
                        .mrope_positions = {static_cast<std::int32_t>(step),
                                            static_cast<std::int32_t>(step),
                                            static_cast<std::int32_t>(step)},
                        .sampling        = {.temperature = 0.0F, .top_p = 1.0F},
                    };
                }

                if (B == 1 && step == 0) {
                    // NAN PROBE: which stage first produces NaN on the synthetic model?
                    std::string first_nan;
                    std::size_t stages = 0;
                    FlashNextDecodeStateSink probe;
                    probe.on_state = [&](std::string_view name, const ninfer::Tensor& t) {
                        ++stages;
                        if (!first_nan.empty() || t.dtype != ninfer::DType::BF16) { return; }
                        std::size_t count = 1;
                        for (int d = 0; d < 4; ++d) { if (t.ne[d] > 0) { count *= static_cast<std::size_t>(t.ne[d]); } }
                        std::vector<std::uint16_t> host(count);
                        CUDA_CHECK(cudaMemcpy(host.data(), t.data, count * sizeof(std::uint16_t),
                                              cudaMemcpyDeviceToHost));
                        std::size_t nans = 0;
                        for (auto v : host) {
                            const float f = std::bit_cast<float>(static_cast<std::uint32_t>(v) << 16U);
                            if (std::isnan(f) || std::isinf(f)) { ++nans; }
                        }
                        if (nans > 0) {
                            first_nan = std::string(name) + " (" + std::to_string(nans) + "/" +
                                        std::to_string(count) + " non-finite)";
                        }
                    };
                    FlashNextRuntimeAllocation alloc_probe(plan_eager);
                    alloc_probe.initialize(device.stream);
                    FlashNextTextExecutor exec_probe(synthetic_model.view, ple_meta, device, alloc_probe);
                    auto lane_probe = exec_probe.allocate_lane();
                    LaneStepRequest req_probe = reqs_eager[0];
                    req_probe.handle          = lane_probe;
                    auto round_probe = exec_probe.execute_round(std::span<const LaneStepRequest>(&req_probe, 1), &probe);
                    std::vector<LaneCommitDecision> d1 = {{.accept = true}};
                    round_probe.commit(d1);
                    device.synchronize();
                    std::cout << "NAN PROBE: stages=" << stages << " first non-finite stage: "
                              << (first_nan.empty() ? "none" : first_nan) << "\n";
                    exec_probe.release_lane(lane_probe);
                }
                std::vector<LaneStepRequest> reqs_mixed = reqs_graph;
                for (std::uint32_t b = 0; b < B; ++b) { reqs_mixed[b].handle = lanes_mixed[b]; }
                auto round_eager = exec_eager.execute_round(reqs_eager);
                auto round_graph = exec_graph.execute_round(reqs_graph);
                auto round_mixed = exec_mixed.execute_round(reqs_mixed);

                // Check sampled tokens equivalence
                auto eager_tokens = round_eager.sampled_tokens();
                auto graph_tokens = round_graph.sampled_tokens();
                for (std::uint32_t b = 0; b < B; ++b) {
                    if (eager_tokens[b] != graph_tokens[b]) {
                        std::cerr << "Sampled token mismatch at B=" << B << " step=" << step
                                  << " b=" << b << ": eager=" << eager_tokens[b]
                                  << " graph=" << graph_tokens[b] << "\n";
                        return 1;
                    }
                }

                // Check logits equivalence
                std::vector<std::uint16_t> logits_eager(248'320 * B);
                std::vector<std::uint16_t> logits_graph(248'320 * B);
                CUDA_CHECK(cudaMemcpy(logits_eager.data(), round_eager.logits().data,
                                      logits_eager.size() * sizeof(std::uint16_t),
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(logits_graph.data(), round_graph.logits().data,
                                      logits_graph.size() * sizeof(std::uint16_t),
                                      cudaMemcpyDeviceToHost));

                auto bf16_to_f = [](std::uint16_t v) {
                    return std::bit_cast<float>(static_cast<std::uint32_t>(v) << 16U);
                };
                std::vector<std::uint16_t> logits_mixed(248'320 * B);
                CUDA_CHECK(cudaMemcpy(logits_mixed.data(), round_mixed.logits().data,
                                      logits_mixed.size() * sizeof(std::uint16_t),
                                      cudaMemcpyDeviceToHost));
                float max_eg = 0.0F, max_em = 0.0F, max_gm = 0.0F;
                std::size_t n_eg = 0, n_em = 0, n_gm = 0;
                for (std::size_t i = 0; i < logits_eager.size(); ++i) {
                    const float e = bf16_to_f(logits_eager[i]);
                    const float g = bf16_to_f(logits_graph[i]);
                    const float m = bf16_to_f(logits_mixed[i]);
                    if (e != g) { ++n_eg; max_eg = std::max(max_eg, std::abs(e - g)); }
                    if (e != m) { ++n_em; max_em = std::max(max_em, std::abs(e - m)); }
                    if (g != m) { ++n_gm; max_gm = std::max(max_gm, std::abs(g - m)); }
                }
                std::size_t n_nan = 0;
                for (std::size_t i = 0; i < logits_eager.size(); ++i) { if (std::isnan(bf16_to_f(logits_eager[i]))) { ++n_nan; } }
                std::cout << "DIAG nan(eager)=" << n_nan << " ";
                std::cout << "DIAG B=" << B << " step=" << step << " eager-vs-graph: " << n_eg
                          << " differ (max " << max_eg << ") | eager-vs-mixed: " << n_em
                          << " differ (max " << max_em << ") | graph-vs-mixed: " << n_gm
                          << " differ (max " << max_gm << ")\n";
                if (max_eg > 1e-3F) {
                    std::cerr << "Logits diff exceeded at B=" << B << " step=" << step << "\n";
                    return 1;
                }

                std::vector<LaneCommitDecision> decisions(B, {.accept = true});
                round_eager.commit(decisions);
                round_graph.commit(decisions);
                round_mixed.commit(decisions);
                device.synchronize();

                if (((alloc_graph.workspace().peak_used() + 255U) & ~255ULL) != expected_cap ||
                    alloc_graph.workspace().peak_used() != peak_capture) {
                    std::cerr << "Workspace peak_used after replay mismatch at B=" << B << ": got "
                              << alloc_graph.workspace().peak_used() << " expected " << peak_capture
                              << "\n";
                    return 1;
                }
            }

            for (std::uint32_t b = 0; b < B; ++b) {
                exec_eager.release_lane(lanes_eager[b]);
                exec_graph.release_lane(lanes_graph[b]);
            }
        }
        std::cout << "PASS: test_cuda_graph_decode_equivalence\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_cuda_graph_decode_equivalence exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "test_cuda_graph_decode_equivalence unknown exception\n";
        return 1;
    }
}

int test_cuda_graph_frontier_masking_and_churn(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);

        auto synthetic_model = make_synthetic_model(device);

        FlashNextRuntimeConfig cfg{
            .max_concurrency     = 4,
            .max_context         = 512,
            .state_slot_capacity = 8,
            .prefill_chunk       = 512,
            .use_cuda_graph      = true,
        };
        const auto curve = flash_next_capacity_curve(cfg);
        auto plan = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

        FlashNextRuntimeAllocation alloc(plan);
        alloc.initialize(device.stream);
        FlashNextTextExecutor exec(synthetic_model.view, ple_meta, device, alloc);

        // 1. Allocate 4 lanes
        auto lane0 = exec.allocate_lane();
        auto lane1 = exec.allocate_lane();
        auto lane2 = exec.allocate_lane();
        auto lane3 = exec.allocate_lane();

        // 2. Prefill lane0 up to frontier 250 (nearing group boundary 256)
        std::vector<std::int32_t> prefill_tokens(250, 100);
        std::vector<std::array<std::int32_t, 3>> prefill_pos(250);
        for (std::int32_t i = 0; i < 250; ++i) { prefill_pos[i] = {i, i, i}; }
        auto pr = exec.execute_prefill_chunk(lane0, prefill_tokens, prefill_pos, 0);
        std::vector<LaneCommitDecision> accept1 = {{.accept = true}};
        pr.commit(accept1);
        device.synchronize();

        // 3. Step lane0 across boundary (N=250 -> 260) with batch B=1 using graph replay
        for (std::int32_t step = 250; step < 260; ++step) {
            LaneStepRequest req{
                .handle          = lane0,
                .token_id        = 100 + step,
                .token_index     = step,
                .mrope_positions = {step, step, step},
                .sampling        = {.temperature = 0.0F, .top_p = 1.0F},
            };
            auto round = exec.execute_round(std::span(&req, 1));
            round.commit(accept1);
            device.synchronize();
        }

        if (exec.committed_frontier(lane0) != 260) {
            std::cerr << "Frontier expected 260 after crossing boundary, got "
                      << exec.committed_frontier(lane0) << "\n";
            return 1;
        }

        // 4. Lane churn and slot recycling under graph replay:
        // Release lane 1 and lane 2
        exec.release_lane(lane1);
        exec.release_lane(lane2);

        // Step remaining lanes (lane 0 and lane 3) as B=2 round
        std::vector<LaneStepRequest> reqs_b2 = {
            {.handle          = lane0,
             .token_id        = 400,
             .token_index     = 260,
             .mrope_positions = {260, 260, 260},
             .sampling        = {.temperature = 0.0F, .top_p = 1.0F}},
            {.handle          = lane3,
             .token_id        = 401,
             .token_index     = 0,
             .mrope_positions = {0, 0, 0},
             .sampling        = {.temperature = 0.0F, .top_p = 1.0F}},
        };
        auto round_b2 = exec.execute_round(reqs_b2);
        std::vector<LaneCommitDecision> accept2(2, {.accept = true});
        round_b2.commit(accept2);
        device.synchronize();

        // Reallocate 2 new lanes (will recycle previously freed state slots)
        auto lane4 = exec.allocate_lane();
        auto lane5 = exec.allocate_lane();

        // Step all 4 lanes as B=4 round
        std::vector<LaneStepRequest> reqs_b4 = {
            {.handle          = lane0,
             .token_id        = 500,
             .token_index     = 261,
             .mrope_positions = {261, 261, 261},
             .sampling        = {.temperature = 0.0F, .top_p = 1.0F}},
            {.handle          = lane3,
             .token_id        = 501,
             .token_index     = 1,
             .mrope_positions = {1, 1, 1},
             .sampling        = {.temperature = 0.0F, .top_p = 1.0F}},
            {.handle          = lane4,
             .token_id        = 502,
             .token_index     = 0,
             .mrope_positions = {0, 0, 0},
             .sampling        = {.temperature = 0.0F, .top_p = 1.0F}},
            {.handle          = lane5,
             .token_id        = 503,
             .token_index     = 0,
             .mrope_positions = {0, 0, 0},
             .sampling        = {.temperature = 0.0F, .top_p = 1.0F}},
        };
        auto round_b4 = exec.execute_round(reqs_b4);
        std::vector<LaneCommitDecision> accept4(4, {.accept = true});
        round_b4.commit(accept4);
        device.synchronize();

        exec.release_lane(lane0);
        exec.release_lane(lane3);
        exec.release_lane(lane4);
        exec.release_lane(lane5);

        std::cout << "PASS: test_cuda_graph_frontier_masking_and_churn\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_cuda_graph_frontier_masking_and_churn exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "test_cuda_graph_frontier_masking_and_churn unknown exception\n";
        return 1;
    }
}

int test_cuda_graph_timing_benchmark(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);

        auto synthetic_model = make_synthetic_model(device);

        const std::vector<std::uint32_t> benchmark_batches = {1, 8};
        constexpr int kWarmupRounds = 5;
        constexpr int kBenchRounds  = 30;

        std::cout << "--- Decode Performance Benchmark (Synthetic Model) ---\n";
        for (std::uint32_t B : benchmark_batches) {
            // 1. Eager mode
            FlashNextRuntimeConfig cfg_eager{
                .max_concurrency     = B,
                .max_context         = 512,
                .state_slot_capacity = 2 * B,
                .prefill_chunk       = 512,
                .use_cuda_graph      = false,
            };
            const auto curve_eager = flash_next_capacity_curve(cfg_eager);
            auto plan_eager =
                finalize_flash_next_runtime_plan(cfg_eager, curve_eager.maximum_main_page_groups);
            FlashNextRuntimeAllocation alloc_eager(plan_eager);
            alloc_eager.initialize(device.stream);
            FlashNextTextExecutor exec_eager(synthetic_model.view, ple_meta, device, alloc_eager);
            std::vector<LaneHandle> lanes_eager;
            for (std::uint32_t b = 0; b < B; ++b) {
                lanes_eager.push_back(exec_eager.allocate_lane());
            }

            std::vector<LaneStepRequest> reqs_eager(B);
            std::vector<LaneCommitDecision> accept(B, {.accept = true});

            int step_eager = 0;
            for (int r = 0; r < kWarmupRounds; ++r, ++step_eager) {
                for (std::uint32_t b = 0; b < B; ++b) {
                    reqs_eager[b] = {
                        .handle          = lanes_eager[b],
                        .token_id        = static_cast<std::int32_t>(100 + b),
                        .token_index     = step_eager,
                        .mrope_positions = {step_eager, step_eager, step_eager},
                        .sampling        = {.temperature = 0.0F, .top_p = 1.0F},
                    };
                }
                auto rd = exec_eager.execute_round(reqs_eager);
                rd.commit(accept);
                device.synchronize();
            }

            auto start_eager = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < kBenchRounds; ++r, ++step_eager) {
                for (std::uint32_t b = 0; b < B; ++b) {
                    reqs_eager[b] = {
                        .handle          = lanes_eager[b],
                        .token_id        = static_cast<std::int32_t>(100 + b),
                        .token_index     = step_eager,
                        .mrope_positions = {step_eager, step_eager, step_eager},
                        .sampling        = {.temperature = 0.0F, .top_p = 1.0F},
                    };
                }
                auto rd = exec_eager.execute_round(reqs_eager);
                rd.commit(accept);
            }
            device.synchronize();
            auto end_eager = std::chrono::high_resolution_clock::now();
            double eager_us_per_round =
                std::chrono::duration<double, std::micro>(end_eager - start_eager).count() /
                kBenchRounds;
            double eager_tok_per_sec = (B * 1e6) / eager_us_per_round;

            // 2. Graph Replay mode
            FlashNextRuntimeConfig cfg_graph{
                .max_concurrency     = B,
                .max_context         = 512,
                .state_slot_capacity = 2 * B,
                .prefill_chunk       = 512,
                .use_cuda_graph      = true,
            };
            const auto curve_graph = flash_next_capacity_curve(cfg_graph);
            auto plan_graph =
                finalize_flash_next_runtime_plan(cfg_graph, curve_graph.maximum_main_page_groups);
            FlashNextRuntimeAllocation alloc_graph(plan_graph);
            alloc_graph.initialize(device.stream);
            FlashNextTextExecutor exec_graph(synthetic_model.view, ple_meta, device, alloc_graph);
            std::vector<LaneHandle> lanes_graph;
            for (std::uint32_t b = 0; b < B; ++b) {
                lanes_graph.push_back(exec_graph.allocate_lane());
            }

            std::vector<LaneStepRequest> reqs_graph(B);
            int step_graph = 0;
            for (int r = 0; r < kWarmupRounds; ++r, ++step_graph) {
                for (std::uint32_t b = 0; b < B; ++b) {
                    reqs_graph[b] = {
                        .handle          = lanes_graph[b],
                        .token_id        = static_cast<std::int32_t>(100 + b),
                        .token_index     = step_graph,
                        .mrope_positions = {step_graph, step_graph, step_graph},
                        .sampling        = {.temperature = 0.0F, .top_p = 1.0F},
                    };
                }
                auto rd = exec_graph.execute_round(reqs_graph);
                rd.commit(accept);
                device.synchronize();
            }

            auto start_graph = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < kBenchRounds; ++r, ++step_graph) {
                for (std::uint32_t b = 0; b < B; ++b) {
                    reqs_graph[b] = {
                        .handle          = lanes_graph[b],
                        .token_id        = static_cast<std::int32_t>(100 + b),
                        .token_index     = step_graph,
                        .mrope_positions = {step_graph, step_graph, step_graph},
                        .sampling        = {.temperature = 0.0F, .top_p = 1.0F},
                    };
                }
                auto rd = exec_graph.execute_round(reqs_graph);
                rd.commit(accept);
            }
            device.synchronize();
            auto end_graph = std::chrono::high_resolution_clock::now();
            double graph_us_per_round =
                std::chrono::duration<double, std::micro>(end_graph - start_graph).count() /
                kBenchRounds;
            double graph_tok_per_sec = (B * 1e6) / graph_us_per_round;

            std::cout << "B=" << B << " | Eager: " << eager_us_per_round << " us/round ("
                      << eager_tok_per_sec << " tok/s) | Graph Replay: " << graph_us_per_round
                      << " us/round (" << graph_tok_per_sec << " tok/s) | Speedup: "
                      << (eager_us_per_round / graph_us_per_round) << "x\n";
        }
        std::cout << "--------------------------------------------------------\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_cuda_graph_timing_benchmark exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "test_cuda_graph_timing_benchmark unknown exception\n";
        return 1;
    }
}

} // namespace

int main() {
    if (test_ledger_cpu() != 0) return 1;
    if (test_ledger_prefill_chunk_cpu() != 0) return 1;
    if (test_ple_boundary_lifecycle_cpu() != 0) return 1;

    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || device_count == 0) {
        std::cout << "SKIP: CUDA device tests (no usable device)\n";
        return 0;
    }
    CUDA_CHECK(count_error);

    ninfer::DeviceContext device(0);

    if (test_cuda_ledger_and_executor(device) != 0) return 1;
    if (test_prefill_chunk_executor(device) != 0) return 1;
    if (test_prefill_chunk_workspace_envelope(device) != 0) return 1;
    if (test_cuda_graph_decode_equivalence(device) != 0) return 1;
    if (test_cuda_graph_frontier_masking_and_churn(device) != 0) return 1;
    if (test_cuda_graph_timing_benchmark(device) != 0) return 1;

    std::cout << "OK Flash-Next Text Executor\n";
    return 0;
}
