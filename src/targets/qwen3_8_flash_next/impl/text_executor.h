#pragma once

#include "core/device.h"
#include "core/tensor.h"
#include "targets/qwen3_8_flash_next/impl/lane_ledger.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"
#include "targets/qwen3_8_flash_next/impl/ple_index.h"
#include "targets/qwen3_8_flash_next/impl/ple_pipeline.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"
#include "targets/qwen3_8_flash_next/impl/runtime_state.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

class PendingRound {
public:
    PendingRound() noexcept = default;

    ~PendingRound() noexcept {
        if (valid()) { abort(); }
    }

    PendingRound(const PendingRound&)            = delete;
    PendingRound& operator=(const PendingRound&) = delete;
    PendingRound(PendingRound&& other) noexcept;
    PendingRound& operator=(PendingRound&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint32_t batch_size() const;
    [[nodiscard]] Tensor logits() const;
    [[nodiscard]] Tensor final_hidden() const;

    void commit(std::span<const LaneCommitDecision> decisions);
    void abort() noexcept;

private:
    friend class FlashNextTextExecutor;
    PendingRound(FlashNextTextExecutor* owner, std::uint64_t transaction_id,
                 std::uint32_t batch_size, Tensor logits, Tensor final_hidden) noexcept;

    FlashNextTextExecutor* owner_ = nullptr;
    std::uint64_t transaction_id_ = 0;
    std::uint32_t batch_size_     = 0;
    Tensor logits_{};
    Tensor final_hidden_{};
};

class FlashNextTextExecutor {
public:
    FlashNextTextExecutor(const TextModelView& model, PleIndexMetadata ple_metadata,
                          DeviceContext& device, FlashNextRuntimeAllocation& allocation,
                          std::int64_t eos_token = 151643);

    ~FlashNextTextExecutor() = default;

    FlashNextTextExecutor(const FlashNextTextExecutor&)            = delete;
    FlashNextTextExecutor& operator=(const FlashNextTextExecutor&) = delete;
    FlashNextTextExecutor(FlashNextTextExecutor&&)                 = delete;
    FlashNextTextExecutor& operator=(FlashNextTextExecutor&&)      = delete;

    [[nodiscard]] LaneHandle allocate_lane();
    void release_lane(LaneHandle handle);

    [[nodiscard]] std::int32_t committed_frontier(LaneHandle handle) const;
    [[nodiscard]] std::size_t active_lanes_count() const noexcept;

    [[nodiscard]] std::size_t available_physical_groups() const noexcept {
        return ledger_.available_physical_groups();
    }

    [[nodiscard]] bool has_pending_round() const noexcept {
        return ledger_.has_pending_transaction();
    }

    [[nodiscard]] std::uint32_t pending_batch_size() const noexcept {
        return ledger_.pending_batch_size();
    }

    // Executes one compact decode round for a batch of 1..max_concurrency requests on
    // device_.stream. Returns an owner-bound PendingRound transaction.
    [[nodiscard]] PendingRound execute_round(std::span<const LaneStepRequest> requests);

    [[nodiscard]] const FlashNextLaneLedger& ledger() const noexcept { return ledger_; }

    [[nodiscard]] FlashNextLaneLedger& ledger() noexcept { return ledger_; }

private:
    friend class PendingRound;

    const TextModelView& model_;
    PleIndexMetadata ple_metadata_;
    DeviceContext& device_;
    FlashNextRuntimeAllocation& alloc_;

    PleGatherPipeline ple_pipeline_;
    FlashNextLaneLedger ledger_;

    // Host staging buffers for round tensor upload
    std::vector<std::int32_t> host_token_ids_;
    std::vector<std::int32_t> host_token_indices_;
    std::vector<std::int32_t> host_mrope_positions_;
    std::vector<std::int32_t> host_table_rows_;
    std::vector<std::int32_t> host_source_slots_;
    std::vector<std::int32_t> host_destination_slots_;

    void commit_transaction(std::uint64_t tx_id, std::span<const LaneCommitDecision> decisions);
    void abort_transaction(std::uint64_t tx_id) noexcept;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
