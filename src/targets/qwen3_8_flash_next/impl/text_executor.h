#pragma once

#include "core/arena.h"
#include "core/decode_graph.h"
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
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct FlashNextDecodeStateSink;
class FlashNextTextExecutor;

struct DecodeGraphProfile {
    std::uint32_t batch_size             = 1;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class         = 0;
    DecodeGraphDefinition definition;
};

struct DecodeGraphTopology {
    std::uint32_t topology_class = 0;
    DecodeGraphExecutable executable;
    std::optional<std::size_t> installed_profile;
};

struct DecodeGraphFamily {
    std::vector<DecodeGraphProfile> profiles;
    std::vector<DecodeGraphTopology> topologies;
};

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
    [[nodiscard]] std::span<const std::int32_t> sampled_tokens() const;

    void commit(std::span<const LaneCommitDecision> decisions);
    void abort() noexcept;

private:
    friend class FlashNextTextExecutor;
    PendingRound(FlashNextTextExecutor* owner, std::uint64_t transaction_id,
                 std::uint32_t batch_size, Tensor logits, Tensor final_hidden,
                 std::span<const std::int32_t> sampled_tokens = {}) noexcept;

    FlashNextTextExecutor* owner_ = nullptr;
    std::uint64_t transaction_id_ = 0;
    std::uint32_t batch_size_     = 0;
    Tensor logits_{};
    Tensor final_hidden_{};
    std::array<std::int32_t, 8> sampled_tokens_{};
};

class FlashNextTextExecutor {
public:
    FlashNextTextExecutor(const TextModelView& model, PleIndexMetadata ple_metadata,
                          DeviceContext& device, FlashNextRuntimeAllocation& allocation);

    ~FlashNextTextExecutor() = default;

    FlashNextTextExecutor(const FlashNextTextExecutor&)            = delete;
    FlashNextTextExecutor& operator=(const FlashNextTextExecutor&) = delete;
    FlashNextTextExecutor(FlashNextTextExecutor&&)                 = delete;
    FlashNextTextExecutor& operator=(FlashNextTextExecutor&&)      = delete;

    [[nodiscard]] LaneHandle allocate_lane();
    void release_lane(LaneHandle handle);

    [[nodiscard]] std::int32_t committed_frontier(LaneHandle handle) const;
    [[nodiscard]] const PleTokenHistory& lane_history(LaneHandle handle) const {
        return ledger_.lane_history(handle);
    }
    [[nodiscard]] std::size_t active_lanes_count() const noexcept;

    [[nodiscard]] std::size_t available_physical_groups() const noexcept {
        return ledger_.available_physical_groups();
    }

    [[nodiscard]] std::span<const std::uint32_t> lane_physical_groups(LaneHandle handle) const {
        return ledger_.lane_physical_groups(handle);
    }

    std::vector<std::uint32_t> take_lane_physical_groups(LaneHandle handle) {
        return ledger_.take_lane_physical_groups(handle);
    }

    void release_physical_groups(std::span<const std::uint32_t> groups) {
        ledger_.release_physical_groups(groups);
    }

    void attach_physical_groups(LaneHandle handle, std::span<const std::uint32_t> groups,
                                std::int32_t committed_frontier, const PleTokenHistory& history) {
        ledger_.attach_physical_groups(handle, groups, committed_frontier, history);
    }

    void copy_state_slot(std::uint32_t src_slot, std::uint32_t dst_slot) {
        alloc_.copy_state_slot(src_slot, dst_slot, device_.stream);
    }

    [[nodiscard]] const FlashNextRuntimeAllocation& allocation() const noexcept { return alloc_; }
    [[nodiscard]] FlashNextRuntimeAllocation& allocation() noexcept { return alloc_; }

    [[nodiscard]] bool has_pending_round() const noexcept {
        return ledger_.has_pending_transaction();
    }

    [[nodiscard]] std::uint32_t pending_batch_size() const noexcept {
        return ledger_.pending_batch_size();
    }

    // Executes one compact decode round for a batch of 1..max_concurrency requests on
    // device_.stream. Returns an owner-bound PendingRound transaction.
    [[nodiscard]] PendingRound execute_round(std::span<const LaneStepRequest> requests,
                                             const FlashNextDecodeStateSink* sink = nullptr);

    [[nodiscard]] PendingRound
    execute_prefill_chunk(LaneHandle handle, std::span<const std::int32_t> token_ids,
                          std::span<const std::array<std::int32_t, 3>> positions,
                          std::int32_t first_token_index,
                          const FlashNextDecodeStateSink* sink                     = nullptr,
                          const Tensor* visual_embeddings                          = nullptr,
                          std::span<const std::int32_t> chunk_local_scatter_indices = {});

    [[nodiscard]] const FlashNextLaneLedger& ledger() const noexcept { return ledger_; }

    [[nodiscard]] FlashNextLaneLedger& ledger() noexcept { return ledger_; }

    [[nodiscard]] bool use_cuda_graph() const noexcept { return use_cuda_graph_; }
    void set_use_cuda_graph(bool enable) noexcept { use_cuda_graph_ = enable; }
    [[nodiscard]] const DecodeGraphFamily& decode_graphs() const noexcept { return decode_graphs_; }

    void instantiate_graphs();
    void execute_round_body(std::uint32_t batch_size, const FlashNextDecodeStateSink* sink);

private:
    friend class PendingRound;

    const TextModelView& model_;
    PleIndexMetadata ple_metadata_;
    DeviceContext& device_;
    FlashNextRuntimeAllocation& alloc_;

    PleGatherPipeline ple_pipeline_;
    FlashNextLaneLedger ledger_;

    bool use_cuda_graph_ = true;
    DecodeGraphFamily decode_graphs_;
    WorkspaceArena sampling_workspace_;
    CudaCompletionEvent round_completion_;
    bool round_in_flight_ = false;
    // Eager-only: custom embedding columns for the round being built (never captured).
    std::vector<const Tensor*> pending_custom_embeddings_;

    bool pending_is_prefill_chunk_                     = false;
    std::uint32_t pending_prefill_lane_                = 0;
    std::int32_t pending_prefill_initial_active_slot_  = 0;
    std::int32_t pending_prefill_initial_standby_slot_ = 0;

    void commit_transaction(std::uint64_t tx_id, std::span<const LaneCommitDecision> decisions);
    void abort_transaction(std::uint64_t tx_id) noexcept;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
