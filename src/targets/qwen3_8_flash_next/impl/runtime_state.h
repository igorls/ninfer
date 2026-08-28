#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"
#include "targets/qwen3_8_flash_next/impl/text_decode_state.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct FlashNextRoundTensors {
    Tensor token_ids;              // I32 [max_concurrency]
    Tensor token_indices;          // I32 [max_concurrency]
    Tensor mrope_positions;        // I32 [max_concurrency, 3]
    Tensor table_rows;             // I32 [max_concurrency]
    Tensor source_slots;           // I32 [max_concurrency]
    Tensor destination_slots;      // I32 [max_concurrency]
    Tensor gathered_ple_embedding; // BF16 [2560, max_concurrency]
    Tensor final_hidden;           // BF16 [2560, max_concurrency]
    Tensor logits;                 // BF16 [248320, max_concurrency]
};

class FlashNextRuntimeAllocation {
public:
    explicit FlashNextRuntimeAllocation(FlashNextRuntimePlan plan);
    ~FlashNextRuntimeAllocation() = default;

    FlashNextRuntimeAllocation(const FlashNextRuntimeAllocation&)                = delete;
    FlashNextRuntimeAllocation& operator=(const FlashNextRuntimeAllocation&)     = delete;
    FlashNextRuntimeAllocation(FlashNextRuntimeAllocation&&) noexcept            = default;
    FlashNextRuntimeAllocation& operator=(FlashNextRuntimeAllocation&&) noexcept = default;

    [[nodiscard]] const FlashNextRuntimePlan& plan() const noexcept { return plan_; }

    [[nodiscard]] const FlashNextDecodeStateView& state_view() const noexcept {
        return state_view_;
    }

    [[nodiscard]] FlashNextDecodeStateView& state_view() noexcept { return state_view_; }

    [[nodiscard]] const FlashNextRoundTensors& round_tensors() const noexcept {
        return round_tensors_;
    }

    [[nodiscard]] FlashNextRoundTensors& round_tensors() noexcept { return round_tensors_; }

    [[nodiscard]] WorkspaceArena& workspace() noexcept { return *workspace_; }

    // Initialize device slot tensors and state before the first decode round
    void initialize(cudaStream_t stream);

    // Transactional state slot mechanics:
    // Swaps active (source) and standby (destination) slot for row b in [0, max_concurrency).
    void commit_row_slot(std::uint32_t row_index, cudaStream_t stream);
    void sync_slots_to_device(cudaStream_t stream);

    [[nodiscard]] std::int32_t current_source_slot(std::uint32_t row_index) const;
    [[nodiscard]] std::int32_t current_destination_slot(std::uint32_t row_index) const;

private:
    FlashNextRuntimePlan plan_;
    std::unique_ptr<DeviceBuffer> storage_;
    std::unique_ptr<WorkspaceArena> workspace_;

    FlashNextDecodeStateView state_view_{};
    FlashNextRoundTensors round_tensors_{};

    // Slot pair for each concurrency row: active (source) and standby (destination)
    std::vector<std::int32_t> host_active_slots_;
    std::vector<std::int32_t> host_standby_slots_;

    void materialize_views();
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
