#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"
#include "targets/qwen3_8_flash_next/impl/text_decode_state.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct FlashNextRoundTensors {
    Tensor token_ids;              // I32 [max_concurrency]
    Tensor token_indices;          // I32 [max_concurrency]
    Tensor mrope_positions;        // I32 [max_concurrency, 3]
    Tensor table_rows;             // I32 [max_concurrency]
    Tensor source_slots;           // I32 [max_concurrency]
    Tensor destination_slots;      // I32 [max_concurrency]
    Tensor sampled_tokens;         // I32 [max_concurrency]
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

    [[nodiscard]] FlashNextDecodeIngress* host_ingress() noexcept {
        return static_cast<FlashNextDecodeIngress*>(host_ingress_.data());
    }
    [[nodiscard]] const FlashNextDecodeIngress* host_ingress() const noexcept {
        return static_cast<const FlashNextDecodeIngress*>(host_ingress_.data());
    }

    [[nodiscard]] FlashNextDecodeEgress* host_egress() noexcept {
        return static_cast<FlashNextDecodeEgress*>(host_egress_.data());
    }
    [[nodiscard]] const FlashNextDecodeEgress* host_egress() const noexcept {
        return static_cast<const FlashNextDecodeEgress*>(host_egress_.data());
    }

    [[nodiscard]] void* device_ingress_ptr() noexcept { return device_ingress_; }
    [[nodiscard]] const void* device_ingress_ptr() const noexcept { return device_ingress_; }

    [[nodiscard]] void* device_egress_ptr() noexcept { return device_egress_; }
    [[nodiscard]] const void* device_egress_ptr() const noexcept { return device_egress_; }

    [[nodiscard]] const ops::SamplingConfig* device_sampling_configs() const noexcept {
        return reinterpret_cast<const ops::SamplingConfig*>(
            static_cast<const std::byte*>(device_ingress_) + offsetof(FlashNextDecodeIngress, sampling));
    }

    // Initialize device slot tensors and state before the first decode round
    void initialize(cudaStream_t stream);

    // Transactional state slot mechanics:
    // Swaps active (source) and standby (destination) slot for row b in [0, max_concurrency).
    void commit_row_slot(std::uint32_t row_index, cudaStream_t stream);
    void commit_slots(std::span<const std::uint32_t> accepted_lanes, cudaStream_t stream);
    void restore_lane_slots(std::uint32_t lane_index, std::int32_t active_slot,
                            std::int32_t standby_slot, cudaStream_t stream);
    void sync_slots_to_device(cudaStream_t stream);

    // Recurrent state zeroing for assigned lane slots
    void zero_slot(std::uint32_t slot_index, cudaStream_t stream);
    void zero_lane_slots(std::uint32_t lane_index, cudaStream_t stream);

    [[nodiscard]] std::int32_t current_source_slot(std::uint32_t row_index) const;
    [[nodiscard]] std::int32_t current_destination_slot(std::uint32_t row_index) const;

private:
    FlashNextRuntimePlan plan_;
    std::unique_ptr<DeviceBuffer> storage_;
    std::unique_ptr<WorkspaceArena> workspace_;

    PinnedHostBuffer host_ingress_;
    PinnedHostBuffer host_egress_;
    void* device_ingress_ = nullptr;
    void* device_egress_  = nullptr;

    FlashNextDecodeStateView state_view_{};
    FlashNextRoundTensors round_tensors_{};

    // Slot pair for each concurrency row: active (source) and standby (destination)
    std::vector<std::int32_t> host_active_slots_;
    std::vector<std::int32_t> host_standby_slots_;

    void materialize_views();
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
