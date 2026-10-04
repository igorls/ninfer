#pragma once

#include "core/arena.h"
#include "targets/qwen3_8_flash_next/impl/expert_cache.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// Device half of the text-expert cache.
//
// The packed text banks stay in the artifact file mapping, the same placement PLE
// uses. This object owns the GPU slot arena, a bounded pinned staging window, and
// one pointer table per layer. CUDA graph replay must load expert addresses from
// that table. The table allocation is fixed for the life of the cache; prepare()
// overwrites its contents before replay, so a changed resident set does not require
// recapture and does not bake a payload address into the graph.
//
// prepare() is the miss path. It admits with the LRU directory, copies the exact
// NVFP4 words through the pinned window, and publishes the affected layer tables
// on the caller's stream. It does not spin: a decode-sized miss list fits in the
// staging window and stays asynchronous, and a longer prefill list waits on that
// stream once per extra wave before reusing a staging slot.
class ExpertCacheDevice {
public:
    explicit ExpertCacheDevice(ExpertCachePlan plan);
    ~ExpertCacheDevice();

    ExpertCacheDevice(const ExpertCacheDevice&)            = delete;
    ExpertCacheDevice& operator=(const ExpertCacheDevice&) = delete;
    ExpertCacheDevice(ExpertCacheDevice&&) noexcept;
    ExpertCacheDevice& operator=(ExpertCacheDevice&&) noexcept;

    [[nodiscard]] const ExpertCachePlan& plan() const noexcept { return plan_; }
    [[nodiscard]] const ExpertCacheDirectory& directory() const noexcept { return directory_; }
    [[nodiscard]] std::uint64_t pinned_staging_bytes() const noexcept {
        return plan_.pinned_staging_bytes;
    }

    // Stable device address of the 512-entry plane for one layer. Kernels capture
    // this address and dereference it at replay.
    [[nodiscard]] const void* plane_table(std::uint32_t layer, ExpertCachePlane plane) const;

    void prepare(std::uint32_t layer, std::span<const std::int32_t> local_experts,
                 const Nvfp4ExpertBankView& gate_up, const Nvfp4ExpertBankView& down,
                 cudaStream_t stream);

private:
    void publish_layer(std::uint32_t layer, cudaStream_t stream);

    void wait_for_inflight();

    ExpertCachePlan plan_{};
    ExpertCacheDirectory directory_;
    DeviceBuffer cache_;
    std::optional<PinnedHostBuffer> staging_;
    DeviceBuffer tables_;
    // Pageable mirror of the device pointer tables. Async publishes read it, so
    // the next prepare waits for the previous publish before writing it again.
    std::vector<void*> pointer_mirror_;
    cudaEvent_t inflight_ = nullptr;
    bool inflight_recorded_ = false;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
