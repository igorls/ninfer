#include "targets/qwen3_8_flash_next/impl/expert_cache_device.h"

#include "core/device.h"

#include <array>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

[[nodiscard]] std::size_t plane_offset(std::uint32_t layer, ExpertCachePlane plane) {
    return (static_cast<std::size_t>(layer) * kExpertCachePlanes +
            static_cast<std::size_t>(plane)) *
           kExpertCacheExpertsPerLayer;
}

} // namespace

ExpertCacheDevice::ExpertCacheDevice(ExpertCachePlan plan)
    : plan_(plan),
      directory_(static_cast<std::uint32_t>(plan.resident_slots)),
      cache_(static_cast<std::size_t>(plan.device_cache_bytes)),
      tables_(static_cast<std::size_t>(kExpertCacheLayers) * kExpertCachePlanes *
              kExpertCacheExpertsPerLayer * sizeof(void*)) {
    if (!plan_.enabled) {
        throw std::invalid_argument("Flash-Next expert cache device requires an enabled plan");
    }
    if (plan_.pinned_staging_bytes > 0) {
        staging_.emplace(static_cast<std::size_t>(plan_.pinned_staging_bytes));
    }
    if (cache_.p != nullptr) { cache_.fill(0); }
    if (tables_.p != nullptr) {
        CUDA_CHECK(cudaMemset(tables_.p, 0, tables_.bytes));
    }
    pointer_mirror_.assign(static_cast<std::size_t>(kExpertCacheLayers) * kExpertCachePlanes *
                               kExpertCacheExpertsPerLayer,
                           nullptr);
    CUDA_CHECK(cudaEventCreateWithFlags(&inflight_, cudaEventDisableTiming));
}

ExpertCacheDevice::~ExpertCacheDevice() {
    if (inflight_ == nullptr) { return; }
    if (inflight_recorded_) { cudaEventSynchronize(inflight_); }
    cudaEventDestroy(inflight_);
    inflight_ = nullptr;
}

ExpertCacheDevice::ExpertCacheDevice(ExpertCacheDevice&& other) noexcept
    : plan_(std::move(other.plan_)),
      directory_(std::move(other.directory_)),
      cache_(std::move(other.cache_)),
      staging_(std::move(other.staging_)),
      tables_(std::move(other.tables_)),
      pointer_mirror_(std::move(other.pointer_mirror_)),
      inflight_(other.inflight_),
      inflight_recorded_(other.inflight_recorded_) {
    other.inflight_          = nullptr;
    other.inflight_recorded_ = false;
}

ExpertCacheDevice& ExpertCacheDevice::operator=(ExpertCacheDevice&& other) noexcept {
    if (this == &other) { return *this; }
    if (inflight_ != nullptr) {
        if (inflight_recorded_) { cudaEventSynchronize(inflight_); }
        cudaEventDestroy(inflight_);
    }
    plan_                = std::move(other.plan_);
    directory_           = std::move(other.directory_);
    cache_               = std::move(other.cache_);
    staging_             = std::move(other.staging_);
    tables_              = std::move(other.tables_);
    pointer_mirror_      = std::move(other.pointer_mirror_);
    inflight_            = other.inflight_;
    inflight_recorded_   = other.inflight_recorded_;
    other.inflight_          = nullptr;
    other.inflight_recorded_ = false;
    return *this;
}

void ExpertCacheDevice::wait_for_inflight() {
    if (!inflight_recorded_) { return; }
    CUDA_CHECK(cudaEventSynchronize(inflight_));
    inflight_recorded_ = false;
}

const void* ExpertCacheDevice::plane_table(std::uint32_t layer, ExpertCachePlane plane) const {
    if (layer >= kExpertCacheLayers || tables_.p == nullptr) {
        throw std::out_of_range("Flash-Next expert cache plane table is unavailable");
    }
    return static_cast<const std::byte*>(tables_.p) +
           plane_offset(layer, plane) * sizeof(void*);
}

void ExpertCacheDevice::publish_layer(std::uint32_t layer, cudaStream_t stream) {
    const ExpertCacheGeometry& geometry = kExpertCacheGeometry;
    void** host = pointer_mirror_.data() + plane_offset(layer, ExpertCachePlane::GateCodes);
    for (std::uint32_t expert = 0; expert < kExpertCacheExpertsPerLayer; ++expert) {
        const std::int32_t slot = directory_.slot_of(flash_next_expert_id(layer, expert));
        void* base              = nullptr;
        if (slot >= 0) {
            base = static_cast<std::byte*>(cache_.p) +
                   static_cast<std::size_t>(slot) * geometry.slot_bytes;
        }
        const ExpertSlotPlanes planes = expert_cache_slot_planes(base, geometry);
        host[static_cast<std::size_t>(ExpertCachePlane::GateCodes) * kExpertCacheExpertsPerLayer +
             expert] = planes.gate_codes;
        host[static_cast<std::size_t>(ExpertCachePlane::GateScales) * kExpertCacheExpertsPerLayer +
             expert] = planes.gate_scales;
        host[static_cast<std::size_t>(ExpertCachePlane::GateDivisors) *
                 kExpertCacheExpertsPerLayer +
             expert] = planes.gate_divisor;
        host[static_cast<std::size_t>(ExpertCachePlane::DownCodes) * kExpertCacheExpertsPerLayer +
             expert] = planes.down_codes;
        host[static_cast<std::size_t>(ExpertCachePlane::DownScales) * kExpertCacheExpertsPerLayer +
             expert] = planes.down_scales;
        host[static_cast<std::size_t>(ExpertCachePlane::DownDivisors) *
                 kExpertCacheExpertsPerLayer +
             expert] = planes.down_divisor;
    }
    auto* device = static_cast<std::byte*>(tables_.p) +
                   plane_offset(layer, ExpertCachePlane::GateCodes) * sizeof(void*);
    CUDA_CHECK(cudaMemcpyAsync(device, host,
                               kExpertCachePlanes * kExpertCacheExpertsPerLayer * sizeof(void*),
                               cudaMemcpyHostToDevice, stream));
}

void ExpertCacheDevice::prepare(std::uint32_t layer, std::span<const std::int32_t> local_experts,
                                const Nvfp4ExpertBankView& gate_up,
                                const Nvfp4ExpertBankView& down, cudaStream_t stream) {
    if (layer >= kExpertCacheLayers || stream == nullptr) {
        throw std::invalid_argument("Flash-Next expert cache prepare received a bad layer");
    }
    // The previous prepare's staging and pointer mirror must stay intact until its
    // copies reach this event. Later kernels queued on the stream are not included.
    wait_for_inflight();
    const ExpertCacheGeometry& geometry = kExpertCacheGeometry;
    auto* staging_bytes =
        staging_ ? static_cast<std::byte*>(staging_->data()) : nullptr;
    struct Pending {
        std::int32_t slot            = -1;
        std::uint32_t staging_index  = 0;
    };
    std::vector<Pending> wave;
    wave.reserve(plan_.pinned_staging_slots == 0 ? 1 : plan_.pinned_staging_slots);
    std::array<bool, kExpertCacheLayers> dirty{};

    auto flush = [&](bool sync_before_reuse) {
        for (const Pending& pending : wave) {
            void* destination = static_cast<std::byte*>(cache_.p) +
                                static_cast<std::size_t>(pending.slot) * geometry.slot_bytes;
            const void* source =
                staging_bytes + pending.staging_index * geometry.slot_bytes;
            CUDA_CHECK(cudaMemcpyAsync(destination, source,
                                       static_cast<std::size_t>(geometry.slot_bytes),
                                       cudaMemcpyHostToDevice, stream));
        }
        wave.clear();
        if (sync_before_reuse) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    };

    for (const std::int32_t expert : local_experts) {
        if (expert < 0 || static_cast<std::uint32_t>(expert) >= kExpertCacheExpertsPerLayer) {
            throw std::invalid_argument("Flash-Next expert cache prepare expert is out of range");
        }
        const ExpertCacheTouch touch = directory_.touch(flash_next_expert_id(layer, expert));
        if (touch.evicted) {
            dirty[flash_next_expert_layer(touch.evicted_id)] = true;
        }
        if (!touch.admitted) { continue; }
        if (staging_bytes == nullptr || cache_.p == nullptr) {
            throw std::logic_error("Flash-Next expert cache admitted into an empty device arena");
        }
        dirty[layer] = true;
        if (wave.size() == plan_.pinned_staging_slots) {
            flush(true);
        }
        const auto staging_index = static_cast<std::uint32_t>(wave.size());
        pack_expert_slot(geometry, gate_up, down, expert,
                         std::span<std::byte>(staging_bytes + staging_index * geometry.slot_bytes,
                                              static_cast<std::size_t>(geometry.slot_bytes)));
        wave.push_back(Pending{touch.slot, staging_index});
    }
    flush(false);
    for (std::uint32_t dirty_layer = 0; dirty_layer < kExpertCacheLayers; ++dirty_layer) {
        if (dirty[dirty_layer]) { publish_layer(dirty_layer, stream); }
    }
    CUDA_CHECK(cudaEventRecord(inflight_, stream));
    inflight_recorded_ = true;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
