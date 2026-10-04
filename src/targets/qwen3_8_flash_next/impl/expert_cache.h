#pragma once

#include "targets/qwen3_8_flash_next/impl/expert_bank.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// Text routed experts only. MTP's one NVFP4 bank stays on the existing loader path:
// it is about one layer (1.32 GiB) and is not part of the 48-layer 63.28 GiB bank.
inline constexpr std::uint32_t kExpertCacheLayers           = 48;
inline constexpr std::uint32_t kExpertCacheExpertsPerLayer  = 512;
inline constexpr std::uint32_t kExpertCacheTextExperts =
    kExpertCacheLayers * kExpertCacheExpertsPerLayer;
inline constexpr std::uint64_t kExpertCacheDeviceAlign = 256;
// Decode selects 10 experts. One prepare of a single layer therefore fits in this
// window with no host/device sync. Prefill waves through the same window.
inline constexpr std::uint32_t kExpertCachePinnedStagingSlots = 16;

// Released artifact device payload with Text, MTP, and Vision enabled and the two
// BF16 MTP expert banks left mapped. See qwen3.8-flash-next-artifact.md. This is
// payload, not the arena size after per-tensor alignment.
inline constexpr std::uint64_t kFlashNextReleasedDevicePayloadBytes = 76'251'938'528ULL;
// Colab G4 total reported for the empty-stand-in measurement (101,974,081,536 bytes).
inline constexpr std::uint64_t kColabG4TotalBytes = 101'974'081'536ULL;
// One 256K FP8 KV plane with MTP: 13 QSA layers, 1024 groups of 256 tokens.
// Per layer per group: 262,144 attention bytes + 16,384 indexer bytes = 278,528.
// 13 * 1024 * 278,528 = 3,707,764,736.
inline constexpr std::uint64_t kFlashNextFp8MtpKvPlaneBytes = 3'707'764'736ULL;
inline constexpr std::uint32_t kFlashNextPro6000KvPlanes    = 8;
// Bytes of text-expert VRAM the PRO 6000 plan returns to KV and launch headroom.
inline constexpr std::uint64_t kExpertCachePro6000ReleaseBytes = 10ULL << 30;

enum class ExpertCachePlane : int {
    GateCodes    = 0,
    GateScales   = 1,
    GateDivisors = 2,
    DownCodes    = 3,
    DownScales   = 4,
    DownDivisors = 5,
};
inline constexpr int kExpertCachePlanes = 6;

struct ExpertCacheGeometry {
    std::uint64_t gate_code_bytes     = 0;
    std::uint64_t gate_scale_bytes    = 0;
    std::uint64_t down_code_bytes     = 0;
    std::uint64_t down_scale_bytes    = 0;
    std::uint64_t gate_code_offset    = 0;
    std::uint64_t gate_scale_offset   = 0;
    std::uint64_t gate_divisor_offset = 0;
    std::uint64_t down_code_offset    = 0;
    std::uint64_t down_scale_offset   = 0;
    std::uint64_t down_divisor_offset = 0;
    std::uint64_t slot_bytes          = 0;
    std::uint64_t gate_bank_bytes     = 0;
    std::uint64_t down_bank_bytes     = 0;
    std::uint64_t layer_bank_bytes    = 0;
    std::uint64_t text_bank_bytes     = 0;
};

struct ExpertSlotPlanes {
    std::byte* gate_codes   = nullptr;
    std::byte* gate_scales  = nullptr;
    float* gate_divisor     = nullptr;
    std::byte* down_codes   = nullptr;
    std::byte* down_scales  = nullptr;
    float* down_divisor     = nullptr;
};

// `requested_bytes == 0` or a request at least as large as the packed text banks
// keeps today's full device residency (`enabled == false`). Any smaller request
// is floored to a whole number of aligned slots, so the device cache never
// exceeds the request and the NVFP4 words themselves are not repacked.
struct ExpertCachePlan {
    bool enabled                         = false;
    std::uint64_t requested_bytes        = 0;
    std::uint64_t resident_slots         = 0;
    std::uint64_t slot_bytes             = 0;
    std::uint64_t device_cache_bytes     = 0;
    std::uint64_t full_text_bank_bytes   = 0;
    std::uint64_t released_bytes         = 0;
    std::uint32_t pinned_staging_slots   = 0;
    std::uint64_t pinned_staging_bytes   = 0;
};

// Payload-level fit on a Colab G4. CUDA context, graph arenas, and activation
// workspaces are not included; a malloc on that card is still required.
// `gather_bank_bytes` is the one-layer device bank the existing MoE kernels
// read after the pointer-table gather. Full residency does not allocate it.
struct ExpertCachePayloadFit {
    std::uint64_t non_expert_device_bytes    = 0;
    std::uint64_t mtp_expert_device_bytes    = 0;
    std::uint64_t routed_expert_device_bytes = 0;
    std::uint64_t gather_bank_bytes          = 0;
    std::uint64_t kv_bytes                   = 0;
    std::uint64_t device_total_bytes         = 0;
    std::uint64_t used_bytes                 = 0;
    std::uint64_t headroom_bytes             = 0;
    bool kv_fits                             = false;
};

struct ExpertCacheTouch {
    bool hit                  = false;
    bool admitted             = false;
    std::uint32_t id          = 0;
    std::int32_t slot         = -1;
    bool evicted              = false;
    std::uint32_t evicted_id  = 0;
};

class ExpertCacheDevice;

struct ExpertLayerCache {
    std::uint32_t layer = 0;
    Nvfp4ExpertBankView gate_up{};
    Nvfp4ExpertBankView down{};
    // Device cache that owns this layer's slots, pointer table, and launch bank.
    // Null when the routed experts are an ordinary device-resident bank.
    ExpertCacheDevice* device = nullptr;
};

[[nodiscard]] constexpr std::uint64_t expert_cache_align_up(std::uint64_t value,
                                                            std::uint64_t alignment) noexcept {
    return (value + alignment - 1ULL) & ~(alignment - 1ULL);
}

// Packed NVFP4 expert-bank bytes. Matches flash_next_nvfp4_expert_bank_payload_bytes
// and artifact::block_scale_bank_geometry for these shapes: code plane, 256-byte
// alignment, scale plane, then one FP32 divisor per expert.
[[nodiscard]] constexpr std::uint64_t expert_cache_nvfp4_bank_bytes(std::uint64_t experts,
                                                                   std::uint64_t rows,
                                                                   std::uint64_t columns) noexcept {
    const std::uint64_t elements   = experts * rows * columns;
    const std::uint64_t code_bytes = elements / 2ULL;
    const std::uint64_t scale_off  = expert_cache_align_up(code_bytes, kExpertCacheDeviceAlign);
    const std::uint64_t scale_bytes = elements / 16ULL;
    return scale_off + scale_bytes + experts * sizeof(float);
}

[[nodiscard]] constexpr ExpertCacheGeometry flash_next_expert_cache_geometry() noexcept {
    constexpr std::uint64_t kGateRows    = 1'280;
    constexpr std::uint64_t kGateColumns = 2'560;
    constexpr std::uint64_t kDownRows    = 2'560;
    constexpr std::uint64_t kDownColumns = 640;
    ExpertCacheGeometry geo{};
    geo.gate_code_bytes  = kGateRows * kGateColumns / 2ULL;
    geo.gate_scale_bytes = kGateRows * kGateColumns / 16ULL;
    geo.down_code_bytes  = kDownRows * kDownColumns / 2ULL;
    geo.down_scale_bytes = kDownRows * kDownColumns / 16ULL;
    geo.gate_code_offset = 0;
    geo.gate_scale_offset =
        expert_cache_align_up(geo.gate_code_offset + geo.gate_code_bytes, kExpertCacheDeviceAlign);
    geo.gate_divisor_offset = expert_cache_align_up(geo.gate_scale_offset + geo.gate_scale_bytes,
                                                    kExpertCacheDeviceAlign);
    geo.down_code_offset    = expert_cache_align_up(geo.gate_divisor_offset + sizeof(float),
                                                    kExpertCacheDeviceAlign);
    geo.down_scale_offset =
        expert_cache_align_up(geo.down_code_offset + geo.down_code_bytes, kExpertCacheDeviceAlign);
    geo.down_divisor_offset = expert_cache_align_up(geo.down_scale_offset + geo.down_scale_bytes,
                                                    kExpertCacheDeviceAlign);
    geo.slot_bytes =
        expert_cache_align_up(geo.down_divisor_offset + sizeof(float), kExpertCacheDeviceAlign);
    geo.gate_bank_bytes =
        expert_cache_nvfp4_bank_bytes(kExpertCacheExpertsPerLayer, kGateRows, kGateColumns);
    geo.down_bank_bytes =
        expert_cache_nvfp4_bank_bytes(kExpertCacheExpertsPerLayer, kDownRows, kDownColumns);
    geo.layer_bank_bytes = geo.gate_bank_bytes + geo.down_bank_bytes;
    geo.text_bank_bytes  = geo.layer_bank_bytes * kExpertCacheLayers;
    return geo;
}

inline constexpr ExpertCacheGeometry kExpertCacheGeometry = flash_next_expert_cache_geometry();

static_assert(kExpertCacheGeometry.gate_bank_bytes == 943'720'448ULL);
static_assert(kExpertCacheGeometry.down_bank_bytes == 471'861'248ULL);
static_assert(kExpertCacheGeometry.layer_bank_bytes == 1'415'581'696ULL);
static_assert(kExpertCacheGeometry.text_bank_bytes == 67'947'921'408ULL);
static_assert(kExpertCacheGeometry.slot_bytes == 2'765'312ULL);
static_assert(kFlashNextReleasedDevicePayloadBytes > kExpertCacheGeometry.text_bank_bytes);

// One decode capture is split once per text layer: the segment ends after that
// layer's route and the next segment begins at its expert kernels. 48 splits
// produce 49 segments.
inline constexpr std::uint32_t kExpertCacheDecodeSegments = kExpertCacheLayers + 1;

// Where one expert's planes sit inside a contiguous NVFP4 bank of 512 experts.
// This is the layout `make_nvfp4_expert_bank_view` and the MoE kernels index
// (`base + expert * stride`), not the packed slot layout.
struct ExpertLaunchPlacement {
    std::uint64_t code_offset     = 0;
    std::uint64_t scale_offset    = 0;
    std::uint64_t divisor_offset  = 0;
    std::uint64_t code_bytes      = 0;
    std::uint64_t scale_bytes     = 0;
};

[[nodiscard]] constexpr ExpertLaunchPlacement
flash_next_expert_launch_placement(bool gate_up, std::uint32_t expert) noexcept {
    const std::uint64_t code_bytes =
        gate_up ? kExpertCacheGeometry.gate_code_bytes : kExpertCacheGeometry.down_code_bytes;
    const std::uint64_t scale_bytes =
        gate_up ? kExpertCacheGeometry.gate_scale_bytes : kExpertCacheGeometry.down_scale_bytes;
    const std::uint64_t scale_plane = expert_cache_align_up(
        static_cast<std::uint64_t>(kExpertCacheExpertsPerLayer) * code_bytes, kExpertCacheDeviceAlign);
    const std::uint64_t divisor_plane =
        scale_plane + static_cast<std::uint64_t>(kExpertCacheExpertsPerLayer) * scale_bytes;
    const std::uint64_t index = expert;
    return {
        .code_offset    = index * code_bytes,
        .scale_offset   = scale_plane + index * scale_bytes,
        .divisor_offset = divisor_plane + index * sizeof(float),
        .code_bytes     = code_bytes,
        .scale_bytes    = scale_bytes,
    };
}

static_assert(flash_next_expert_launch_placement(true, 0).code_offset == 0);
static_assert(flash_next_expert_launch_placement(true, 1).code_offset ==
              kExpertCacheGeometry.gate_code_bytes);
static_assert(flash_next_expert_launch_placement(true, 511).divisor_offset + sizeof(float) ==
              kExpertCacheGeometry.gate_bank_bytes);
static_assert(flash_next_expert_launch_placement(false, 511).divisor_offset + sizeof(float) ==
              kExpertCacheGeometry.down_bank_bytes);
static_assert(kExpertCacheDecodeSegments == 49);

// Stable across admission and eviction. Layer is the high part, local expert the low part.
[[nodiscard]] constexpr std::uint32_t flash_next_expert_id(std::uint32_t layer,
                                                          std::uint32_t expert) noexcept {
    return layer * kExpertCacheExpertsPerLayer + expert;
}

[[nodiscard]] constexpr std::uint32_t flash_next_expert_layer(std::uint32_t id) noexcept {
    return id / kExpertCacheExpertsPerLayer;
}

[[nodiscard]] constexpr std::uint32_t flash_next_expert_index(std::uint32_t id) noexcept {
    return id % kExpertCacheExpertsPerLayer;
}

static_assert(flash_next_expert_id(47, 511) == kExpertCacheTextExperts - 1U);
static_assert(flash_next_expert_layer(flash_next_expert_id(3, 7)) == 3U);
static_assert(flash_next_expert_index(flash_next_expert_id(3, 7)) == 7U);

// Resident-byte budget that frees kExpertCachePro6000ReleaseBytes of packed text experts.
// Slot alignment then floors the cache, so the realized release is slightly larger.
[[nodiscard]] constexpr std::uint64_t flash_next_pro6000_expert_cache_budget_bytes() noexcept {
    return kExpertCacheGeometry.text_bank_bytes - kExpertCachePro6000ReleaseBytes;
}

[[nodiscard]] inline ExpertCachePlan flash_next_plan_expert_cache(std::uint64_t requested_bytes) {
    ExpertCachePlan plan{};
    plan.requested_bytes      = requested_bytes;
    plan.full_text_bank_bytes = kExpertCacheGeometry.text_bank_bytes;
    plan.slot_bytes           = kExpertCacheGeometry.slot_bytes;
    if (requested_bytes == 0 || requested_bytes >= kExpertCacheGeometry.text_bank_bytes) {
        plan.enabled = false;
        return plan;
    }
    plan.enabled             = true;
    plan.resident_slots      = requested_bytes / kExpertCacheGeometry.slot_bytes;
    if (plan.resident_slots > kExpertCacheTextExperts) {
        plan.resident_slots = kExpertCacheTextExperts;
    }
    plan.device_cache_bytes = plan.resident_slots * kExpertCacheGeometry.slot_bytes;
    plan.released_bytes = kExpertCacheGeometry.text_bank_bytes - plan.device_cache_bytes;
    plan.pinned_staging_slots =
        plan.resident_slots == 0
            ? 0U
            : (plan.resident_slots < kExpertCachePinnedStagingSlots
                   ? static_cast<std::uint32_t>(plan.resident_slots)
                   : kExpertCachePinnedStagingSlots);
    plan.pinned_staging_bytes = plan.pinned_staging_slots * kExpertCacheGeometry.slot_bytes;
    return plan;
}

[[nodiscard]] inline ExpertCachePayloadFit
flash_next_expert_cache_payload_fit(const ExpertCachePlan& cache,
                                    std::uint32_t kv_planes = kFlashNextPro6000KvPlanes,
                                    std::uint64_t device_total_bytes = kColabG4TotalBytes) {
    ExpertCachePayloadFit fit{};
    fit.non_expert_device_bytes =
        kFlashNextReleasedDevicePayloadBytes - kExpertCacheGeometry.text_bank_bytes;
    fit.mtp_expert_device_bytes = kExpertCacheGeometry.layer_bank_bytes;
    fit.routed_expert_device_bytes =
        cache.enabled ? cache.device_cache_bytes : kExpertCacheGeometry.text_bank_bytes;
    fit.gather_bank_bytes =
        cache.enabled ? kExpertCacheGeometry.layer_bank_bytes : 0;
    fit.kv_bytes           = static_cast<std::uint64_t>(kv_planes) * kFlashNextFp8MtpKvPlaneBytes;
    fit.device_total_bytes = device_total_bytes;
    fit.used_bytes = fit.non_expert_device_bytes + fit.mtp_expert_device_bytes +
                     fit.routed_expert_device_bytes + fit.gather_bank_bytes + fit.kv_bytes;
    fit.kv_fits = fit.used_bytes <= device_total_bytes;
    fit.headroom_bytes = fit.kv_fits ? device_total_bytes - fit.used_bytes : 0;
    return fit;
}

// Unset keeps full residency. NINFER_FLASH_NEXT_EXPERT_CACHE=pro6000 selects the
// PRO 6000 budget. NINFER_FLASH_NEXT_EXPERT_CACHE_BUDGET_BYTES overrides it with
// an integer byte count, "full", or a "<n>GiB" resident budget.
[[nodiscard]] std::uint64_t flash_next_expert_cache_budget_from_environment();

[[nodiscard]] ExpertSlotPlanes expert_cache_slot_planes(void* slot,
                                                       const ExpertCacheGeometry& geometry);

// Copies one expert's NVFP4 code, scale, and divisor words into an aligned slot.
// Padding between planes is zero. The words themselves are not requantized.
void pack_expert_slot(const ExpertCacheGeometry& geometry, const Nvfp4ExpertBankView& gate_up,
                      const Nvfp4ExpertBankView& down, std::int32_t expert,
                      std::span<std::byte> slot);

// First-seen order, duplicates removed. An id outside 0..511 is rejected.
[[nodiscard]] std::vector<std::int32_t>
flash_next_unique_routed_experts(std::span<const std::int32_t> routed_ids);

// LRU directory. Expert ids are stable; only the slot assignment moves.
// A hit refreshes recency. A miss admits into a free slot or evicts the oldest
// resident. Capacity zero records a miss and admits nothing. Lookup is direct;
// there is no polling loop.
class ExpertCacheDirectory {
public:
    explicit ExpertCacheDirectory(std::uint32_t capacity);

    [[nodiscard]] ExpertCacheTouch touch(std::uint32_t id);
    [[nodiscard]] std::int32_t slot_of(std::uint32_t id) const;
    [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::uint32_t resident_count() const noexcept { return resident_; }

private:
    struct Slot {
        std::uint32_t id = 0;
        bool occupied    = false;
        std::int32_t older = -1;
        std::int32_t newer = -1;
    };

    void unlink(std::int32_t slot);
    void push_mru(std::int32_t slot);

    std::uint32_t capacity_  = 0;
    std::uint32_t resident_  = 0;
    std::int32_t lru_        = -1;
    std::int32_t mru_        = -1;
    std::vector<Slot> slots_;
    std::vector<std::int32_t> free_;
    std::vector<std::int32_t> id_to_slot_;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
