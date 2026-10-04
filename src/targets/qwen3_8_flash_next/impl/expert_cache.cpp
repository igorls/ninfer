#include "targets/qwen3_8_flash_next/impl/expert_cache.h"

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

[[nodiscard]] std::uint64_t parse_budget_spec(std::string_view spec) {
    if (spec == "full" || spec == "0") { return 0; }
    std::uint64_t value = 0;
    const char* const begin = spec.data();
    const char* const end   = spec.data() + spec.size();
    const auto parsed       = std::from_chars(begin, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr == begin) {
        throw std::invalid_argument(
            "NINFER_FLASH_NEXT_EXPERT_CACHE_BUDGET_BYTES must be an integer byte count, "
            "\"<n>GiB\", or \"full\"");
    }
    if (parsed.ptr == end) { return value; }
    const std::string_view suffix(parsed.ptr, static_cast<std::size_t>(end - parsed.ptr));
    if (suffix != "GiB") {
        throw std::invalid_argument(
            "NINFER_FLASH_NEXT_EXPERT_CACHE_BUDGET_BYTES suffix must be GiB");
    }
    if (value > (std::numeric_limits<std::uint64_t>::max() >> 30U)) {
        throw std::invalid_argument("NINFER_FLASH_NEXT_EXPERT_CACHE_BUDGET_BYTES overflows");
    }
    return value << 30U;
}

} // namespace

std::uint64_t flash_next_expert_cache_budget_from_environment() {
    if (const char* spec = std::getenv("NINFER_FLASH_NEXT_EXPERT_CACHE_BUDGET_BYTES");
        spec != nullptr && spec[0] != '\0') {
        return parse_budget_spec(spec);
    }
    if (const char* mode = std::getenv("NINFER_FLASH_NEXT_EXPERT_CACHE");
        mode != nullptr && mode[0] != '\0') {
        if (std::strcmp(mode, "0") == 0 || std::strcmp(mode, "off") == 0 ||
            std::strcmp(mode, "full") == 0) {
            return 0;
        }
        if (std::strcmp(mode, "1") == 0 || std::strcmp(mode, "on") == 0 ||
            std::strcmp(mode, "pro6000") == 0) {
            return flash_next_pro6000_expert_cache_budget_bytes();
        }
        throw std::invalid_argument(
            "NINFER_FLASH_NEXT_EXPERT_CACHE must be pro6000, on, off, or full");
    }
    return 0;
}

ExpertSlotPlanes expert_cache_slot_planes(void* slot, const ExpertCacheGeometry& geometry) {
    if (slot == nullptr) { return {}; }
    auto* base = static_cast<std::byte*>(slot);
    return {
        .gate_codes   = base + geometry.gate_code_offset,
        .gate_scales  = base + geometry.gate_scale_offset,
        .gate_divisor = reinterpret_cast<float*>(base + geometry.gate_divisor_offset),
        .down_codes   = base + geometry.down_code_offset,
        .down_scales  = base + geometry.down_scale_offset,
        .down_divisor = reinterpret_cast<float*>(base + geometry.down_divisor_offset),
    };
}

void pack_expert_slot(const ExpertCacheGeometry& geometry, const Nvfp4ExpertBankView& gate_up,
                      const Nvfp4ExpertBankView& down, std::int32_t expert,
                      std::span<std::byte> slot) {
    if (expert < 0 || static_cast<std::uint32_t>(expert) >= kExpertCacheExpertsPerLayer ||
        gate_up.codes == nullptr || gate_up.scales == nullptr ||
        gate_up.weight_scale_divisors == nullptr || down.codes == nullptr ||
        down.scales == nullptr || down.weight_scale_divisors == nullptr ||
        gate_up.code_bytes_per_expert != geometry.gate_code_bytes ||
        gate_up.scale_bytes_per_expert != geometry.gate_scale_bytes ||
        down.code_bytes_per_expert != geometry.down_code_bytes ||
        down.scale_bytes_per_expert != geometry.down_scale_bytes ||
        slot.size() != geometry.slot_bytes) {
        throw std::invalid_argument("Flash-Next expert cache slot pack received a bad view");
    }
    std::memset(slot.data(), 0, slot.size());
    const auto planes = expert_cache_slot_planes(slot.data(), geometry);
    const auto expert_index = static_cast<std::uint64_t>(expert);
    std::memcpy(planes.gate_codes,
                gate_up.codes + expert_index * gate_up.code_bytes_per_expert,
                geometry.gate_code_bytes);
    std::memcpy(planes.gate_scales,
                gate_up.scales + expert_index * gate_up.scale_bytes_per_expert,
                geometry.gate_scale_bytes);
    std::memcpy(planes.gate_divisor, gate_up.weight_scale_divisors + expert, sizeof(float));
    std::memcpy(planes.down_codes, down.codes + expert_index * down.code_bytes_per_expert,
                geometry.down_code_bytes);
    std::memcpy(planes.down_scales, down.scales + expert_index * down.scale_bytes_per_expert,
                geometry.down_scale_bytes);
    std::memcpy(planes.down_divisor, down.weight_scale_divisors + expert, sizeof(float));
}

ExpertCacheDirectory::ExpertCacheDirectory(std::uint32_t capacity)
    : capacity_(capacity),
      slots_(capacity),
      id_to_slot_(kExpertCacheTextExperts, -1) {
    if (capacity > kExpertCacheTextExperts) {
        throw std::invalid_argument("Flash-Next expert cache capacity exceeds the text experts");
    }
    free_.reserve(capacity);
    for (std::int32_t slot = static_cast<std::int32_t>(capacity) - 1; slot >= 0; --slot) {
        free_.push_back(slot);
    }
}

void ExpertCacheDirectory::unlink(std::int32_t slot) {
    Slot& node = slots_[static_cast<std::size_t>(slot)];
    if (node.older >= 0) {
        slots_[static_cast<std::size_t>(node.older)].newer = node.newer;
    } else {
        lru_ = node.newer;
    }
    if (node.newer >= 0) {
        slots_[static_cast<std::size_t>(node.newer)].older = node.older;
    } else {
        mru_ = node.older;
    }
    node.older = -1;
    node.newer = -1;
}

void ExpertCacheDirectory::push_mru(std::int32_t slot) {
    Slot& node = slots_[static_cast<std::size_t>(slot)];
    node.newer = -1;
    node.older = mru_;
    if (mru_ >= 0) {
        slots_[static_cast<std::size_t>(mru_)].newer = slot;
    } else {
        lru_ = slot;
    }
    mru_ = slot;
}

ExpertCacheTouch ExpertCacheDirectory::touch(std::uint32_t id) {
    if (id >= kExpertCacheTextExperts) {
        throw std::out_of_range("Flash-Next expert cache id is outside the text experts");
    }
    ExpertCacheTouch result{};
    result.id = id;
    const std::int32_t existing = id_to_slot_[id];
    if (existing >= 0) {
        result.hit      = true;
        result.admitted = false;
        result.slot     = existing;
        unlink(existing);
        push_mru(existing);
        return result;
    }
    result.hit = false;
    if (capacity_ == 0) {
        result.admitted = false;
        result.slot     = -1;
        return result;
    }
    std::int32_t slot = -1;
    if (!free_.empty()) {
        slot = free_.back();
        free_.pop_back();
        ++resident_;
    } else {
        slot = lru_;
        unlink(slot);
        result.evicted    = true;
        result.evicted_id = slots_[static_cast<std::size_t>(slot)].id;
        id_to_slot_[result.evicted_id] = -1;
    }
    Slot& node    = slots_[static_cast<std::size_t>(slot)];
    node.id       = id;
    node.occupied = true;
    id_to_slot_[id] = slot;
    push_mru(slot);
    result.admitted = true;
    result.slot     = slot;
    return result;
}

std::int32_t ExpertCacheDirectory::slot_of(std::uint32_t id) const {
    if (id >= kExpertCacheTextExperts) {
        throw std::out_of_range("Flash-Next expert cache id is outside the text experts");
    }
    return id_to_slot_[id];
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
