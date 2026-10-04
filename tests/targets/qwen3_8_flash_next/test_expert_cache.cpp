#include "targets/qwen3_8_flash_next/impl/expert_cache.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::targets::qwen3_8_flash_next::detail::ExpertCacheDirectory;
using ninfer::targets::qwen3_8_flash_next::detail::ExpertCachePlane;
using ninfer::targets::qwen3_8_flash_next::detail::Nvfp4ExpertBankView;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_expert_cache_budget_from_environment;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_expert_cache_payload_fit;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_expert_id;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_expert_index;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_expert_layer;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_expert_launch_placement;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_plan_expert_cache;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_pro6000_expert_cache_budget_bytes;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_unique_routed_experts;
using ninfer::targets::qwen3_8_flash_next::detail::expert_cache_slot_planes;
using ninfer::targets::qwen3_8_flash_next::detail::kExpertCacheDecodeSegments;
using ninfer::targets::qwen3_8_flash_next::detail::kColabG4TotalBytes;
using ninfer::targets::qwen3_8_flash_next::detail::kExpertCacheGeometry;
using ninfer::targets::qwen3_8_flash_next::detail::kExpertCachePlanes;
using ninfer::targets::qwen3_8_flash_next::detail::kFlashNextFp8MtpKvPlaneBytes;
using ninfer::targets::qwen3_8_flash_next::detail::pack_expert_slot;

int failures = 0;

void set_env(const char* key, const char* value) {
#ifdef _WIN32
    _putenv_s(key, value == nullptr ? "" : value);
#else
    if (value == nullptr || value[0] == '\0') {
        unsetenv(key);
    } else {
        setenv(key, value, 1);
    }
#endif
}

void expect(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << "\n";
        ++failures;
    }
}

void test_budget() {
    const auto full = flash_next_plan_expert_cache(0);
    expect(!full.enabled, "zero budget keeps full residency");
    expect(full.released_bytes == 0, "full residency releases nothing");
    expect(full.device_cache_bytes == 0, "full residency allocates no slot cache");
    expect(!flash_next_plan_expert_cache(kExpertCacheGeometry.text_bank_bytes).enabled,
           "a budget covering the packed banks stays full residency");

    const auto pro =
        flash_next_plan_expert_cache(flash_next_pro6000_expert_cache_budget_bytes());
    expect(pro.enabled, "PRO 6000 budget enables the cache");
    expect(pro.resident_slots ==
               flash_next_pro6000_expert_cache_budget_bytes() / kExpertCacheGeometry.slot_bytes,
           "PRO 6000 slot count is the floored budget");
    expect(pro.device_cache_bytes == pro.resident_slots * pro.slot_bytes,
           "device cache is a whole number of slots");
    expect(pro.device_cache_bytes <= pro.requested_bytes, "cache does not exceed the request");
    expect(pro.released_bytes ==
               kExpertCacheGeometry.text_bank_bytes - pro.device_cache_bytes,
           "release is the packed banks minus the slot cache");
    expect(pro.released_bytes >= (10ULL << 30), "PRO 6000 release is at least 10 GiB");
    expect(pro.released_bytes < (12ULL << 30), "PRO 6000 release stays inside 8-12 GiB");
    expect(pro.device_cache_bytes > (18ULL << 30),
           "PRO 6000 resident cache is larger than the 12-18 GiB hypothesis");
    expect(pro.pinned_staging_slots == 16, "pinned window is 16 slots");
    expect(pro.pinned_staging_bytes == 16ULL * kExpertCacheGeometry.slot_bytes,
           "pinned bytes match the window");

    const auto tiny = flash_next_plan_expert_cache(1);
    expect(tiny.enabled && tiny.resident_slots == 0, "a one-byte budget clamps to zero slots");
    expect(tiny.pinned_staging_bytes == 0, "zero slots pin nothing");

    const auto one = flash_next_plan_expert_cache(kExpertCacheGeometry.slot_bytes +
                                                  kExpertCacheGeometry.slot_bytes - 1);
    expect(one.resident_slots == 1, "budget clamp floors to one slot");

    const auto small = flash_next_plan_expert_cache(16ULL << 30);
    expect(small.enabled, "a 16 GiB budget still enables a cache");
    expect(small.device_cache_bytes <= (16ULL << 30), "16 GiB budget is clamped");
    expect(small.released_bytes > (40ULL << 30),
           "a 12-18 GiB class cache frees far more than the PRO 6000 lever");

    const auto eight = flash_next_expert_cache_payload_fit(pro, 8, kColabG4TotalBytes);
    expect(eight.kv_fits, "PRO 6000 cache payload fits 8 KV planes on a G4");
    expect(eight.gather_bank_bytes == kExpertCacheGeometry.layer_bank_bytes,
           "enabled cache counts the one-layer launch bank");
    expect(eight.headroom_bytes > (3ULL << 30),
           "headroom after the launch bank stays above 3 GiB");
    expect(eight.headroom_bytes + eight.gather_bank_bytes >= (4ULL << 30),
           "headroom before the launch bank is at least 4 GiB");
    expect(eight.non_expert_device_bytes + eight.mtp_expert_device_bytes +
                   kExpertCacheGeometry.text_bank_bytes >
               eight.non_expert_device_bytes + eight.mtp_expert_device_bytes +
                   pro.device_cache_bytes,
           "the cache is smaller than full expert residency");

    const auto full_eight = flash_next_expert_cache_payload_fit(full, 8, kColabG4TotalBytes);
    expect(full_eight.gather_bank_bytes == 0, "full residency allocates no launch bank");
    expect(!full_eight.kv_fits, "full residency payload does not fit 8 KV planes");
    const auto full_six = flash_next_expert_cache_payload_fit(full, 6, kColabG4TotalBytes);
    expect(full_six.kv_fits, "full residency payload fits 6 KV planes");
    expect(full_six.headroom_bytes < kFlashNextFp8MtpKvPlaneBytes,
           "the 7th plane does not fit in the full-residency remainder");
}

void test_directory() {
    const std::uint32_t id_a = flash_next_expert_id(0, 1);
    const std::uint32_t id_b = flash_next_expert_id(0, 2);
    const std::uint32_t id_c = flash_next_expert_id(4, 9);
    expect(flash_next_expert_layer(id_c) == 4 && flash_next_expert_index(id_c) == 9,
           "expert id packing is stable");

    ExpertCacheDirectory empty(0);
    const auto blocked = empty.touch(id_a);
    expect(!blocked.hit && !blocked.admitted && blocked.slot < 0, "zero capacity does not admit");
    expect(blocked.id == id_a, "a rejected miss keeps the expert id");
    expect(empty.resident_count() == 0, "zero capacity stays empty");

    ExpertCacheDirectory cache(2);
    const auto first = cache.touch(id_a);
    const auto second = cache.touch(id_b);
    expect(!first.hit && first.admitted && first.slot == 0, "first miss takes slot 0");
    expect(!second.hit && second.admitted && second.slot == 1, "second miss takes slot 1");
    expect(first.id == id_a && second.id == id_b, "admission does not renumber expert ids");
    expect(cache.resident_count() == 2, "two admissions fill the cache");

    const auto again = cache.touch(id_a);
    expect(again.hit && !again.admitted && again.slot == 0 && again.id == id_a,
           "a second touch of the same id is a hit in the same slot");

    const auto third = cache.touch(id_c);
    expect(!third.hit && third.admitted && third.evicted && third.evicted_id == id_b,
           "the least recently used expert is evicted");
    expect(third.slot == 1 && third.id == id_c, "the new expert reuses the evicted slot");
    expect(cache.slot_of(id_a) == 0, "the refreshed expert stays resident");
    expect(cache.slot_of(id_b) < 0, "the evicted expert id is no longer resident");
    expect(cache.slot_of(id_c) == 1, "the admitted expert id resolves to its slot");
    expect(cache.resident_count() == 2, "eviction keeps the cache at capacity");

    const auto restored = cache.touch(id_b);
    expect(restored.admitted && restored.evicted && restored.evicted_id == id_a,
           "restoring the evicted id evicts the new least recently used id");
    expect(cache.slot_of(id_c) == 1 && cache.slot_of(id_b) == restored.slot,
           "survivor expert id stays bound to its slot");

    bool threw = false;
    try {
        (void)cache.touch(48U * 512U);
    } catch (const std::out_of_range&) { threw = true; }
    expect(threw, "an id past the text experts is rejected");
}

void test_pack_preserves_words() {
    const auto& geo = kExpertCacheGeometry;
    constexpr int kExperts = 2;
    std::vector<std::byte> gate_codes(static_cast<std::size_t>(kExperts) * geo.gate_code_bytes);
    std::vector<std::byte> gate_scales(static_cast<std::size_t>(kExperts) * geo.gate_scale_bytes);
    std::vector<float> gate_divisors(kExperts);
    std::vector<std::byte> down_codes(static_cast<std::size_t>(kExperts) * geo.down_code_bytes);
    std::vector<std::byte> down_scales(static_cast<std::size_t>(kExperts) * geo.down_scale_bytes);
    std::vector<float> down_divisors(kExperts);
    for (int expert = 0; expert < kExperts; ++expert) {
        const auto mark = static_cast<std::byte>(0x10 + expert);
        std::memset(gate_codes.data() + static_cast<std::size_t>(expert) * geo.gate_code_bytes,
                    std::to_integer<int>(mark), geo.gate_code_bytes);
        std::memset(gate_scales.data() + static_cast<std::size_t>(expert) * geo.gate_scale_bytes,
                    std::to_integer<int>(mark) + 1, geo.gate_scale_bytes);
        std::memset(down_codes.data() + static_cast<std::size_t>(expert) * geo.down_code_bytes,
                    std::to_integer<int>(mark) + 2, geo.down_code_bytes);
        std::memset(down_scales.data() + static_cast<std::size_t>(expert) * geo.down_scale_bytes,
                    std::to_integer<int>(mark) + 3, geo.down_scale_bytes);
        gate_divisors[static_cast<std::size_t>(expert)] = 1.5F + static_cast<float>(expert);
        down_divisors[static_cast<std::size_t>(expert)] = 4.5F + static_cast<float>(expert);
    }
    Nvfp4ExpertBankView gate{};
    gate.codes                  = gate_codes.data();
    gate.scales                 = gate_scales.data();
    gate.weight_scale_divisors  = gate_divisors.data();
    gate.experts                = 512;
    gate.code_bytes_per_expert  = geo.gate_code_bytes;
    gate.scale_bytes_per_expert = geo.gate_scale_bytes;
    Nvfp4ExpertBankView down{};
    down.codes                  = down_codes.data();
    down.scales                 = down_scales.data();
    down.weight_scale_divisors  = down_divisors.data();
    down.experts                = 512;
    down.code_bytes_per_expert  = geo.down_code_bytes;
    down.scale_bytes_per_expert = geo.down_scale_bytes;

    std::vector<std::byte> slot(static_cast<std::size_t>(geo.slot_bytes), std::byte{0xFF});
    pack_expert_slot(geo, gate, down, 1, slot);
    const auto planes_offset = geo.gate_code_offset;
    expect(slot[static_cast<std::size_t>(planes_offset)] == std::byte{0x11},
           "packed gate codes are expert 1's words");
    expect(slot[static_cast<std::size_t>(geo.gate_scale_offset)] == std::byte{0x12},
           "packed gate scales are expert 1's words");
    expect(slot[static_cast<std::size_t>(geo.down_code_offset)] == std::byte{0x13},
           "packed down codes are expert 1's words");
    expect(slot[static_cast<std::size_t>(geo.down_scale_offset)] == std::byte{0x14},
           "packed down scales are expert 1's words");
    float gate_divisor = 0;
    float down_divisor = 0;
    std::memcpy(&gate_divisor, slot.data() + geo.gate_divisor_offset, sizeof(float));
    std::memcpy(&down_divisor, slot.data() + geo.down_divisor_offset, sizeof(float));
    expect(gate_divisor == 2.5F && down_divisor == 5.5F, "packed divisors are expert 1's words");
    expect(slot[static_cast<std::size_t>(geo.gate_divisor_offset + sizeof(float))] == std::byte{0},
           "slot padding is cleared");
    expect(slot.back() == std::byte{0}, "the tail pad of the slot is cleared");
    expect(slot[0] != std::byte{0x10}, "expert 0's words are not what was packed");
    expect(static_cast<int>(ExpertCachePlane::DownDivisors) + 1 == kExpertCachePlanes,
           "pointer-table plane count covers both matrices");
}

void test_launch_placement_and_unique_ids() {
    expect(kExpertCacheDecodeSegments == 49, "a decode capture splits once per text layer");
    const auto& geo = kExpertCacheGeometry;
    const auto gate = flash_next_expert_launch_placement(true, 7);
    const auto down = flash_next_expert_launch_placement(false, 7);
    expect(gate.code_offset == 7 * geo.gate_code_bytes, "gate codes use the contiguous-bank stride");
    expect(gate.scale_offset == flash_next_expert_launch_placement(true, 0).scale_offset +
                                    7 * geo.gate_scale_bytes,
           "gate scales sit in the scale plane at the expert stride");
    expect(down.code_offset == 7 * geo.down_code_bytes, "down codes use the contiguous-bank stride");
    expect(down.divisor_offset == flash_next_expert_launch_placement(false, 0).divisor_offset +
                                      7 * sizeof(float),
           "down divisors are one float per expert");
    expect(gate.code_offset != geo.gate_code_offset,
           "expert 7's launch-bank codes are not at the slot origin");

    constexpr int kExperts = 8;
    std::vector<std::byte> gate_codes(static_cast<std::size_t>(kExperts) * geo.gate_code_bytes);
    std::vector<std::byte> gate_scales(static_cast<std::size_t>(kExperts) * geo.gate_scale_bytes);
    std::vector<float> gate_divisors(kExperts, 3.25F);
    std::vector<std::byte> down_codes(static_cast<std::size_t>(kExperts) * geo.down_code_bytes);
    std::vector<std::byte> down_scales(static_cast<std::size_t>(kExperts) * geo.down_scale_bytes);
    std::vector<float> down_divisors(kExperts, 8.5F);
    const auto mark = std::byte{0x5A};
    std::memset(gate_codes.data() + 7 * geo.gate_code_bytes, std::to_integer<int>(mark),
                geo.gate_code_bytes);
    std::memset(gate_scales.data() + 7 * geo.gate_scale_bytes, 0x5B, geo.gate_scale_bytes);
    std::memset(down_codes.data() + 7 * geo.down_code_bytes, 0x5C, geo.down_code_bytes);
    std::memset(down_scales.data() + 7 * geo.down_scale_bytes, 0x5D, geo.down_scale_bytes);
    gate_divisors[7] = 1.25F;
    down_divisors[7] = 9.25F;
    Nvfp4ExpertBankView gate_bank{};
    gate_bank.codes                 = gate_codes.data();
    gate_bank.scales                = gate_scales.data();
    gate_bank.weight_scale_divisors = gate_divisors.data();
    gate_bank.code_bytes_per_expert = geo.gate_code_bytes;
    gate_bank.scale_bytes_per_expert = geo.gate_scale_bytes;
    Nvfp4ExpertBankView down_bank{};
    down_bank.codes                 = down_codes.data();
    down_bank.scales                = down_scales.data();
    down_bank.weight_scale_divisors = down_divisors.data();
    down_bank.code_bytes_per_expert = geo.down_code_bytes;
    down_bank.scale_bytes_per_expert = geo.down_scale_bytes;
    std::vector<std::byte> slot(static_cast<std::size_t>(geo.slot_bytes));
    pack_expert_slot(geo, gate_bank, down_bank, 7, slot);
    const auto planes = expert_cache_slot_planes(slot.data(), geo);
    expect(std::memcmp(planes.gate_codes, gate_codes.data() + gate.code_offset, gate.code_bytes) == 0,
           "slot gate codes match the launch-bank stride for that expert");
    expect(std::memcmp(planes.gate_scales, gate_scales.data() + 7 * geo.gate_scale_bytes,
                       gate.scale_bytes) == 0,
           "slot gate scales match the launch-bank scale words");
    expect(std::memcmp(planes.down_codes, down_codes.data() + down.code_offset, down.code_bytes) == 0,
           "slot down codes match the launch-bank stride for that expert");
    float packed_divisor = 0;
    std::memcpy(&packed_divisor, planes.down_divisor, sizeof(float));
    expect(packed_divisor == 9.25F, "slot down divisor matches the launch-bank word");

    const std::vector<std::int32_t> routed = {4, 1, 4, 9, 1, 0};
    const auto unique                      = flash_next_unique_routed_experts(routed);
    expect(unique.size() == 4 && unique[0] == 4 && unique[1] == 1 && unique[2] == 9 &&
               unique[3] == 0,
           "duplicate routed ids keep first-seen order");
    const std::vector<std::int32_t> rejected = {1, 512};
    bool threw = false;
    try {
        (void)flash_next_unique_routed_experts(rejected);
    } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "an expert id of 512 is rejected");
}

void test_environment() {
    const char* saved_budget = std::getenv("NINFER_FLASH_NEXT_EXPERT_CACHE_BUDGET_BYTES");
    const char* saved_mode   = std::getenv("NINFER_FLASH_NEXT_EXPERT_CACHE");
    const std::string budget = saved_budget == nullptr ? std::string() : saved_budget;
    const std::string mode   = saved_mode == nullptr ? std::string() : saved_mode;
    set_env("NINFER_FLASH_NEXT_EXPERT_CACHE_BUDGET_BYTES", "");
    set_env("NINFER_FLASH_NEXT_EXPERT_CACHE", "");
    expect(flash_next_expert_cache_budget_from_environment() == 0, "unset env is full residency");
    set_env("NINFER_FLASH_NEXT_EXPERT_CACHE", "pro6000");
    expect(flash_next_expert_cache_budget_from_environment() ==
               flash_next_pro6000_expert_cache_budget_bytes(),
           "pro6000 mode selects the PRO 6000 budget");
    set_env("NINFER_FLASH_NEXT_EXPERT_CACHE_BUDGET_BYTES", "12GiB");
    expect(flash_next_expert_cache_budget_from_environment() == (12ULL << 30),
           "an explicit GiB budget overrides the mode");
    set_env("NINFER_FLASH_NEXT_EXPERT_CACHE_BUDGET_BYTES", "full");
    expect(flash_next_expert_cache_budget_from_environment() == 0, "full restores residency");
    bool threw = false;
    set_env("NINFER_FLASH_NEXT_EXPERT_CACHE_BUDGET_BYTES", "nope");
    try {
        (void)flash_next_expert_cache_budget_from_environment();
    } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "a bad budget string is rejected");
    set_env("NINFER_FLASH_NEXT_EXPERT_CACHE_BUDGET_BYTES", budget.c_str());
    set_env("NINFER_FLASH_NEXT_EXPERT_CACHE", mode.c_str());
}

} // namespace

int main() {
    test_budget();
    test_directory();
    test_pack_preserves_words();
    test_launch_placement_and_unique_ids();
    test_environment();
    if (failures != 0) {
        std::cerr << failures << " expert-cache checks failed\n";
        return 1;
    }
    std::cout << "PASS: flash-next expert cache policy\n";
    return 0;
}
