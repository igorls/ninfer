#include "core/cpu/system.h"

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#include <cstring>

namespace ninfer {

CpuFeatures cpu_detail::usable_features(const FeatureRegisters& r) noexcept {
    CpuFeatures result;
    constexpr auto avx_state = (1U << 26) | (1U << 27) | (1U << 28);
    if ((r.leaf1_ecx & avx_state) != avx_state || (r.xcr0 & 0x6U) != 0x6U) return result;
    const auto add = [&](bool available, CpuFeature feature) {
        if (available) result.bits |= static_cast<std::uint32_t>(feature);
    };
    add(r.leaf1_ecx & (1U << 12), CpuFeature::Fma);
    add(r.leaf1_ecx & (1U << 29), CpuFeature::F16c);
    add(r.leaf7_0_ebx & (1U << 5), CpuFeature::Avx2);
    add(r.leaf7_1_eax & (1U << 4), CpuFeature::AvxVnni);
    if ((r.xcr0 & 0xe6U) != 0xe6U || !(r.leaf7_0_ebx & (1U << 16))) return result;
    add(true, CpuFeature::Avx512F);
    add(r.leaf7_0_ebx & (1U << 17), CpuFeature::Avx512Dq);
    add(r.leaf7_0_ebx & (1U << 30), CpuFeature::Avx512Bw);
    add(r.leaf7_0_ebx & (1U << 31), CpuFeature::Avx512Vl);
    add(r.leaf7_0_ecx & (1U << 11), CpuFeature::Avx512Vnni);
    add(r.leaf7_1_eax & (1U << 5), CpuFeature::Avx512Bf16);
    return result;
}

namespace {
struct CpuidWords { unsigned eax{}, ebx{}, ecx{}, edx{}; };

CpuidWords query_cpuid(unsigned leaf, unsigned subleaf = 0) noexcept {
    CpuidWords result;
#ifdef _MSC_VER
    int words[4];
    __cpuidex(words, static_cast<int>(leaf), static_cast<int>(subleaf));
    result = {static_cast<unsigned>(words[0]), static_cast<unsigned>(words[1]),
              static_cast<unsigned>(words[2]), static_cast<unsigned>(words[3])};
#else
    __cpuid_count(leaf, subleaf, result.eax, result.ebx, result.ecx, result.edx);
#endif
    return result;
}

std::uint64_t query_xcr0() noexcept {
#ifdef _MSC_VER
    return _xgetbv(0);
#else
    unsigned low, high;
    asm volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
    return (static_cast<std::uint64_t>(high) << 32) | low;
#endif
}
} // namespace

CpuFeatures current_cpu_features() noexcept {
    cpu_detail::FeatureRegisters state;
    const auto maximum = query_cpuid(0).eax;
    if (maximum < 1) return {};
    const auto basic = query_cpuid(1);
    state.leaf1_ecx = basic.ecx;
    if ((basic.ecx & ((1U << 26) | (1U << 27))) == ((1U << 26) | (1U << 27)))
        state.xcr0 = query_xcr0();
    if (maximum >= 7) {
        const auto extended = query_cpuid(7);
        state.leaf7_0_ebx = extended.ebx;
        state.leaf7_0_ecx = extended.ecx;
        if (extended.eax >= 1) state.leaf7_1_eax = query_cpuid(7, 1).eax;
    }
    return cpu_detail::usable_features(state);
}

CpuIdentity current_cpu_identity() noexcept {
    CpuIdentity result;
    const auto vendor = query_cpuid(0);
    std::memcpy(result.vendor.data(), &vendor.ebx, 4);
    std::memcpy(result.vendor.data() + 4, &vendor.edx, 4);
    std::memcpy(result.vendor.data() + 8, &vendor.ecx, 4);
    if (vendor.eax < 1) return result;
    const auto basic = query_cpuid(1);
    const auto family = (basic.eax >> 8) & 15U;
    result.family = family + (family == 15 ? (basic.eax >> 20) & 255U : 0);
    result.model = ((basic.eax >> 4) & 15U) |
                   ((family == 6 || family == 15) ? (basic.eax >> 12) & 240U : 0);
    result.stepping = basic.eax & 15U;
    result.virtualized = (basic.ecx & (1U << 31)) != 0;
    return result;
}

} // namespace ninfer
