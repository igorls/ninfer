#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include "runtime/engine/kv_capacity.h"

#include <iostream>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int test_sequence_plan_curve_consistency() {
    using Package = ninfer::targets::qwen3_8_flash_next::Package;

    int failures = 0;

    ninfer::EngineOptions options{};
    options.max_concurrency = 4;
    options.max_context     = 8192;

    ninfer::DeviceContext dummy_device{};
    auto planner = Package::make_sequence_planner(
        dummy_device, options, Package::WeightsProfile::MixedNvfp4Fp8PleInt4);

    const auto curve = planner.capacity_curve();
    failures += check(curve.main_page_tokens == 256, "main_page_tokens must be 256");
    failures += check(curve.minimum_main_page_groups == (8192 + 255) / 256,
                      "minimum_main_page_groups mismatch");
    failures += check(curve.bytes_per_additional_main_page_group == 6488064,
                      "bytes_per_additional_main_page_group mismatch");

    // Test multiple main_page_groups values
    for (std::uint32_t groups : {curve.minimum_main_page_groups,
                                 curve.minimum_main_page_groups + 1,
                                 curve.minimum_main_page_groups + 10,
                                 curve.minimum_main_page_groups + 50}) {
        auto p = Package::make_sequence_planner(
            dummy_device, options, Package::WeightsProfile::MixedNvfp4Fp8PleInt4);
        auto plan = std::move(p).finalize(groups);

        const std::size_t expected_reservation =
            curve.minimum_device_reservation_bytes +
            static_cast<std::size_t>(groups - curve.minimum_main_page_groups) *
                curve.bytes_per_additional_main_page_group;
        const std::uint32_t expected_tokens = groups * 256;

        failures += check(plan.device_reservation_bytes() == expected_reservation,
                          "plan.device_reservation_bytes() does not match expected curve reservation");
        failures += check(plan.kv_capacity() == expected_tokens,
                          "plan.kv_capacity() does not match expected resolved tokens");
        failures += check(plan.capacity() == expected_tokens,
                          "plan.capacity() does not match expected resolved tokens");
        failures += check(plan.max_concurrency() == 4,
                          "plan.max_concurrency() mismatch");
    }

    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_sequence_plan_curve_consistency();

    if (failures == 0) {
        std::cout << "All SequencePlan tests passed cleanly.\n";
        return 0;
    }
    std::cerr << failures << " SequencePlan test(s) failed.\n";
    return 1;
}
