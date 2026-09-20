#include "ninfer/ops/candidate_logprobs.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr ReductionCriterion kFp32Criterion{
    /*relative_l2=*/2.0e-5,
    /*gross_absolute=*/2.0e-4,
    /*gross_relative_to_max_reference=*/0.0,
};

struct Case {
    std::string label;
    std::int32_t physical_rows = 0;
    std::int32_t valid_rows    = 0;
    std::int32_t columns       = 0;
    std::vector<std::uint16_t> logits;
    std::vector<std::int32_t> sampled;
    std::vector<std::int32_t> candidates;
    std::vector<std::int32_t> allowed; // empty: unconstrained
};

std::vector<std::uint16_t> make_logits(std::int32_t physical_rows, std::int32_t valid_rows,
                                       std::int32_t columns, std::uint32_t salt) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u +
                                        static_cast<std::uint32_t>(column + 1) * 1013904223u + salt;
            const float value = -24.0f + static_cast<float>(mixed % 6144u) * (1.0f / 128.0f);
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(value);
        }
        // Physical padding must not participate: make it the largest value present.
        for (std::int32_t row = valid_rows; row < physical_rows; ++row) {
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(96.0f);
        }
    }
    return logits;
}

bool allowed_row(const std::vector<std::int32_t>& allowed, std::int32_t row) {
    if (allowed.empty()) { return true; }
    return ((static_cast<std::uint32_t>(allowed[static_cast<std::size_t>(row) / 32U]) >>
             (static_cast<std::uint32_t>(row) % 32U)) &
            1U) != 0U;
}

// Independent FP64 evaluation of both normalisations for one id.
void oracle(const Case& c, std::int32_t column, std::int32_t id, double& raw, double& masked) {
    const std::size_t base = static_cast<std::size_t>(column) * c.physical_rows;
    double raw_max = -std::numeric_limits<double>::infinity();
    double masked_max = raw_max;
    for (std::int32_t row = 0; row < c.valid_rows; ++row) {
        const double value = bf16_to_f32(c.logits[base + row]);
        raw_max            = std::max(raw_max, value);
        if (allowed_row(c.allowed, row)) { masked_max = std::max(masked_max, value); }
    }
    double raw_sum = 0.0, masked_sum = 0.0;
    for (std::int32_t row = 0; row < c.valid_rows; ++row) {
        const double value = bf16_to_f32(c.logits[base + row]);
        raw_sum += std::exp(value - raw_max);
        if (allowed_row(c.allowed, row)) { masked_sum += std::exp(value - masked_max); }
    }
    const double value = bf16_to_f32(c.logits[base + id]);
    raw                = value - raw_max - std::log(raw_sum);
    masked             = allowed_row(c.allowed, id) ? value - masked_max - std::log(masked_sum)
                                                    : -std::numeric_limits<double>::infinity();
}

std::vector<double> fp32_as_double(const void* device, std::size_t count) {
    const auto values = from_device<float>(device, count);
    return {values.begin(), values.end()};
}

// -inf entries are compared exactly and removed before the reduction criterion.
int verify_with_infinities(const std::string& label, std::vector<double> actual,
                           std::vector<double> expected) {
    int failures = 0;
    std::vector<double> finite_actual, finite_expected;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::isinf(expected[i])) {
            if (!(std::isinf(actual[i]) && actual[i] < 0.0)) {
                std::cerr << label << ": element " << i << " expected -inf, got " << actual[i]
                          << '\n';
                ++failures;
            }
            continue;
        }
        finite_actual.push_back(actual[i]);
        finite_expected.push_back(expected[i]);
    }
    if (!finite_expected.empty()) {
        failures += verify_reduction(label, finite_actual, finite_expected, kFp32Criterion);
    }
    return failures;
}

int run_case(const Case& c) {
    const std::size_t columns    = static_cast<std::size_t>(c.columns);
    const std::size_t candidates = c.candidates.size();
    std::vector<double> expected_sampled(2 * columns);
    std::vector<double> expected_candidates(2 * candidates * columns);
    for (std::int32_t column = 0; column < c.columns; ++column) {
        oracle(c, column, c.sampled[static_cast<std::size_t>(column)],
               expected_sampled[static_cast<std::size_t>(column)],
               expected_sampled[columns + static_cast<std::size_t>(column)]);
        for (std::size_t n = 0; n < candidates; ++n) {
            oracle(c, column, c.candidates[n],
                   expected_candidates[n * columns + static_cast<std::size_t>(column)],
                   expected_candidates[candidates * columns + n * columns +
                                       static_cast<std::size_t>(column)]);
        }
    }

    GuardedDeviceBuffer device_logits(c.logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_sampled(columns * sizeof(std::int32_t));
    GuardedDeviceBuffer device_candidates(std::max<std::size_t>(candidates, 1) *
                                          sizeof(std::int32_t));
    GuardedDeviceBuffer device_allowed(std::max<std::size_t>(c.allowed.size(), 1) *
                                       sizeof(std::int32_t));
    GuardedDeviceBuffer device_sampled_out(2 * columns * sizeof(float));
    GuardedDeviceBuffer device_candidates_out(std::max<std::size_t>(2 * candidates * columns, 1) *
                                              sizeof(float));
    device_logits.copy_from_host(c.logits.data(), device_logits.bytes());
    device_sampled.copy_from_host(c.sampled.data(), columns * sizeof(std::int32_t));
    if (candidates != 0) {
        device_candidates.copy_from_host(c.candidates.data(), candidates * sizeof(std::int32_t));
    }
    if (!c.allowed.empty()) {
        device_allowed.copy_from_host(c.allowed.data(), c.allowed.size() * sizeof(std::int32_t));
    }
    device_sampled_out.fill(0xcd);
    device_candidates_out.fill(0xcd);

    Tensor logits(device_logits.data(), DType::BF16, {c.physical_rows, c.columns});
    Tensor sampled(device_sampled.data(), DType::I32, {c.columns});
    Tensor sampled_out(device_sampled_out.data(), DType::FP32, {c.columns, 2});
    std::optional<Tensor> candidate_ids, candidates_out, allowed;
    if (candidates != 0) {
        candidate_ids.emplace(device_candidates.data(), DType::I32,
                              std::initializer_list<std::int32_t>{
                                  static_cast<std::int32_t>(candidates)});
        candidates_out.emplace(device_candidates_out.data(), DType::FP32,
                               std::initializer_list<std::int32_t>{
                                   c.columns, static_cast<std::int32_t>(candidates), 2});
    }
    if (!c.allowed.empty()) {
        allowed.emplace(device_allowed.data(), DType::I32,
                        std::initializer_list<std::int32_t>{
                            static_cast<std::int32_t>(c.allowed.size())});
    }
    ops::candidate_logprobs(logits, c.valid_rows, sampled,
                            candidate_ids ? &*candidate_ids : nullptr,
                            allowed ? &*allowed : nullptr, sampled_out,
                            candidates_out ? &*candidates_out : nullptr, nullptr);
    cuda_synchronize();

    int failures = verify_with_infinities(
        c.label + " sampled", fp32_as_double(device_sampled_out.data(), 2 * columns),
        expected_sampled);
    if (candidates != 0) {
        failures += verify_with_infinities(
            c.label + " candidates",
            fp32_as_double(device_candidates_out.data(), 2 * candidates * columns),
            expected_candidates);
    }
    failures += verify_exact((c.label + " preserves logits").c_str(),
                             from_device<std::uint16_t>(device_logits.data(), c.logits.size()),
                             c.logits);
    failures += device_logits.verify_guards(c.label + " logits guards");
    failures += device_sampled_out.verify_guards(c.label + " sampled guards");
    failures += device_candidates_out.verify_guards(c.label + " candidates guards");
    return failures;
}

Case make_case(const std::string& label, std::int32_t physical_rows, std::int32_t valid_rows,
               std::int32_t columns, std::int32_t candidates, bool masked, std::uint32_t salt) {
    Case c;
    c.label         = label;
    c.physical_rows = physical_rows;
    c.valid_rows    = valid_rows;
    c.columns       = columns;
    c.logits        = make_logits(physical_rows, valid_rows, columns, salt);
    for (std::int32_t column = 0; column < columns; ++column) {
        c.sampled.push_back(static_cast<std::int32_t>(
            (static_cast<std::uint64_t>(column + 1) * 7919u + salt) %
            static_cast<std::uint32_t>(valid_rows)));
    }
    for (std::int32_t n = 0; n < candidates; ++n) {
        c.candidates.push_back(static_cast<std::int32_t>(
            (static_cast<std::uint64_t>(n + 3) * 104729u + salt) %
            static_cast<std::uint32_t>(valid_rows)));
    }
    if (masked) {
        // Allow every third row, plus half the candidates and every sampled token, so the
        // masked normaliser, the -inf branch, and the allowed-candidate branch all occur.
        c.allowed.assign(static_cast<std::size_t>((valid_rows + 31) / 32), 0);
        const auto allow = [&](std::int32_t row) {
            c.allowed[static_cast<std::size_t>(row) / 32U] |=
                static_cast<std::int32_t>(1U << (static_cast<std::uint32_t>(row) % 32U));
        };
        for (std::int32_t row = 0; row < valid_rows; row += 3) { allow(row); }
        for (const std::int32_t id : c.sampled) { allow(id); }
        for (std::size_t n = 0; n < c.candidates.size(); n += 2) { allow(c.candidates[n]); }
    }
    return c;
}

template <class Function>
int expect_invalid(const char* label, Function&& function) {
    try {
        function();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& error) {
        std::cerr << label << ": expected invalid_argument, got " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

int run_validation_cases() {
    DeviceBuffer logits_data(64 * 2 * sizeof(std::uint16_t));
    DeviceBuffer ids_data(4 * sizeof(std::int32_t));
    DeviceBuffer mask_data(2 * sizeof(std::int32_t));
    DeviceBuffer out_data(2 * 4 * 2 * sizeof(float));
    Tensor logits(logits_data.p, DType::BF16, {64, 2});
    Tensor sampled(ids_data.p, DType::I32, {2});
    Tensor candidates(ids_data.p, DType::I32, {4});
    Tensor allowed(mask_data.p, DType::I32, {2});
    Tensor sampled_out(out_data.p, DType::FP32, {2, 2});
    Tensor candidates_out(static_cast<float*>(out_data.p) + 4, DType::FP32, {2, 4, 2});
    int failures = 0;
    failures += expect_invalid("rejects valid_rows=0", [&] {
        ops::candidate_logprobs(logits, 0, sampled, &candidates, nullptr, sampled_out,
                                &candidates_out, nullptr);
    });
    failures += expect_invalid("rejects candidates without output", [&] {
        ops::candidate_logprobs(logits, 64, sampled, &candidates, nullptr, sampled_out, nullptr,
                                nullptr);
    });
    failures += expect_invalid("rejects a mask that does not cover the rows", [&] {
        Tensor short_mask(mask_data.p, DType::I32, {1});
        ops::candidate_logprobs(logits, 64, sampled, &candidates, &short_mask, sampled_out,
                                &candidates_out, nullptr);
    });
    failures += expect_invalid("rejects sampled_out shape", [&] {
        Tensor wrong(out_data.p, DType::FP32, {2});
        ops::candidate_logprobs(logits, 64, sampled, nullptr, nullptr, wrong, nullptr, nullptr);
    });
    failures += expect_invalid("rejects output aliasing an input", [&] {
        Tensor alias(logits_data.p, DType::FP32, {2, 2});
        ops::candidate_logprobs(logits, 64, sampled, nullptr, nullptr, alias, nullptr, nullptr);
    });
    return failures;
}

} // namespace

int main() try {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    failures += run_case(make_case("small unconstrained", 64, 50, 3, 4, false, 1));
    failures += run_case(make_case("small masked", 64, 50, 3, 4, true, 2));
    failures += run_case(make_case("sampled only", 1024, 1000, 5, 0, true, 3));
    failures += run_case(make_case("vocabulary unconstrained", 248320, 248077, 2, 77, false, 4));
    failures += run_case(make_case("vocabulary masked", 248320, 248077, 6, 20, true, 5));
    failures += run_validation_cases();
    if (failures != 0) {
        std::cerr << "candidate_logprobs: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "OK candidate_logprobs\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "candidate_logprobs: " << error.what() << '\n';
    return 1;
}
