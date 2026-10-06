#include "core/arena.h"
#include "core/device.h"
#include "core/weight.h"
#include "core/weight_view.h"
#include "ninfer/ops/weight_input.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <iostream>
#include <utility>
#include <vector>

// Row-scaled FP8 with FP32 row multipliers (FP8_E4M3FN_ROW_FP32): the registered Flash-Next
// problems through the public Linear Op. The oracle decodes w_hat = binary32(c32 * s) from the
// stored words (quantized_weight.h) and accumulates in FP64; the A16 and A8 criteria are the
// Linear suite's named activation-compute criteria.
namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::linear;

std::vector<Invocation> a16_calls() {
    std::vector<Invocation> calls;
    for (int t = 1; t <= 16; ++t) calls.push_back({t});
    for (int t : {17, 24, 25, 32, 33, 47, 48, 49, 63, 64, 65, 95, 96, 97, 127, 128, 129, 512, 1024})
        calls.push_back({t});
    // Permissive policies keep the A16 route below the A8 boundary.
    for (int t : {1, 2, 4, 8, 16}) calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8});
    calls.push_back({1, CallForm::A16Convenience});
    for (int t : {1, 3, 16, 33, 48, 49, 129})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    return calls;
}

std::vector<Invocation> a8_calls() {
    std::vector<Invocation> calls;
    for (int t : {17,  31,  32,  33,  63,  64,  65,  95,  96,  97,  127, 128,  129,  191,
                  192, 193, 287, 288, 289, 384, 385, 511, 512, 513, 768, 769, 1024, 1025,
                  2048, 8192})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8});
    calls.push_back({48, CallForm::Policy, ops::LinearPolicy::AllowA4});
    for (int t : {17, 97, 193, 385, 769, 1025})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8, true});
    return calls;
}

// The artifact route: a complete RowScaleFp32 parent prepared through the public weight-input
// entry must yield the same native operand as the fixture and the same Linear output.
int prepared_native_weight() {
    constexpr std::int32_t n = 2560, k = 6144, tokens = 33;
    int failures             = 0;
    const auto packed        = make_fp8_fp32_weight(n, k, 907U);
    DeviceBuffer payload(packed.payload.size());
    payload.copy_from_host(packed.payload.data(), payload.bytes);

    const std::vector<std::uint64_t> shape{n, k};
    WeightParent parent;
    parent.geometry = weight_geometry(QType::FP8_E4M3FN_ROW_FP32, QuantLayout::RowScaleFp32, shape);
    parent.data     = static_cast<const std::byte*>(payload.p);
    const WeightView view{shape, {WeightRegion{&parent, 0, parent.geometry.elements}}};
    const auto prepared = ops::prepare_linear_weight({view, ops::LinearPolicy::AllowA8});
    const Weight expected = packed.device_weight(payload.p);
    const Weight& actual  = prepared.weight;
    if (prepared.policy != ops::LinearPolicy::AllowA8 || actual.qtype != expected.qtype ||
        actual.layout != expected.layout || actual.scale_dtype != expected.scale_dtype ||
        actual.n != n || actual.k != k || actual.qdata != expected.qdata ||
        actual.scales != expected.scales || actual.group != expected.group ||
        actual.group_size != expected.group_size || actual.scale_nb[0] != expected.scale_nb[0] ||
        actual.scale_nb[1] != expected.scale_nb[1] || actual.scale_ne[0] != expected.scale_ne[0]) {
        std::cerr << "prepared FP32-scale FP8 native Weight differs from its encoded parent\n";
        ++failures;
    }
    try {
        (void)ops::detail::validate_fp8_row_weight(actual, "prepared FP8 FP32");
    } catch (const std::exception& error) {
        std::cerr << "prepared FP32-scale FP8 weight was rejected: " << error.what() << '\n';
        return failures + 1;
    }

    std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * tokens);
    for (std::size_t i = 0; i < bits.size(); ++i)
        bits[i] = static_cast<std::uint16_t>(0x3c00U + (i * 37U) % 0x0300U) ^
                  static_cast<std::uint16_t>((i % 3U) == 0 ? 0x8000U : 0U);
    DeviceBuffer x_buffer(bits.size() * 2);
    x_buffer.copy_from_host(bits.data(), x_buffer.bytes);
    DeviceBuffer out_a(static_cast<std::size_t>(n) * tokens * 2);
    DeviceBuffer out_b(out_a.bytes);
    Tensor x(x_buffer.p, DType::BF16, {k, tokens});
    Tensor ya(out_a.p, DType::BF16, {n, tokens});
    Tensor yb(out_b.p, DType::BF16, {n, tokens});
    DeviceArena workspace(std::max<std::size_t>(
        ops::linear_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_FP32, n, k,
                                             ops::LinearPolicy::AllowA8, tokens, tokens),
        256));
    ops::linear(x, actual, ya, prepared.policy, workspace, nullptr);
    ops::linear(x, expected, yb, ops::LinearPolicy::AllowA8, workspace, nullptr);
    cuda_synchronize();
    std::vector<std::uint16_t> a(static_cast<std::size_t>(n) * tokens), b(a.size());
    out_a.copy_to_host(a.data(), out_a.bytes);
    out_b.copy_to_host(b.data(), out_b.bytes);
    if (a != b) {
        std::cerr << "prepared FP32-scale FP8 Weight changed the Linear output\n";
        ++failures;
    }
    return failures;
}

// Chunk invariance: above the small-T routes (T > 288), an A8 column's output must not depend on
// how many columns share the launch, whether through the tile choice or a split of the final
// wave, so one prompt yields the same projections under every prefill chunking. The input is a
// sequential stream, so each wider case extends the previous one; comparing their whole shared
// prefix also covers the previous launch's final-wave tiles.
int a8_column_invariance() {
    int failures = 0;
    for (const auto& [n, k] : {std::pair{13312, 2560}, std::pair{16384, 2560}, std::pair{2560, 6144}}) {
        const auto packed = make_fp8_fp32_weight(n, k, 941U);
        DeviceBuffer payload(packed.payload.size());
        payload.copy_from_host(packed.payload.data(), payload.bytes);
        const Weight weight = packed.device_weight(payload.p);
        std::vector<std::uint16_t> previous;
        for (const std::int32_t tokens : {289, 385, 513, 769, 2048, 4096, 8192}) {
            // Uniform values make the FP32 accumulation round, so a different K order shows.
            std::vector<float> values(static_cast<std::size_t>(k) * tokens);
            fill_uniform(values, 0x5eedU, -1.0F, 1.0F);
            std::vector<std::uint16_t> bits(values.size());
            for (std::size_t i = 0; i < bits.size(); ++i) bits[i] = f32_to_bf16(values[i]);
            DeviceBuffer x_buffer(bits.size() * 2);
            x_buffer.copy_from_host(bits.data(), x_buffer.bytes);
            DeviceBuffer out(static_cast<std::size_t>(n) * tokens * 2);
            Tensor x(x_buffer.p, DType::BF16, {k, tokens});
            Tensor y(out.p, DType::BF16, {n, tokens});
            DeviceArena workspace(std::max<std::size_t>(
                ops::linear_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_FP32, n, k,
                                                     ops::LinearPolicy::AllowA8, tokens, tokens),
                256));
            ops::linear(x, weight, y, ops::LinearPolicy::AllowA8, workspace, nullptr);
            cuda_synchronize();
            std::vector<std::uint16_t> current(static_cast<std::size_t>(n) * tokens);
            out.copy_to_host(current.data(), current.size() * 2);
            if (!previous.empty() && !std::equal(previous.begin(), previous.end(), current.begin())) {
                std::cerr << "FP8 FP32 A8 [" << n << ',' << k << "]: shared columns change at T="
                          << tokens << '\n';
                ++failures;
            }
            previous = std::move(current);
        }
    }
    return failures;
}

int run_fp8_fp32() {
    int failures = 0;
    struct Problem {
        std::int32_t n, k;
        std::uint32_t seed;
    };
    for (const Problem p : {Problem{13312, 2560, 911U}, Problem{16384, 2560, 919U},
                            Problem{2560, 6144, 929U}}) {
        const auto a16 = a16_calls();
        failures += run_shape("FP8_FP32_A16", ActivationCompute::A16, make_fp8_fp32_weight,
                              {p.n, p.k, p.seed, Comparison::Sampled, true, a16});
        const auto a8 = a8_calls();
        failures += run_shape("FP8_FP32_A8", ActivationCompute::A8, make_fp8_fp32_weight,
                              {p.n, p.k, p.seed + 2U, Comparison::Sampled, true, a8});
        failures += verify_workspace_envelopes(QType::FP8_E4M3FN_ROW_FP32, p.n, p.k);
        const auto a16_capacity = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_FP32, p.n, p.k, ops::LinearPolicy::A16Only, 1, 8192);
        const auto decode_capacity = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_FP32, p.n, p.k, ops::LinearPolicy::AllowA8, 1, 16);
        if (a16_capacity != 0 || decode_capacity != 0) {
            std::cerr << "FP8 FP32 A16 interval reported workspace for [" << p.n << ',' << p.k
                      << "]\n";
            ++failures;
        }
    }

    // A BF16-scale weight never resolves an FP32-scale problem and vice versa.
    for (const auto qtype : {QType::FP8_E4M3FN_ROW_BF16, QType::FP8_E4M3FN_ROW_FP32}) {
        const std::int32_t n = qtype == QType::FP8_E4M3FN_ROW_BF16 ? 13312 : 14336;
        const std::int32_t k = qtype == QType::FP8_E4M3FN_ROW_BF16 ? 2560 : 5120;
        try {
            (void)ops::linear_workspace_capacity_bytes(qtype, n, k, ops::LinearPolicy::AllowA8,
                                                       1, 1);
            std::cerr << "FP8 scale format was not part of the registered problem\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    }

    auto packed = make_fp8_fp32_weight(2560, 6144, 931U);
    try {
        (void)ops::detail::validate_fp8_row_weight(packed.weight, "FP8 FP32 validator test");
    } catch (const std::exception& error) {
        std::cerr << "valid FP8 FP32 metadata was rejected: " << error.what() << '\n';
        ++failures;
    }
    const auto expect_invalid = [&](const char* label, Weight invalid) {
        try {
            (void)ops::detail::validate_fp8_row_weight(invalid, "FP8 FP32 validator test");
            std::cerr << "invalid FP8 FP32 " << label << " was accepted\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    Weight invalid      = packed.weight;
    invalid.layout      = QuantLayout::RowScale;
    expect_invalid("layout", invalid);
    invalid             = packed.weight;
    invalid.scale_dtype = DType::BF16;
    expect_invalid("scale dtype", invalid);
    invalid             = packed.weight;
    invalid.scale_nb[0] = 2;
    expect_invalid("scale word", invalid);
    invalid               = packed.weight;
    invalid.payload_bytes = invalid.payload_bytes - 1;
    expect_invalid("payload bound", invalid);
    try {
        (void)ops::detail::validate_fp8_weight(packed.weight, "BF16-only FP8 consumer");
        std::cerr << "a BF16-scale FP8 consumer accepted FP32 row multipliers\n";
        ++failures;
    } catch (const std::invalid_argument&) {}

    failures += prepared_native_weight();
    failures += a8_column_invariance();
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_fp8_fp32();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " FP8 FP32-scale Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FP8 FP32-scale Linear: " << error.what() << '\n';
        return 1;
    }
}
