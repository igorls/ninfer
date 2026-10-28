#include "core/decode_graph.h"
#include "ninfer/ops/linear_split.h"
#include "ops/direct_bf16_weight.h"
#include "ops/input_projection_test_common.h"

#include <algorithm>
#include <array>
#include <functional>
#include <memory>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {
// Same public-input criteria as the existing segmented A16 and A8 suites. The
// oracle independently decodes stored weights and sums represented BF16 inputs in FP64.
constexpr ReductionCriterion a16{2.9e-3, 4.0e-3, 4.5e-3};
constexpr ReductionCriterion fp8_a16{1.0 / 256, 1.0 / 256, 2.0 / 256};
constexpr ReductionCriterion a8{0.04, 1.0 / 256, 0.06};

int run(const Weight& weight, std::span<const int> segments,
         const std::function<double(int, std::span<const float>)>& oracle,
         int tokens, ops::LinearPolicy policy, bool replay) {
    auto activation = make_bf16_activation(2560, tokens, 911U + tokens);
    auto bits = bf16_bits(activation);
    auto input = to_device(bits);
    Tensor x(input.span(), DType::BF16, {2560, tokens});
    const ops::ProjectionWeights projection = ops::SingleProjectionWeight{weight, policy};
    const auto capacity = ops::linear_split_workspace_capacity_bytes(projection, segments, 1, tokens);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 1));
    MemoryArena arena(scratch.span());
    std::vector<std::unique_ptr<GuardedBf16Tensor>> guarded;
    std::vector<Tensor> output;
    for (int rows : segments) {
        guarded.push_back(std::make_unique<GuardedBf16Tensor>(rows, tokens));
        output.push_back(guarded.back()->tensor());
    }
    DeviceContext device;
    const auto launch = [&] {
        auto scope = arena.scope();
        ops::linear_split(x, projection, output, arena, device.execution_view());
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    cuda_synchronize();
    if (replay) {
        definition.capture(device.stream(), launch);
        graph.instantiate(definition);
    }
    int failures = 0;
    for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
        if (phase) {
            for (auto& value : activation) value = -value;
            bits = bf16_bits(activation);
            input.copy_from_host(bits.data(), input.size());
        }
        scratch.fill(phase ? 0xA5 : 0x5A);
        cuda_synchronize();
        for (auto& value : guarded) value->repaint(device.stream());
        if (replay) graph.launch(device.stream()); else launch();
        cuda_synchronize(device.stream());
        int offset = 0;
        for (std::size_t part = 0; part < output.size(); ++part) {
            const auto label = "Flash split N" + std::to_string(weight.n) + " T" +
                                std::to_string(tokens) + " part" + std::to_string(part) +
                                " format=" + std::to_string(int(weight.qtype)) +
                                " policy=" + std::to_string(int(policy));
            failures += guarded[part]->verify_guards(label);
            failures += guarded[part]->verify_fully_written(label);
            const auto values = guarded[part]->values();
            std::vector<double> actual, expected;
            const std::array token_samples{0, tokens / 2, tokens - 1};
            for (auto row : sampled_rows(segments[part], 17))
                for (auto token : token_samples) {
                    actual.push_back(values[std::size_t(token) * segments[part] + row]);
                    expected.push_back(oracle(offset + row,
                        {activation.data() + std::size_t(token) * 2560, 2560}));
                }
            const auto criterion = weight.qtype == QType::BF16 ? a16 :
                                     policy == ops::LinearPolicy::A16Only || tokens <= 48
                                         ? fp8_a16 : a8;
            failures += compare(label, actual, expected, criterion);
            offset += segments[part];
        }
        failures += scratch.verify_guards("Flash split scratch");
        failures += verify_preserved("Flash split input", input, bits);
    }
    return failures;
}

// Target verification changes the physical column count while retaining each
// column's represented input. Preserve the BF16 projection boundary across it.
int decode_columns() {
    constexpr int k = 2560, maximum = 48;
    const std::array segments{6144, 6144, 512, 512};
    auto packed = quantized_weight::make_patterned_weight(
        QType::FP8_E4M3FN_ROW_FP32, 13312, k, 971U);
    std::uint32_t state = 0x5eedU;
    for (std::size_t i = 0; i < packed.code_plane_bytes; ++i) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        auto code = static_cast<std::uint8_t>(state);
        if ((code & 127U) == 127U) code ^= 1U;
        packed.payload[i] = code;
    }
    DevicePackedWeight bank(std::move(packed));
    const auto weight = bank.view();
    const auto values = make_bf16_activation(k, maximum, 977U);
    const auto bits = bf16_bits(values);
    auto input = to_device(bits);
    DeviceContext device;
    const ops::ProjectionWeights projection =
        ops::SingleProjectionWeight{weight, ops::LinearPolicy::AllowA8};
    const auto capacity = ops::linear_split_workspace_capacity_bytes(
        projection, segments, 1, maximum);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 1));
    MemoryArena arena(scratch.span());
    std::vector<std::unique_ptr<GuardedBf16Tensor>> single;
    std::vector<Tensor> single_out;
    std::array<std::vector<std::uint16_t>, 4> reference;
    for (int rows : segments) {
        single.push_back(std::make_unique<GuardedBf16Tensor>(rows, 1));
        single_out.push_back(single.back()->tensor());
    }
    for (int column = 0; column < maximum; ++column) {
        Tensor x(input.span().subspan(std::size_t(column) * k * 2, k * 2),
                 DType::BF16, {k, 1});
        {
            auto scope = arena.scope();
            ops::linear_split(x, projection, single_out, arena, device.execution_view());
        }
        cuda_synchronize(device.stream());
        for (std::size_t part = 0; part < single.size(); ++part) {
            const auto result = single[part]->bits();
            reference[part].insert(reference[part].end(), result.begin(), result.end());
        }
    }
    int failures = 0;
    for (const int tokens : {2, 8, 16, 17, 32, 47, 48}) {
        Tensor x(input.span().subspan(0, std::size_t(tokens) * k * 2),
                 DType::BF16, {k, tokens});
        std::vector<std::unique_ptr<GuardedBf16Tensor>> guarded;
        std::vector<Tensor> outputs;
        for (int rows : segments) {
            guarded.push_back(std::make_unique<GuardedBf16Tensor>(rows, tokens));
            outputs.push_back(guarded.back()->tensor());
        }
        {
            auto scope = arena.scope();
            ops::linear_split(x, projection, outputs, arena, device.execution_view());
        }
        cuda_synchronize(device.stream());
        int offset = 0;
        for (std::size_t part = 0; part < guarded.size(); ++part) {
            const auto result = guarded[part]->bits();
            if (!std::equal(result.begin(), result.end(), reference[part].begin())) {
                std::cerr << "Flash split decode columns change at T=" << tokens
                          << " part=" << part << '\n';
                ++failures;
            }
            failures += guarded[part]->verify_guards("Flash split decode columns");
            std::vector<double> actual, expected;
            for (const auto row : sampled_rows(segments[part], 17)) {
                for (const auto column : {0, tokens / 2, tokens - 1}) {
                    actual.push_back(bf16_to_f32(result[std::size_t(column) * segments[part] + row]));
                    expected.push_back(quantized_weight::dot_fp64(
                        bank.host, offset + row, values.data() + std::size_t(column) * k, k));
                }
            }
            failures += compare("Flash split decode FP64", actual, expected, fp8_a16);
            offset += segments[part];
        }
    }
    failures += scratch.verify_guards("Flash split decode scratch");
    failures += verify_preserved("Flash split decode input", input, bits);
    failures += bank.verify_preserved("Flash split decode weight");
    return failures;
}
} // namespace

int main() {
    if (cuda_unavailable()) return 77;
    int failures = 0;
    for (const bool index : {false, true}) {
        const auto segments = index ? std::vector<int>{512, 128}
                                    : std::vector<int>{6144, 6144, 512, 512};
        direct_bf16_weight::DeviceWeight weight(
            direct_bf16_weight::make_patterned(index ? 640 : 13312, 2560, 919));
        const auto oracle = [&](int row, std::span<const float> input) {
            return direct_bf16_weight::dot_fp64(weight.host, row, input);
        };
        for (int tokens : {1, 2, 8, 16, 17, 32, 48, 64, 65, 128, 129, 193, 449, 1025})
            failures += run(weight.view(), segments, oracle, tokens,
                             ops::LinearPolicy::A16Only, tokens == 17 || tokens == 449);
        failures += weight.verify_preserved("Flash BF16 split weight");
    }
    DevicePackedWeight weight(quantized_weight::make_patterned_weight(
        QType::FP8_E4M3FN_ROW_FP32, 13312, 2560, 929));
    const auto oracle = [&](int row, std::span<const float> input) {
        return quantized_weight::dot_fp64(weight.host, row, input.data(), 2560);
    };
    const std::array segments{6144, 6144, 512, 512};
    failures += decode_columns();
    for (auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8})
        for (int tokens : {1, 2, 8, 16, 17, 32, 47, 48, 49, 64, 65, 128, 129, 193, 449, 1025})
            failures += run(weight.view(), segments, oracle, tokens, policy,
                             tokens == 17 || tokens == 449);
    failures += weight.verify_preserved("Flash FP8 split weight");
    std::cout << (failures ? "FAIL" : "PASS") << " Flash segmented projections\n";
    return failures ? 1 : 0;
}
