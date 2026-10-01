#include "ninfer/ops/hyper_connection.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/op_check.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

// Four-stream hyper-connection Ops at the Flash-Next geometry (4 x 2560, low rank 320). The oracle
// evaluates the complete formula of include/ninfer/ops/hyper_connection.h in FP64 from the
// represented BF16 inputs; it keeps no intermediate BF16 rounding.
namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr int kHidden  = 2560;
constexpr int kConcat  = 10240;
constexpr int kLowRank = 320;
constexpr int kStreams = 4;

// block_input: the production profile rounds N and the low-rank activation to BF16 before the
// two projections and stores BF16; each contributes O(2^-9) relative error to M*N.
constexpr ReductionCriterion kBlockInputCriterion{6.0e-3, 1.0e-4, 2.5e-2};
// injection is stored in FP32 and sees only the BF16 staging of N inside a 10240-term dot, scaled
// by 1/4 and passed through 2*sigmoid (slope <= 1/2): a reduction error, judged normwise.
constexpr ReductionCriterion kInjectionCriterion{1.0e-3, 1.0e-3, 3.0e-3};
// inject: one FP32 FMA (relative 2^-24 of its operands) and the final BF16 store (2^-8).
constexpr PointwiseCriterion kInjectCriterion{1.0e-6, 1.01 / 256.0};

struct Bf16Host {
    std::vector<std::uint16_t> bits;
    std::vector<double> values;
};

Bf16Host random_bf16(std::size_t count, std::uint32_t seed, float scale) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> normal(0.0F, scale);
    Bf16Host result;
    result.bits.resize(count);
    result.values.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        result.bits[i]   = f32_to_bf16(normal(generator));
        result.values[i] = bf16_to_f32(result.bits[i]);
    }
    return result;
}

// Hidden streams with distinct magnitudes so the group norm cannot pass as a flat RMSNorm.
Bf16Host make_hidden(std::int32_t tokens, std::uint32_t seed) {
    Bf16Host result = random_bf16(static_cast<std::size_t>(kConcat) * tokens, seed, 1.0F);
    for (std::int32_t t = 0; t < tokens; ++t)
        for (int s = 0; s < kStreams; ++s) {
            const float scale = 0.25F + 0.6F * static_cast<float>((s + t) % 4);
            for (int c = 0; c < kHidden; ++c) {
                const std::size_t i = static_cast<std::size_t>(t) * kConcat + s * kHidden + c;
                result.bits[i]      = f32_to_bf16(static_cast<float>(result.values[i]) * scale);
                result.values[i]    = bf16_to_f32(result.bits[i]);
            }
        }
    return result;
}

Weight bf16_weight(const DeviceBuffer& buffer, std::int32_t n, std::int32_t k) {
    Weight weight{};
    weight.payload         = buffer.p;
    weight.payload_bytes   = buffer.bytes;
    weight.qtype           = QType::BF16;
    weight.layout          = QuantLayout::Contiguous;
    weight.ndim            = 2;
    weight.shape[0]        = n;
    weight.shape[1]        = k;
    weight.padded_shape[0] = n;
    weight.padded_shape[1] = k;
    weight.qdata           = buffer.p;
    weight.n               = n;
    weight.k               = k;
    return weight;
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

struct Fixture {
    Bf16Host norm, down, up, inject;
    DeviceBuffer norm_d, down_d, up_d, inject_d;

    explicit Fixture(std::uint32_t seed)
        : norm(random_bf16(kConcat, seed, 0.1F)),
          down(random_bf16(static_cast<std::size_t>(kLowRank) * kConcat, seed + 1, 0.02F)),
          up(random_bf16(static_cast<std::size_t>(kConcat) * kLowRank, seed + 2, 0.1F)),
          inject(random_bf16(static_cast<std::size_t>(kStreams) * kConcat, seed + 3, 0.02F)),
          norm_d(to_device(norm.bits)), down_d(to_device(down.bits)), up_d(to_device(up.bits)),
          inject_d(to_device(inject.bits)) {}

    ops::HyperConnectionWeights weights() const {
        return {Tensor(norm_d.p, DType::BF16, {kConcat}), bf16_weight(down_d, kLowRank, kConcat),
                bf16_weight(up_d, kConcat, kLowRank)};
    }

    Weight inject_weight() const { return bf16_weight(inject_d, kStreams, kConcat); }

    // FP64 oracle of one token: block_input [2560] and injection [4].
    void oracle(const double* h, double* block_input, double* injection) const {
        std::vector<double> n(kConcat), low(kLowRank);
        for (int s = 0; s < kStreams; ++s) {
            double sum = 0.0;
            for (int c = 0; c < kHidden; ++c) sum += h[s * kHidden + c] * h[s * kHidden + c];
            const double inv = 1.0 / std::sqrt(sum / kHidden + 1e-6);
            for (int c = 0; c < kHidden; ++c) {
                const int j = s * kHidden + c;
                n[j]        = h[j] * inv * (1.0 + norm.values[j]);
            }
        }
        for (int r = 0; r < kLowRank; ++r) {
            double dot = 0.0;
            for (int j = 0; j < kConcat; ++j) dot += down.values[std::size_t(r) * kConcat + j] * n[j];
            const double x = dot / 4.0;
            low[r]         = x * sigmoid(x);
        }
        for (int c = 0; c < kHidden; ++c) {
            double mean = 0.0;
            for (int s = 0; s < kStreams; ++s) {
                const int row = s * kHidden + c;
                double dot    = 0.0;
                for (int r = 0; r < kLowRank; ++r) dot += up.values[std::size_t(row) * kLowRank + r] * low[r];
                mean += sigmoid(dot) * n[row];
            }
            block_input[c] = mean / 4.0;
        }
        for (int s = 0; s < kStreams; ++s) {
            double dot = 0.0;
            for (int j = 0; j < kConcat; ++j) dot += inject.values[std::size_t(s) * kConcat + j] * n[j];
            injection[s] = 2.0 * sigmoid(dot / 4.0);
        }
    }
};

template <class Function>
void parallel_tokens(std::int32_t tokens, Function function) {
    const int threads = std::max(1, std::min<int>(tokens, std::thread::hardware_concurrency()));
    std::vector<std::thread> workers;
    for (int w = 0; w < threads; ++w)
        workers.emplace_back([&, w] {
            for (std::int32_t t = w; t < tokens; t += threads) function(t);
        });
    for (auto& worker : workers) worker.join();
}

int run_mix_case(const Fixture& fixture, std::int32_t tokens, bool prepare, bool graph) {
    const std::string label = std::string(prepare ? "hyper prepare" : "hyper mix") +
                              " T=" + std::to_string(tokens) + (graph ? " graph" : "");
    int failures      = 0;
    Bf16Host hidden   = make_hidden(tokens, 1000U + static_cast<std::uint32_t>(tokens));
    DeviceBuffer h_d  = to_device(hidden.bits);
    GuardedDeviceBuffer block_d(static_cast<std::size_t>(kHidden) * tokens * 2);
    GuardedDeviceBuffer inj_d(static_cast<std::size_t>(kStreams) * tokens * 4);
    Tensor h(h_d.p, DType::BF16, {kConcat, tokens});
    Tensor block(block_d.data(), DType::BF16, {kHidden, tokens});
    Tensor injection(inj_d.data(), DType::FP32, {kStreams, tokens});
    const auto capacity = ops::hyper_connection_workspace_capacity_bytes(tokens, tokens);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    DeviceContext context;
    const auto weights = fixture.weights();
    const auto inject  = fixture.inject_weight();
    const auto launch  = [&] {
        if (prepare)
            ops::hyper_connection_prepare(h, weights, inject, block, injection, workspace,
                                          context.stream);
        else
            ops::hyper_connection_mix(h, weights, block, workspace, context.stream);
    };
    block_d.fill(0xff);
    inj_d.fill(0xff);
    cuda_synchronize();
    if (graph) {
        // Capture once on a different input, then replay on the tested one.
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        auto other = make_hidden(tokens, 77U);
        h_d.copy_from_host(other.bits.data(), h_d.bytes);
        cuda_synchronize();
        definition.capture(context.stream, launch);
        executable.instantiate(definition);
        executable.launch(context.stream);
        cuda_synchronize(context.stream);
        h_d.copy_from_host(hidden.bits.data(), h_d.bytes);
        block_d.fill(0xff);
        inj_d.fill(0xff);
        cuda_synchronize();
        executable.launch(context.stream);
    } else {
        launch();
    }
    cuda_synchronize(context.stream);
    if (workspace.peak_used() > capacity || workspace.used() != 0) {
        std::cerr << label << ": workspace query/scope mismatch\n";
        ++failures;
    }
    failures += block_d.verify_guards(label + " block_input");
    if (prepare) failures += inj_d.verify_guards(label + " injection");

    std::vector<double> expected_block(static_cast<std::size_t>(kHidden) * tokens);
    std::vector<double> expected_injection(static_cast<std::size_t>(kStreams) * tokens);
    parallel_tokens(tokens, [&](std::int32_t t) {
        fixture.oracle(hidden.values.data() + static_cast<std::size_t>(t) * kConcat,
                       expected_block.data() + static_cast<std::size_t>(t) * kHidden,
                       expected_injection.data() + static_cast<std::size_t>(t) * kStreams);
    });
    const auto got_block = from_device_bf16(block_d.data(), expected_block.size());
    failures += verify_reduction(label + " block_input", got_block, expected_block,
                                 kBlockInputCriterion);
    if (prepare) {
        const auto got_injection = from_device<float>(inj_d.data(), expected_injection.size());
        const std::vector<double> got(got_injection.begin(), got_injection.end());
        failures += verify_reduction(label + " injection", got, expected_injection,
                                     kInjectionCriterion);
    }
    if (from_device<std::uint16_t>(h_d, hidden.bits.size()) != hidden.bits) {
        std::cerr << label << ": hidden was modified\n";
        ++failures;
    }
    return failures;
}

int run_inject_case(std::int32_t tokens, bool graph) {
    const std::string label = "hyper inject T=" + std::to_string(tokens) + (graph ? " graph" : "");
    auto hidden             = make_hidden(tokens, 3000U + static_cast<std::uint32_t>(tokens));
    auto output = random_bf16(static_cast<std::size_t>(kHidden) * tokens, 17U + tokens, 0.7F);
    std::vector<float> injection(static_cast<std::size_t>(kStreams) * tokens);
    std::mt19937 generator(5U + tokens);
    std::uniform_real_distribution<float> gate(0.0F, 2.0F);
    for (float& value : injection) value = gate(generator);
    GuardedDeviceBuffer h_d(hidden.bits.size() * 2);
    h_d.copy_from_host(hidden.bits.data(), h_d.bytes());
    DeviceBuffer out_d   = to_device(output.bits);
    DeviceBuffer inj_d   = to_device(injection);
    Tensor h(h_d.data(), DType::BF16, {kConcat, tokens});
    Tensor out(out_d.p, DType::BF16, {kHidden, tokens});
    Tensor inj(inj_d.p, DType::FP32, {kStreams, tokens});
    DeviceContext context;
    if (graph) {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        definition.capture(context.stream,
                           [&] { ops::hyper_connection_inject(out, inj, h, context.stream); });
        executable.instantiate(definition);
        executable.launch(context.stream);
    } else {
        ops::hyper_connection_inject(out, inj, h, context.stream);
    }
    cuda_synchronize(context.stream);
    int failures = h_d.verify_guards(label);
    std::vector<double> expected(hidden.values.size());
    for (std::int32_t t = 0; t < tokens; ++t)
        for (int s = 0; s < kStreams; ++s)
            for (int c = 0; c < kHidden; ++c) {
                const std::size_t i = static_cast<std::size_t>(t) * kConcat + s * kHidden + c;
                expected[i]         = hidden.values[i] +
                              output.values[static_cast<std::size_t>(t) * kHidden + c] *
                                  static_cast<double>(injection[static_cast<std::size_t>(t) * kStreams + s]);
            }
    const auto got = from_device_bf16(h_d.data(), expected.size());
    failures += verify_pointwise(label, got, expected, kInjectCriterion);
    if (from_device<std::uint16_t>(out_d, output.bits.size()) != output.bits ||
        from_device<float>(inj_d, injection.size()) != injection) {
        std::cerr << label << ": read-only input was modified\n";
        ++failures;
    }
    return failures;
}

// expand is an exact index mapping: every stream of every token must carry the bit pattern of x.
int run_expand_case(std::int32_t tokens, bool graph) {
    const std::string label = "hyper expand T=" + std::to_string(tokens) + (graph ? " graph" : "");
    auto x = random_bf16(static_cast<std::size_t>(kHidden) * tokens, 4000U + tokens, 1.0F);
    DeviceBuffer x_d = to_device(x.bits);
    GuardedDeviceBuffer h_d(static_cast<std::size_t>(kConcat) * tokens * 2);
    Tensor input(x_d.p, DType::BF16, {kHidden, tokens});
    Tensor hidden(h_d.data(), DType::BF16, {kConcat, tokens});
    DeviceContext context;
    h_d.fill(0xff);
    cuda_synchronize();
    if (graph) {
        // Capture on a different input, then replay on the tested one.
        const auto other = random_bf16(x.bits.size(), 91U, 1.0F);
        x_d.copy_from_host(other.bits.data(), x_d.bytes);
        cuda_synchronize();
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        definition.capture(context.stream,
                           [&] { ops::hyper_connection_expand(input, hidden, context.stream); });
        executable.instantiate(definition);
        executable.launch(context.stream);
        cuda_synchronize(context.stream);
        x_d.copy_from_host(x.bits.data(), x_d.bytes);
        h_d.fill(0xff);
        cuda_synchronize();
        executable.launch(context.stream);
    } else {
        ops::hyper_connection_expand(input, hidden, context.stream);
    }
    cuda_synchronize(context.stream);
    int failures = h_d.verify_guards(label);
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(kConcat) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t)
        for (int s = 0; s < kStreams; ++s)
            std::copy_n(x.bits.begin() + static_cast<std::ptrdiff_t>(t) * kHidden, kHidden,
                        expected.begin() + static_cast<std::ptrdiff_t>(t) * kConcat +
                            static_cast<std::ptrdiff_t>(s) * kHidden);
    if (from_device<std::uint16_t>(h_d.data(), expected.size()) != expected) {
        std::cerr << label << ": hidden is not four exact copies of x\n";
        ++failures;
    }
    if (from_device<std::uint16_t>(x_d, x.bits.size()) != x.bits) {
        std::cerr << label << ": x was modified\n";
        ++failures;
    }
    return failures;
}

int run_hyper_connection() {
    int failures = 0;
    const Fixture fixture(11U);
    for (std::int32_t tokens : {1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 31, 32, 33, 63, 64, 65, 256, 512,
                                8192}) {
        failures += run_mix_case(fixture, tokens, true, false);
        failures += run_mix_case(fixture, tokens, false, false);
    }
    for (std::int32_t tokens : {1, 8, 9, 65}) {
        failures += run_mix_case(fixture, tokens, true, true);
        failures += run_mix_case(fixture, tokens, false, true);
    }
    for (std::int32_t tokens : {1, 7, 9, 300, 8192, 65537}) failures += run_inject_case(tokens, false);
    failures += run_inject_case(4, true);
    for (std::int32_t tokens : {1, 2, 7, 9, 513, 8192, 65537}) failures += run_expand_case(tokens, false);
    failures += run_expand_case(4, true);

    // The capacity of an interval covers every point in it.
    const auto interval = ops::hyper_connection_workspace_capacity_bytes(1, 512);
    for (std::int32_t tokens : {1, 8, 9, 511, 512}) {
        if (ops::hyper_connection_workspace_capacity_bytes(tokens, tokens) > interval) {
            std::cerr << "hyper workspace interval does not cover T=" << tokens << '\n';
            ++failures;
        }
    }
    for (const auto [first, last] : {std::pair{0, 1}, std::pair{4, 3}}) {
        try {
            (void)ops::hyper_connection_workspace_capacity_bytes(first, last);
            std::cerr << "invalid hyper workspace interval accepted\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_hyper_connection();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " hyper_connection\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "hyper_connection: " << error.what() << '\n';
        return 1;
    }
}
