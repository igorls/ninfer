// Qualification of the NVFP4-bank SparseMoe overload (512 experts, top-10, H 2560,
// intermediate 640) against an independent FP64 oracle at the Qwen3.8-Flash-Next geometry.
//
// The fixture fills both complete 512-expert banks with random E2M1 codes, UE4M3 block scales
// (including all-zero groups) and per-expert divisors; the oracle decodes every weight it uses from
// those stored bytes. Every token column is one of 64 input patterns whose FP64 oracle is computed
// once: pattern 0 forces an exact 16-way score tie across the 10th/11th boundary (experts
// 100..115 share one router row), every other pattern is resampled until its 10th and 11th scores
// are separated, so FP32 scoring cannot legitimately change the selected set.

#include "core/weight.h"
#include "ninfer/ops/sparse_moe.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using ninfer::ops::LinearPolicy;
using ninfer::ops::Nvfp4ExpertBankWeight;

namespace {

constexpr std::int32_t kHidden       = 2560;
constexpr std::int32_t kExperts      = 512;
constexpr std::int32_t kTopK         = 10;
constexpr std::int32_t kIntermediate = 640;
constexpr std::int32_t kPatterns     = 64;
constexpr std::int32_t kTieFirst     = 100; // experts 100..115 share one router row
constexpr std::int32_t kTieCount     = 16;

// Named criteria, one per arithmetic profile of the contract. Both are normwise over the complete
// [2560,T] output with a gross pointwise cap relative to the largest reference magnitude.
//
// A16: the expert inputs keep their represented BF16 values; the private BF16 storage of each
// SwiGLU product and the final BF16 store bound the error (measured maxima over the complete case
// matrix: rel-L2 3.55e-3, pointwise 3.5e-3 of the largest reference).
constexpr ReductionCriterion kSparseMoeNvfp4A16Criterion{
    /*relative_l2*/ 6.0e-3,
    /*gross_absolute*/ 0.0,
    /*gross_relative_to_max_reference*/ 6.0e-3,
};
// A4 (admitted by AllowA4 banks): x and every SwiGLU product are quantized to E2M1 with one
// dynamic UE4M3 scale per 16 values before the routed projections, so the routed contribution
// carries two FP4 activation quantizations (measured maxima: rel-L2 7.49e-2, pointwise 6.1e-2 of
// the largest reference).
constexpr ReductionCriterion kSparseMoeNvfp4A4Criterion{
    /*relative_l2*/ 1.2e-1,
    /*gross_absolute*/ 0.0,
    /*gross_relative_to_max_reference*/ 1.2e-1,
};

std::uint64_t splitmix(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z               = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z               = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

template <class Function>
void parallel_for(std::int32_t count, Function&& function) {
    const unsigned available   = std::max(1U, std::thread::hardware_concurrency());
    const std::int32_t threads = std::min(count, static_cast<std::int32_t>(available));
    std::vector<std::thread> workers;
    for (std::int32_t thread = 0; thread < threads; ++thread) {
        const std::int32_t begin =
            static_cast<std::int32_t>(static_cast<std::int64_t>(count) * thread / threads);
        const std::int32_t end =
            static_cast<std::int32_t>(static_cast<std::int64_t>(count) * (thread + 1) / threads);
        workers.emplace_back([&, begin, end] {
            for (std::int32_t index = begin; index < end; ++index) { function(index); }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }
}

// Test-owned NVFP4 expert bank: expert-major codes, expert-major M128x4-swizzled scale planes,
// and one FP32 divisor per expert (expert_block_scale_k16_m128x4_v1).
class HostBank {
public:
    HostBank(std::int32_t n, std::int32_t k, std::uint64_t seed)
        : n_(n), k_(k), code_bytes_(static_cast<std::size_t>(n) * k / 2),
          scale_bytes_(static_cast<std::size_t>(n) * k / 16), codes_(code_bytes_ * kExperts),
          scales_(scale_bytes_ * kExperts), divisors_(kExperts) {
        std::uint64_t state = seed;
        for (std::size_t i = 0; i < codes_.size(); i += 8) {
            const std::uint64_t word = splitmix(state);
            std::memcpy(codes_.data() + i, &word, 8);
        }
        // UE4M3 scales 0x30..0x57 (0.5..14), with about 1/64 of the groups all-zero.
        for (std::size_t i = 0; i < scales_.size(); i += 8) {
            std::uint64_t word = splitmix(state);
            for (int b = 0; b < 8; ++b, word >>= 8) {
                const std::uint8_t r = static_cast<std::uint8_t>(word);
                scales_[i + b] = (r & 63U) == 0 ? 0 : static_cast<std::uint8_t>(0x30 + r % 40);
            }
        }
        for (std::int32_t e = 0; e < kExperts; ++e) {
            divisors_[e] = 384.0F + static_cast<float>((e * 37) % 512);
        }
        for (int v = 0; v < 16; ++v) {
            e2m1_[v] = quantized_weight::detail::decode_e2m1(static_cast<std::uint8_t>(v));
        }
        for (int v = 0; v < 256; ++v) { // the fixture never stores the NaN codes 0x7f/0xff
            e4m3_[v] = (v & 0x7f) == 0x7f
                           ? 0.0
                           : quantized_weight::detail::decode_e4m3fn(static_cast<std::uint8_t>(v));
        }
    }

    // FP64 dot product of logical row `row` of expert `e` with x[0..K).
    double dot(std::int32_t e, std::int32_t row, const double* x) const {
        const std::uint8_t* codes =
            codes_.data() + code_bytes_ * e + static_cast<std::size_t>(row) * (k_ / 2);
        const std::uint8_t* scales = scales_.data() + scale_bytes_ * e;
        double sum                 = 0.0;
        for (std::int32_t group = 0; group < k_ / 16; ++group) {
            const std::size_t offset =
                static_cast<std::size_t>((row / 128) * (k_ / 64) + group / 4) * 512 +
                static_cast<std::size_t>(row % 32) * 16 +
                static_cast<std::size_t>((row % 128) / 32) * 4 +
                static_cast<std::size_t>(group % 4);
            double group_sum = 0.0;
            for (int j = 0; j < 16; j += 2) {
                const std::uint8_t pair = codes[group * 8 + j / 2];
                group_sum += e2m1_[pair & 15U] * x[group * 16 + j] +
                             e2m1_[pair >> 4] * x[group * 16 + j + 1];
            }
            sum += e4m3_[scales[offset]] * group_sum;
        }
        return sum / static_cast<double>(divisors_[e]);
    }

    Nvfp4ExpertBankWeight device(const DeviceBuffer& codes, const DeviceBuffer& scales,
                                 const DeviceBuffer& divisors, LinearPolicy policy) const {
        Nvfp4ExpertBankWeight bank;
        bank.codes                  = static_cast<const std::byte*>(codes.p);
        bank.scales                 = static_cast<const std::byte*>(scales.p);
        bank.weight_scale_divisors  = static_cast<const float*>(divisors.p);
        bank.experts                = kExperts;
        bank.n                      = n_;
        bank.k                      = k_;
        bank.code_bytes_per_expert  = code_bytes_;
        bank.scale_bytes_per_expert = scale_bytes_;
        bank.policy                 = policy;
        return bank;
    }

    const std::vector<std::uint8_t>& codes() const { return codes_; }

    const std::vector<std::uint8_t>& scales() const { return scales_; }

    const std::vector<float>& divisors() const { return divisors_; }

private:
    std::int32_t n_, k_;
    std::size_t code_bytes_, scale_bytes_;
    std::vector<std::uint8_t> codes_, scales_;
    std::vector<float> divisors_;
    double e2m1_[16]{};
    double e4m3_[256]{};
};

struct HostBf16 {
    std::int32_t n = 0, k = 0;
    std::vector<std::uint16_t> bits;
    std::vector<double> values;

    HostBf16(std::int32_t rows, std::int32_t columns, float limit, std::uint32_t seed)
        : n(rows), k(columns), bits(static_cast<std::size_t>(rows) * columns), values(bits.size()) {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(-limit, limit);
        for (std::size_t i = 0; i < bits.size(); ++i) {
            bits[i]   = f32_to_bf16(distribution(generator));
            values[i] = bf16_to_f32(bits[i]);
        }
    }

    void copy_row(std::int32_t from, std::int32_t to) {
        std::copy_n(bits.begin() + static_cast<std::ptrdiff_t>(from) * k, k,
                    bits.begin() + static_cast<std::ptrdiff_t>(to) * k);
        std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(from) * k, k,
                    values.begin() + static_cast<std::ptrdiff_t>(to) * k);
    }

    double dot(std::int32_t row, const double* x) const {
        const double* w = values.data() + static_cast<std::size_t>(row) * k;
        double sum      = 0.0;
        for (std::int32_t c = 0; c < k; ++c) { sum += w[c] * x[c]; }
        return sum;
    }

    Weight device(const DeviceBuffer& buffer) const {
        Weight weight{};
        weight.payload         = buffer.p;
        weight.payload_bytes   = bits.size() * sizeof(std::uint16_t);
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
};

double silu(double v) { return v / (1.0 + std::exp(-v)); }

struct Pattern {
    std::vector<std::uint16_t> x_bits;
    std::vector<double> x;
    std::vector<double> reference; // FP64 moe(x), [2560]
    double boundary_gap = 0.0;     // s[9] - s[10] of the FP64 ranking
};

class Fixture {
public:
    Fixture()
        : gate_up_(2 * kIntermediate, kHidden, 0x51ULL), down_(kHidden, kIntermediate, 0x52ULL),
          router_(kExperts, kHidden, 0.05F, 11U), shared_expert_gate_(1, kHidden, 0.05F, 12U),
          shared_gate_(kIntermediate, kHidden, 0.05F, 13U),
          shared_up_(kIntermediate, kHidden, 0.05F, 14U),
          shared_down_(kHidden, kIntermediate, 0.08F, 15U) {
        for (std::int32_t e = 1; e < kTieCount; ++e) { router_.copy_row(kTieFirst, kTieFirst + e); }

        d_gate_up_codes_      = to_device(gate_up_.codes());
        d_gate_up_scales_     = to_device(gate_up_.scales());
        d_gate_up_divisors_   = to_device(gate_up_.divisors());
        d_down_codes_         = to_device(down_.codes());
        d_down_scales_        = to_device(down_.scales());
        d_down_divisors_      = to_device(down_.divisors());
        d_router_             = to_device(router_.bits);
        d_shared_expert_gate_ = to_device(shared_expert_gate_.bits);
        d_shared_gate_        = to_device(shared_gate_.bits);
        d_shared_up_          = to_device(shared_up_.bits);
        d_shared_down_        = to_device(shared_down_.bits);

        patterns_.resize(kPatterns);
        for (std::int32_t p = 0; p < kPatterns; ++p) { make_pattern(p); }
    }

    ops::SparseMoeNvfp4BankWeights weights(LinearPolicy policy) const {
        return {router_.device(d_router_),
                shared_expert_gate_.device(d_shared_expert_gate_),
                gate_up_.device(d_gate_up_codes_, d_gate_up_scales_, d_gate_up_divisors_, policy),
                down_.device(d_down_codes_, d_down_scales_, d_down_divisors_, policy),
                shared_gate_.device(d_shared_gate_),
                shared_up_.device(d_shared_up_),
                shared_down_.device(d_shared_down_)};
    }

    const Pattern& pattern(std::int32_t p) const { return patterns_[p]; }

    int verify_persistent_inputs() const {
        int failures = 0;
        failures += verify_exact(
            "gate_up codes preserved",
            from_device<std::uint8_t>(d_gate_up_codes_, gate_up_.codes().size()), gate_up_.codes());
        failures += verify_exact("down codes preserved",
                                 from_device<std::uint8_t>(d_down_codes_, down_.codes().size()),
                                 down_.codes());
        failures +=
            verify_exact("router preserved",
                         from_device<std::uint16_t>(d_router_, router_.bits.size()), router_.bits);
        return failures;
    }

private:
    void make_pattern(std::int32_t index) {
        Pattern& pattern = patterns_[index];
        pattern.x_bits.resize(kHidden);
        pattern.x.resize(kHidden);
        std::vector<double> scores(kExperts);
        std::vector<std::int32_t> order(kExperts);
        for (int attempt = 0;; ++attempt) {
            if (attempt > 200) { throw std::logic_error("could not draw a separated pattern"); }
            std::mt19937 generator(0x1000U + static_cast<std::uint32_t>(index) * 977U +
                                   static_cast<std::uint32_t>(attempt));
            std::uniform_real_distribution<float> distribution(-1.7F, 1.7F);
            for (std::int32_t c = 0; c < kHidden; ++c) {
                float value = distribution(generator);
                if (index == 0) {
                    // Aligned with the shared tie row: experts 100..115 dominate with one score.
                    const double w =
                        router_.values[static_cast<std::size_t>(kTieFirst) * kHidden + c];
                    value = w > 0.0 ? 1.5F : (w < 0.0 ? -1.5F : 0.0F);
                }
                pattern.x_bits[c] = f32_to_bf16(value);
                pattern.x[c]      = bf16_to_f32(pattern.x_bits[c]);
            }
            for (std::int32_t e = 0; e < kExperts; ++e) {
                scores[e] = router_.dot(e, pattern.x.data());
            }
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                             [&](std::int32_t a, std::int32_t b) { return scores[a] > scores[b]; });
            pattern.boundary_gap = scores[order[kTopK - 1]] - scores[order[kTopK]];
            if (index == 0) { break; }
            bool tie_member = false;
            for (int r = 0; r <= kTopK; ++r) {
                tie_member |= order[r] >= kTieFirst && order[r] < kTieFirst + kTieCount;
            }
            if (!tie_member && pattern.boundary_gap >= 5.0e-3) { break; }
        }
        if (index == 0) {
            for (int r = 0; r < kTopK; ++r) {
                if (order[r] != kTieFirst + r) {
                    throw std::logic_error("tie pattern did not rank experts 100..109 first");
                }
            }
        }

        // FP64 oracle: softmax over the ten selected scores equals the renormalized top-10.
        const double top   = scores[order[0]];
        double denominator = 0.0;
        std::array<double, kTopK> weight{};
        for (int r = 0; r < kTopK; ++r) {
            weight[r] = std::exp(scores[order[r]] - top);
            denominator += weight[r];
        }
        for (double& w : weight) { w /= denominator; }

        std::array<std::vector<double>, kTopK> products;
        for (int r = 0; r < kTopK; ++r) {
            products[r].resize(kIntermediate);
            const std::int32_t e = order[r];
            parallel_for(kIntermediate, [&](std::int32_t row) {
                const double g   = gate_up_.dot(e, row, pattern.x.data());
                const double u   = gate_up_.dot(e, kIntermediate + row, pattern.x.data());
                products[r][row] = silu(g) * u;
            });
        }
        std::vector<double> shared_product(kIntermediate);
        parallel_for(kIntermediate, [&](std::int32_t row) {
            shared_product[row] = silu(shared_gate_.dot(row, pattern.x.data())) *
                                  shared_up_.dot(row, pattern.x.data());
        });
        const double shared_scale =
            1.0 / (1.0 + std::exp(-shared_expert_gate_.dot(0, pattern.x.data())));
        pattern.reference.assign(kHidden, 0.0);
        parallel_for(kHidden, [&](std::int32_t row) {
            double value = shared_scale * shared_down_.dot(row, shared_product.data());
            for (int r = 0; r < kTopK; ++r) {
                value += weight[r] * down_.dot(order[r], row, products[r].data());
            }
            pattern.reference[row] = value;
        });
    }

    HostBank gate_up_, down_;
    HostBf16 router_, shared_expert_gate_, shared_gate_, shared_up_, shared_down_;
    DeviceBuffer d_gate_up_codes_, d_gate_up_scales_, d_gate_up_divisors_;
    DeviceBuffer d_down_codes_, d_down_scales_, d_down_divisors_;
    DeviceBuffer d_router_, d_shared_expert_gate_, d_shared_gate_, d_shared_up_, d_shared_down_;
    std::vector<Pattern> patterns_;
};

struct Case {
    std::int32_t tokens;
    bool graph_replay;
};

struct Profile {
    const char* name;
    LinearPolicy policy;
    const ReductionCriterion& criterion;
    std::span<const Case> cases;
};

int run_case(const Fixture& fixture, const Profile& profile, const Case& c) {
    const std::int32_t tokens = c.tokens;
    const std::string label   = std::string(profile.name) + " T=" + std::to_string(tokens) +
                              (c.graph_replay ? " graph" : "");
    std::vector<std::uint16_t> input(static_cast<std::size_t>(kHidden) * tokens);
    std::vector<double> reference(input.size());
    for (std::int32_t t = 0; t < tokens; ++t) {
        // Stride 7 is coprime with 64: every run of 64 consecutive tokens covers every pattern,
        // and pattern 0 (the exact tie) appears at t = 0.
        const Pattern& pattern = fixture.pattern((t * 7) % kPatterns);
        std::copy(pattern.x_bits.begin(), pattern.x_bits.end(),
                  input.begin() + static_cast<std::ptrdiff_t>(t) * kHidden);
        std::copy(pattern.reference.begin(), pattern.reference.end(),
                  reference.begin() + static_cast<std::ptrdiff_t>(t) * kHidden);
    }

    DeviceBuffer device_input = to_device(input);
    GuardedDeviceBuffer output(input.size() * sizeof(std::uint16_t));
    output.fill(0xff); // BF16 NaN: every element must be overwritten
    const auto weights = fixture.weights(profile.policy);
    Tensor x(device_input.p, DType::BF16, {kHidden, tokens});
    Tensor destination(output.data(), DType::BF16, {kHidden, tokens});
    const std::size_t capacity =
        ops::sparse_moe_workspace_capacity_bytes(profile.policy, profile.policy, tokens, tokens);
    WorkspaceArena workspace(capacity);

    if (c.graph_replay) {
        cudaStream_t stream  = nullptr;
        cudaGraph_t graph    = nullptr;
        cudaGraphExec_t exec = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "graph stream");
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");
        ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::Store, destination, workspace, stream);
        cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
        cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate");
        cuda_check(cudaGraphLaunch(exec, stream), "launch");
        cuda_check(cudaStreamSynchronize(stream), "synchronize launch");
        output.fill(0xff);
        cuda_check(cudaGraphLaunch(exec, stream), "replay");
        cuda_check(cudaStreamSynchronize(stream), "synchronize replay");
        cuda_check(cudaGraphExecDestroy(exec), "destroy exec");
        cuda_check(cudaGraphDestroy(graph), "destroy graph");
        cuda_check(cudaStreamDestroy(stream), "destroy stream");
    } else {
        ops::sparse_moe(x, weights, ops::SparseMoeEpilogue::Store, destination, workspace, nullptr);
        cuda_synchronize();
    }

    const std::vector<double> actual = from_device_bf16(output.data(), input.size());
    const ReductionStats stats       = compute_reduction_stats(actual.data(), reference.data(),
                                                               static_cast<std::int64_t>(actual.size()));
    const bool pass =
        reduction_passes(stats, static_cast<std::int64_t>(actual.size()), profile.criterion);
    std::cout << (pass ? "PASS " : "FAIL ") << label << " rel_l2=" << stats.relative_l2
              << " max_abs=" << stats.maximum_absolute_error
              << " max_ref=" << stats.maximum_absolute_reference << " gross_ratio="
              << stats.maximum_absolute_error / gross_error_limit(stats, profile.criterion)
              << " non_finite_at=" << stats.first_non_finite << "\n";
    int failures = pass ? 0 : 1;

    // The tie pattern's column must match its oracle closely even under A4: a wrong selection
    // inside experts 100..115 changes whole experts, not rounding.
    {
        std::vector<double> tie_actual(actual.begin(), actual.begin() + kHidden);
        const ReductionStats tie =
            compute_reduction_stats(tie_actual.data(), reference.data(), kHidden);
        if (!reduction_passes(tie, kHidden, profile.criterion)) {
            std::cerr << label << ": exact-tie column (experts 100..109 expected) failed rel_l2="
                      << tie.relative_l2 << "\n";
            ++failures;
        }
    }
    failures += output.verify_guards(label);
    failures += verify_exact((label + " input preserved").c_str(),
                             from_device<std::uint16_t>(device_input, input.size()), input);
    if (workspace.used() != 0 || workspace.peak_used() != capacity) {
        std::cerr << label << ": workspace query/execution high-water mismatch ("
                  << workspace.peak_used() << " vs " << capacity << ")\n";
        ++failures;
    }
    return failures;
}

int run_profile(const Fixture& fixture, const Profile& profile) {
    int failures        = 0;
    std::size_t witness = 0;
    std::int32_t last   = 0;
    for (const Case& c : profile.cases) {
        failures += run_case(fixture, profile, c);
        witness = std::max(witness, ops::sparse_moe_workspace_capacity_bytes(
                                        profile.policy, profile.policy, c.tokens, c.tokens));
        last    = std::max(last, c.tokens);
    }
    const std::size_t interval =
        ops::sparse_moe_workspace_capacity_bytes(profile.policy, profile.policy, 1, last);
    if (interval < witness) {
        std::cerr << profile.name << ": interval capacity below a single-T requirement\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    Fixture fixture;
    double smallest_gap = 1.0e30;
    for (std::int32_t p = 1; p < kPatterns; ++p) {
        smallest_gap = std::min(smallest_gap, fixture.pattern(p).boundary_gap);
    }
    std::cout << "patterns: " << kPatterns << ", smallest separated top-10 gap " << smallest_gap
              << "\n";

    // Public extents only: T = 1, a small decode batch, its end, the first grouped extents, the
    // grouped interior, the prefill anchor and a full 8192-token chunk. A16Only keeps every
    // routed product on represented BF16 activations at every T.
    static constexpr std::array<Case, 12> kA16Cases{{{1, false},
                                                    {1, true},
                                                    {2, false},
                                                    {8, false},
                                                    {9, false},
                                                    {47, false},
                                                    {48, true},
                                                    {49, false},
                                                    {64, true},
                                                    {255, false},
                                                    {256, false},
                                                    {1000, false}}};
    static constexpr std::array<Case, 15> kA4Cases{{{1, false},
                                                    {7, false},
                                                    {8, true},
                                                    {9, false},
                                                    {10, false},
                                                    {47, false},
                                                    {48, true},
                                                    {49, false},
                                                    {255, false},
                                                    {256, false},
                                                    {257, false},
                                                    {300, true},
                                                    {1000, false},
                                                    {4097, false},
                                                    {8192, false}}};
    const std::array<Profile, 2> profiles{{
        {"sparse_moe nvfp4-bank A16Only", LinearPolicy::A16Only, kSparseMoeNvfp4A16Criterion,
         kA16Cases},
        {"sparse_moe nvfp4-bank AllowA4", LinearPolicy::AllowA4, kSparseMoeNvfp4A4Criterion,
         kA4Cases},
    }};
    int failures = 0;
    for (const Profile& profile : profiles) { failures += run_profile(fixture, profile); }
    failures += fixture.verify_persistent_inputs();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " sparse_moe nvfp4-bank correctness\n";
    return failures == 0 ? 0 : 1;
}
