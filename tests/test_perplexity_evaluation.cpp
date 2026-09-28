#include "evaluation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int require(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

int check_partition(std::size_t tokens, std::uint32_t context, std::uint32_t stride) {
    const auto windows   = ninfer::perplexity::plan_windows(tokens, context, stride);
    int failures         = 0;
    std::size_t frontier = 1;
    for (const auto& window : windows) {
        failures += require(window.input_begin < window.input_end, "window input is nonempty");
        failures +=
            require(window.input_end - window.input_begin <= context, "window respects context");
        failures += require(window.target_begin == frontier, "target ranges are contiguous");
        failures +=
            require(window.target_end <= window.input_end, "targets are inside input window");
        failures += require(window.first_target == window.target_begin - window.input_begin,
                            "local target matches global range");
        frontier = window.target_end;
    }
    failures += require(frontier == tokens, "targets cover every token after the first");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += check_partition(2, 4096, 2048);
    failures += check_partition(4096, 4096, 2048);
    failures += check_partition(4097, 4096, 2048);
    failures += check_partition(12001, 4096, 2048);

    const std::array<ninfer::perplexity::WindowPlan, 3> expected{{
        {.input_begin = 0, .input_end = 6, .target_begin = 1, .target_end = 6, .first_target = 1},
        {.input_begin = 2, .input_end = 8, .target_begin = 6, .target_end = 8, .first_target = 4},
        {.input_begin = 4, .input_end = 10, .target_begin = 8, .target_end = 10, .first_target = 4},
    }};
    const auto planned = ninfer::perplexity::plan_windows(10, 6, 2);
    failures += require(planned.size() == expected.size(), "window count matches the protocol");
    for (std::size_t index = 0; index < std::min(planned.size(), expected.size()); ++index) {
        const auto& actual = planned[index];
        const auto& wanted = expected[index];
        failures += require(actual.input_begin == wanted.input_begin &&
                                actual.input_end == wanted.input_end &&
                                actual.target_begin == wanted.target_begin &&
                                actual.target_end == wanted.target_end &&
                                actual.first_target == wanted.first_target,
                            "window boundaries match the protocol");
    }

    const std::vector<float> first{-1.0F, -2.0F};
    const std::vector<float> second{-3.0F};
    ninfer::perplexity::ScoreAggregate a;
    ninfer::perplexity::ScoreAggregate b;
    a.add(first);
    b.add(second);
    a.add(b);
    failures += require(a.scored_tokens == 3, "aggregate counts target tokens");
    failures += require(std::abs(a.total_nll - 6.0) < 1e-12, "aggregate accumulates FP64 NLL");
    failures += require(std::abs(a.mean_nll() - 2.0) < 1e-12, "aggregate computes mean NLL");
    failures += require(std::abs(a.ppl() - std::exp(2.0)) < 1e-12, "aggregate computes perplexity");

    // Top-K divergence: P = (0.7, 0.2 | tail 0.1), Q = (0.5, 0.3 | tail 0.2).
    const std::vector<float> p{std::log(0.7F), std::log(0.2F)};
    const std::vector<float> q{std::log(0.5F), std::log(0.3F)};
    const double expected_kl = 0.7 * std::log(0.7 / 0.5) + 0.2 * std::log(0.2 / 0.3) +
                               0.1 * std::log(0.1 / 0.2);
    failures += require(std::abs(ninfer::perplexity::top_k_kl_divergence(p, q) - expected_kl) < 1e-6,
                        "top-K divergence includes the tail bucket");
    failures += require(ninfer::perplexity::top_k_kl_divergence(p, p) < 1e-9,
                        "top-K divergence of equal distributions is zero");
    // Q with (numerically) no tail while P keeps 0.1 of its mass there stays finite and large.
    const std::vector<float> saturated{std::log(0.999999F), std::log(1.0e-7F)};
    const double large = ninfer::perplexity::top_k_kl_divergence(p, saturated);
    failures += require(std::isfinite(large) && large > 1.0, "top-K divergence floors the tail");

    ninfer::perplexity::DistributionAggregate d;
    for (const float value : {4.0F, 1.0F, 3.0F, 2.0F}) { d.add(value, value < 2.5F, -1.0F, -0.5F); }
    failures += require(d.positions() == 4, "distribution aggregate counts positions");
    failures += require(std::abs(d.mean_kl() - 2.5) < 1e-12, "distribution aggregate mean");
    failures += require(d.kl_quantile(0.5) == 2.0 && d.kl_quantile(0.99) == 4.0 &&
                            d.kl_quantile(0.0) == 1.0,
                        "distribution aggregate nearest-rank quantiles");
    failures += require(std::abs(d.top1_agreement() - 0.5) < 1e-12, "top-1 agreement rate");
    failures += require(std::abs(d.mean_evaluated_nll() - 1.0) < 1e-12 &&
                            std::abs(d.mean_reference_nll() - 0.5) < 1e-12,
                        "distribution aggregate NLL means");

    std::cout << (failures == 0 ? "OK" : "FAIL") << " perplexity_evaluation\n";
    return failures == 0 ? 0 : 1;
}
