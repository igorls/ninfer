#include "evaluation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::perplexity {

std::vector<WindowPlan> plan_windows(std::size_t tokens, std::uint32_t context,
                                     std::uint32_t stride) {
    if (tokens < 2) { throw std::invalid_argument("perplexity stream must contain two tokens"); }
    if (context < 2 || stride == 0 || stride >= context) {
        throw std::invalid_argument("perplexity requires context>=2 and 1<=stride<context");
    }

    std::vector<WindowPlan> windows;
    std::size_t previous_end = std::min<std::size_t>(tokens, context);
    windows.push_back(WindowPlan{.input_begin  = 0,
                                 .input_end    = previous_end,
                                 .target_begin = 1,
                                 .target_end   = previous_end,
                                 .first_target = 1});
    while (previous_end < tokens) {
        const std::size_t end          = std::min(tokens, previous_end + stride);
        const std::size_t begin        = end > context ? end - context : 0;
        const std::size_t local_target = previous_end - begin;
        if (local_target == 0 || local_target >= end - begin ||
            local_target > std::numeric_limits<std::uint32_t>::max()) {
            throw std::logic_error("perplexity window has an invalid target suffix");
        }
        windows.push_back(WindowPlan{
            .input_begin  = begin,
            .input_end    = end,
            .target_begin = previous_end,
            .target_end   = end,
            .first_target = static_cast<std::uint32_t>(local_target),
        });
        previous_end = end;
    }
    return windows;
}

void ScoreAggregate::add(std::span<const float> logprobs) {
    for (const float logprob : logprobs) {
        if (!std::isfinite(logprob)) {
            throw std::runtime_error("causal scoring returned a non-finite logprob");
        }
        total_nll -= static_cast<double>(logprob);
    }
    if (logprobs.size() > std::numeric_limits<std::uint64_t>::max() - scored_tokens) {
        throw std::overflow_error("perplexity scored-token count overflowed");
    }
    scored_tokens += static_cast<std::uint64_t>(logprobs.size());
}

void ScoreAggregate::add(const ScoreAggregate& other) noexcept {
    scored_tokens += other.scored_tokens;
    total_nll += other.total_nll;
}

double ScoreAggregate::mean_nll() const {
    if (scored_tokens == 0) { throw std::logic_error("perplexity aggregate is empty"); }
    return total_nll / static_cast<double>(scored_tokens);
}

double ScoreAggregate::ppl() const { return std::exp(mean_nll()); }

double top_k_kl_divergence(std::span<const float> reference, std::span<const float> evaluated) {
    if (reference.empty() || reference.size() != evaluated.size()) {
        throw std::invalid_argument("top-K divergence needs equal nonempty distributions");
    }
    // Smallest tail probability the evaluated side is credited with, so a model that puts
    // (numerically) all of its mass on the K tokens yields a large finite divergence.
    constexpr double kTailFloor = 1.0e-10;
    double divergence = 0.0;
    double p_mass     = 0.0;
    double q_mass     = 0.0;
    for (std::size_t k = 0; k < reference.size(); ++k) {
        const double p_log = reference[k];
        const double q_log = evaluated[k];
        if (!std::isfinite(p_log) || std::isnan(q_log)) {
            throw std::runtime_error("top-K divergence received a non-finite log probability");
        }
        const double p = std::exp(p_log);
        p_mass += p;
        q_mass += std::exp(q_log);
        if (p > 0.0) { divergence += p * (p_log - std::max(q_log, std::log(kTailFloor))); }
    }
    const double p_tail = std::max(0.0, 1.0 - p_mass);
    const double q_tail = std::max(kTailFloor, 1.0 - q_mass);
    if (p_tail > 0.0) { divergence += p_tail * (std::log(p_tail) - std::log(q_tail)); }
    // Rounding of the two mass sums can leave a tiny negative value for identical inputs.
    return std::max(0.0, divergence);
}

void DistributionAggregate::add(double kl_value, bool top1_agrees, float evaluated_logprob,
                                float reference_logprob) {
    if (!std::isfinite(kl_value) || !std::isfinite(evaluated_logprob) ||
        !std::isfinite(reference_logprob)) {
        throw std::runtime_error("distribution comparison received a non-finite value");
    }
    kl.push_back(static_cast<float>(kl_value));
    top1_agreements += top1_agrees ? 1U : 0U;
    evaluated_nll -= static_cast<double>(evaluated_logprob);
    reference_nll -= static_cast<double>(reference_logprob);
}

void DistributionAggregate::add(const DistributionAggregate& other) {
    kl.insert(kl.end(), other.kl.begin(), other.kl.end());
    top1_agreements += other.top1_agreements;
    evaluated_nll += other.evaluated_nll;
    reference_nll += other.reference_nll;
}

namespace {
double require_positions(std::size_t count) {
    if (count == 0) { throw std::logic_error("distribution aggregate is empty"); }
    return static_cast<double>(count);
}
} // namespace

double DistributionAggregate::mean_kl() const {
    double sum = 0.0;
    for (const float value : kl) { sum += value; }
    return sum / require_positions(kl.size());
}

double DistributionAggregate::kl_quantile(double q) const {
    require_positions(kl.size());
    if (!(q >= 0.0 && q <= 1.0)) { throw std::invalid_argument("quantile must be in [0,1]"); }
    std::vector<float> sorted(kl);
    std::sort(sorted.begin(), sorted.end());
    const auto rank = static_cast<std::size_t>(std::ceil(q * static_cast<double>(sorted.size())));
    return sorted[std::max<std::size_t>(rank, 1) - 1];
}

double DistributionAggregate::top1_agreement() const {
    return static_cast<double>(top1_agreements) / require_positions(kl.size());
}

double DistributionAggregate::mean_evaluated_nll() const {
    return evaluated_nll / require_positions(kl.size());
}

double DistributionAggregate::mean_reference_nll() const {
    return reference_nll / require_positions(kl.size());
}

} // namespace ninfer::perplexity
