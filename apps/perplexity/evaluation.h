#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::perplexity {

struct WindowPlan {
    std::size_t input_begin    = 0;
    std::size_t input_end      = 0;
    std::size_t target_begin   = 0;
    std::size_t target_end     = 0;
    std::uint32_t first_target = 0;
};

[[nodiscard]] std::vector<WindowPlan> plan_windows(std::size_t tokens, std::uint32_t context,
                                                   std::uint32_t stride);

struct ScoreAggregate {
    std::uint64_t scored_tokens = 0;
    double total_nll            = 0.0;

    void add(std::span<const float> logprobs);
    void add(const ScoreAggregate& other) noexcept;
    [[nodiscard]] double mean_nll() const;
    [[nodiscard]] double ppl() const;
};

// KL(P || Q) for one position, restricted to the reference distribution P's top-K tokens plus one
// bucket holding the remaining mass of each distribution. `reference` and `evaluated` are the
// natural log probabilities of the same K tokens under P and Q. Merging the tail into one bucket
// makes this a lower bound on the full-vocabulary divergence (log-sum inequality); it is exact
// when the K tokens carry all of P's mass.
[[nodiscard]] double top_k_kl_divergence(std::span<const float> reference,
                                         std::span<const float> evaluated);

// Distribution agreement of an evaluated model with a reference over scored positions.
struct DistributionAggregate {
    std::vector<float> kl;
    std::uint64_t top1_agreements = 0;
    double evaluated_nll          = 0.0;
    double reference_nll          = 0.0;

    void add(double kl_value, bool top1_agrees, float evaluated_logprob, float reference_logprob);
    void add(const DistributionAggregate& other);
    [[nodiscard]] std::uint64_t positions() const noexcept { return kl.size(); }
    [[nodiscard]] double mean_kl() const;
    // Nearest-rank quantile of the per-position divergences, q in [0,1].
    [[nodiscard]] double kl_quantile(double q) const;
    [[nodiscard]] double top1_agreement() const;
    [[nodiscard]] double mean_evaluated_nll() const;
    [[nodiscard]] double mean_reference_nll() const;
};

} // namespace ninfer::perplexity
