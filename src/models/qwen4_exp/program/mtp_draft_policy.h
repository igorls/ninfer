#pragma once

#include <algorithm>
#include <cstdint>

namespace ninfer::models::qwen4_exp {

// Adaptive MTP draft count for one request.
//
// The per-position acceptance is modelled as a constant conditional probability p, so k drafts
// yield E(k) = sum_{j=0..k} p^j committed tokens per round. A round with k drafts costs
// c(k) = kMtpRoundBase + kMtpCostPerDraft * k ordinary decode steps: one proposal step plus one
// wider verification column per draft, over a fixed teacher/accept overhead. Each round takes
// the k in [1, maximum] with the highest E(k) / c(k).
//
// The constants were fitted on an RTX PRO 6000 (Colab G4, 2026-10-06): fixed-K rounds of
// iterative code editing at 93-99% acceptance cost 1.42 / 2.06 / 2.82 ordinary steps at
// K = 1 / 3 / 5.
inline constexpr double kMtpRoundBase    = 1.07;
inline constexpr double kMtpCostPerDraft = 0.35;
// Evidence decays per round (about 20 rounds of memory), so the count follows the content.
inline constexpr double kMtpEvidenceDecay = 0.95;
// A request starts from this conditional acceptance, weighted as this many observed drafts.
inline constexpr double kMtpPriorAcceptance = 0.85;
inline constexpr double kMtpPriorWeight     = 4.0;

struct MtpDraftEstimate {
    double accepted = 0.0; // decayed accepted drafts
    double trials   = 0.0; // decayed drafts whose outcome was observed
};

[[nodiscard]] inline double mtp_conditional_acceptance(const MtpDraftEstimate& estimate) noexcept {
    return (estimate.accepted + kMtpPriorAcceptance * kMtpPriorWeight) /
           (estimate.trials + kMtpPriorWeight);
}

[[nodiscard]] inline std::uint32_t choose_mtp_drafts(const MtpDraftEstimate& estimate,
                                                     std::uint32_t maximum) noexcept {
    const double p        = std::clamp(mtp_conditional_acceptance(estimate), 0.0, 1.0);
    std::uint32_t best    = 1;
    double best_rate      = 0.0;
    double expected       = 1.0;
    double reach          = 1.0;
    for (std::uint32_t k = 1; k <= maximum; ++k) {
        reach *= p;
        expected += reach;
        const double rate = expected / (kMtpRoundBase + kMtpCostPerDraft * k);
        if (rate > best_rate) {
            best_rate = rate;
            best      = k;
        }
    }
    return best;
}

// One verified round: `drafted` drafts were proposed and the first `accepted` matched. Under
// the geometric model the accepted drafts are successes, and the first rejected draft (when
// there is one) is the only observed failure; drafts after it carry no evidence.
inline void observe_mtp_round(MtpDraftEstimate& estimate, std::uint32_t drafted,
                              std::uint32_t accepted) noexcept {
    if (drafted == 0) { return; }
    const double successes = std::min(accepted, drafted);
    const double failures  = accepted < drafted ? 1.0 : 0.0;
    estimate.accepted      = kMtpEvidenceDecay * estimate.accepted + successes;
    estimate.trials        = kMtpEvidenceDecay * estimate.trials + successes + failures;
}

} // namespace ninfer::models::qwen4_exp
