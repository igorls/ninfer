#include "models/qwen4_exp/program/mtp_draft_policy.h"

#include <cstdint>
#include <iostream>

// Behavior of the adaptive MTP draft policy, independent of its fitted constants: the count stays
// in [1, maximum], never shrinks as acceptance rises, follows a change in the content within a few
// dozen rounds, and ignores rounds that proposed nothing.
namespace {
namespace q4 = ninfer::models::qwen4_exp;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

q4::MtpDraftEstimate after(std::uint32_t rounds, std::uint32_t drafted, std::uint32_t accepted,
                           q4::MtpDraftEstimate estimate = {}) {
    for (std::uint32_t i = 0; i < rounds; ++i) { q4::observe_mtp_round(estimate, drafted, accepted); }
    return estimate;
}

} // namespace

int main() {
    for (std::uint32_t maximum = 1; maximum <= 5; ++maximum) {
        for (std::uint32_t accepted = 0; accepted <= maximum; ++accepted) {
            const auto chosen = q4::choose_mtp_drafts(after(40, maximum, accepted), maximum);
            check(chosen >= 1 && chosen <= maximum, "draft count leaves [1, maximum]");
        }
    }

    // Monotone in acceptance: more accepted drafts per round never selects fewer drafts.
    std::uint32_t previous = 0;
    for (std::uint32_t accepted = 0; accepted <= 5; ++accepted) {
        const auto chosen = q4::choose_mtp_drafts(after(60, 5, accepted), 5);
        check(chosen >= previous, "draft count shrinks as acceptance rises");
        previous = chosen;
    }

    // Sustained full acceptance reaches the maximum; sustained rejection falls to one draft.
    check(q4::choose_mtp_drafts(after(60, 5, 5), 5) == 5, "full acceptance does not reach maximum");
    check(q4::choose_mtp_drafts(after(60, 5, 0), 5) == 1, "rejection does not fall to one draft");

    // A change in the content is followed within 40 rounds, in both directions.
    const auto copying = after(200, 5, 5);
    check(q4::choose_mtp_drafts(after(40, 5, 0, copying), 5) == 1,
          "policy does not follow a drop in acceptance");
    const auto prose = after(200, 1, 0);
    check(q4::choose_mtp_drafts(after(40, 1, 1, prose), 5) > 1,
          "policy does not follow a rise in acceptance observed at one draft");

    // A round that proposed nothing (logprob fallback, budget or context edge) carries no evidence.
    const auto before = after(10, 3, 2);
    auto unchanged    = before;
    q4::observe_mtp_round(unchanged, 0, 0);
    check(unchanged.accepted == before.accepted && unchanged.trials == before.trials,
          "a round without drafts changed the estimate");

    if (failures == 0) { std::cout << "PASS: adaptive MTP draft policy\n"; }
    return failures == 0 ? 0 : 1;
}
