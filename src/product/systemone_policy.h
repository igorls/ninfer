#pragma once

#include <cstdint>

namespace ninfer::product {

// Shared defaults for TypeSafe System One and its in-process rerank consumer.
struct SystemOnePolicy {
    enum class Readout : std::uint8_t { Single, Averaged };
    Readout readout                = Readout::Single;
    std::uint32_t reasoning_budget = 0;
    double escalate_entropy        = 0.66;
    double answer_weight           = 0.7;

    [[nodiscard]] bool active() const noexcept {
        return readout == Readout::Averaged || reasoning_budget != 0;
    }
};

} // namespace ninfer::product
