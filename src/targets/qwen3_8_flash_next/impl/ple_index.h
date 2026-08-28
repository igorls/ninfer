#pragma once

#include <array>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct PleIndexMetadata {
    std::array<std::int64_t, 3> multipliers;
    std::array<std::int64_t, 16> head_offsets;
    std::array<std::int64_t, 16> head_vocab_sizes;
};

class PleTokenHistory {
public:
    explicit PleTokenHistory(std::int64_t eos_token) noexcept;

    void commit(std::int64_t token) noexcept;

    std::int64_t previous_token() const noexcept { return previous_; }

    std::int64_t second_previous_token() const noexcept { return second_previous_; }

private:
    std::int64_t eos_token_;
    std::int64_t previous_;
    std::int64_t second_previous_;
};

// Computes the eight bigram and eight trigram embedding rows for one token.
// The history is deliberately not mutated: speculative callers commit only
// tokens accepted by the output transaction.
[[nodiscard]] std::array<std::int64_t, 16>
ple_indices(const PleIndexMetadata& metadata, const PleTokenHistory& history, std::int64_t token);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
