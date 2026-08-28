#pragma once

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct Nvfp4ExpertMatrixView {
    const std::byte* codes;
    const std::byte* scales;
    const float* weight_scale_divisor;
    float input_scale_divisor;
    std::int32_t rows;
    std::int32_t columns;
};

struct Nvfp4ExpertBankView {
    const std::byte* codes;
    const std::byte* scales;
    const float* weight_scale_divisors;
    std::int32_t experts;
    std::int32_t rows;
    std::int32_t columns;
    std::uint64_t code_bytes_per_expert;
    std::uint64_t scale_bytes_per_expert;

    [[nodiscard]] Nvfp4ExpertMatrixView expert(std::int32_t index) const;
};

[[nodiscard]] Nvfp4ExpertBankView
make_nvfp4_expert_bank_view(const void* payload, std::uint64_t payload_bytes, std::int32_t experts,
                            std::int32_t rows, std::int32_t columns);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
