#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer::ops::detail {

struct Fp8WeightGeometry {
    std::uint64_t code_plane_bytes;
    std::uint64_t scale_plane_offset;
    std::uint64_t scale_plane_bytes;
    std::uint64_t required_payload_bytes;
    std::uint64_t scale_word_bytes;
};

// Row-scaled FP8 with BF16 multipliers (FP8_E4M3FN_ROW_BF16 / RowScale), the form every fused
// FP8 consumer admits.
Fp8WeightGeometry validate_fp8_weight(const Weight& weight, const char* operation);
// Either registered row-scaled FP8 form: BF16 (RowScale) or FP32 (FP8_E4M3FN_ROW_FP32 /
// RowScaleFp32) multipliers.
Fp8WeightGeometry validate_fp8_row_weight(const Weight& weight, const char* operation);

} // namespace ninfer::ops::detail
