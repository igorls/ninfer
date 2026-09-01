#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

inline constexpr std::int32_t kOutputHeadRows    = 248'320;
inline constexpr std::int32_t kOutputHeadColumns = 2'560;

// E4M3FN max finite is 448. Scale is per-row amax/448. Codes use round-to-nearest-even
// saturate-to-finite conversion (__nv_cvt_float2_to_fp8x2, __NV_SATFINITE, __NV_E4M3).
[[nodiscard]] std::size_t flash_next_fp8_output_head_payload_bytes();

void quantize_bf16_output_head_to_fp8_e4m3_row_f32s(const Weight& bf16_head, DeviceBuffer& payload,
                                                    Weight& fp8_head, cudaStream_t stream);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
