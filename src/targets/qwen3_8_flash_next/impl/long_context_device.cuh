#pragma once

#include "targets/qwen3_8_flash_next/impl/long_context.h"

#include <cuda_bf16.h>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// Default path is the historical expression: theta^(-2 * pair / 64) via expf/logf.
// YaRN reads the host table from flash_next_rope_scaling and scales cos/sin by the
// published attention factor. The branch is uniform for a launch.
__device__ __forceinline__ float
flash_next_device_rope_frequency(int pair, const FlashNextRopeScaling& rope) {
    if (rope.yarn == 0) {
        return expf((-2.0F * static_cast<float>(pair) / 64.0F) * logf(1.0e7F));
    }
    return rope.inv_freq[pair];
}

__device__ __forceinline__ void flash_next_device_store_partial_mrope(
    const __nv_bfloat16* normalized, const std::int32_t* positions, __nv_bfloat16* output, int dim,
    const FlashNextRopeScaling& rope) {
    if (dim < 32) {
        const float angle = static_cast<float>(positions[dim % 3]) *
                            flash_next_device_rope_frequency(dim, rope);
        float sine   = 0.0F;
        float cosine = 0.0F;
        sincosf(angle, &sine, &cosine);
        if (rope.yarn != 0) {
            sine *= rope.attention_factor;
            cosine *= rope.attention_factor;
        }
        const float first  = __bfloat162float(normalized[dim]);
        const float second = __bfloat162float(normalized[dim + 32]);
        output[dim]        = __float2bfloat16_rn(first * cosine - second * sine);
        output[dim + 32]   = __float2bfloat16_rn(second * cosine + first * sine);
    } else if (dim >= 64) {
        output[dim] = normalized[dim];
    }
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
