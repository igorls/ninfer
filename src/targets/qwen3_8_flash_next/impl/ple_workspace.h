#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct FlashNextPleWorkspace {
    Tensor projected_key;
    Tensor projected_value;
    Tensor gated;
    Tensor normalized_gated;
};

template <class Arena>
FlashNextPleWorkspace allocate_flash_next_ple_workspace(Arena& arena, std::int32_t batch) {
    return {
        .projected_key    = arena.alloc(DType::BF16, {10'240, batch}, 256),
        .projected_value  = arena.alloc(DType::BF16, {2'560, batch}, 256),
        .gated            = arena.alloc(DType::BF16, {10'240, batch}, 256),
        .normalized_gated = arena.alloc(DType::BF16, {10'240, batch}, 256),
    };
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
