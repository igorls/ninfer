#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct FlashNextHyperWorkspace {
    Tensor normalized;
    Tensor low_rank;
    Tensor injection;
    Tensor up_gemm;
};

template <class Arena>
FlashNextHyperWorkspace allocate_flash_next_hyper_workspace(Arena& arena, std::int32_t tokens) {
    return {
        .normalized = arena.alloc(DType::BF16, {10'240, tokens}, 256),
        .low_rank   = arena.alloc(DType::BF16, {320, tokens}, 256),
        .injection  = arena.alloc(DType::FP32, {4, tokens}, 16),
        .up_gemm    = arena.alloc(DType::BF16, {10'240, tokens}, 256),
    };
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
