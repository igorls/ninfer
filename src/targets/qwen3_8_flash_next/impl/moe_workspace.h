#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

struct FlashNextMoeWorkspace {
    Tensor scores;
    Tensor ids;
    Tensor alpha;
    Tensor shared_scale;
    Tensor activations;
};

template <class Arena>
FlashNextMoeWorkspace allocate_flash_next_moe_workspace(Arena& arena, std::int32_t tokens) {
    FlashNextMoeWorkspace out;
    out.scores       = arena.alloc(DType::FP32, {513, tokens}, 256);
    out.ids          = arena.alloc(DType::I32, {10, tokens}, 16);
    out.alpha        = arena.alloc(DType::FP32, {10, tokens}, 16);
    out.shared_scale = arena.alloc(DType::FP32, {tokens}, 16);
    // Ten routed SwiGLU paths followed by the always-on shared path.
    out.activations = arena.alloc(DType::BF16, {640, 11, tokens}, 256);
    return out;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
