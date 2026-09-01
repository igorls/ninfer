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
    // Prefill grouping & amortized GEMM workspace (tokens > 8)
    Tensor expert_counts;
    Tensor expert_offsets;
    Tensor active_expert_ids;
    Tensor active_count;
    Tensor grouped_tokens;
    Tensor grouped_paths;
    Tensor grouped_experts;
    Tensor token_to_pos;
    Tensor staged_down;
    Tensor down_intermediate;
    Tensor task_counter;
    // Prefill NVFP4 MMA activation quantization buffers (tokens > 8)
    Tensor act_codes;
    Tensor act_scales;
    Tensor down_act_codes;
    Tensor down_act_scales;
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

    if (tokens > 8) {
        out.expert_counts     = arena.alloc(DType::I32, {512}, 16);
        out.expert_offsets    = arena.alloc(DType::I32, {513}, 16);
        out.active_expert_ids = arena.alloc(DType::I32, {512}, 16);
        out.active_count      = arena.alloc(DType::I32, {1}, 16);
        out.grouped_tokens    = arena.alloc(DType::I32, {10 * tokens}, 16);
        out.grouped_paths     = arena.alloc(DType::I32, {10 * tokens}, 16);
        out.grouped_experts   = arena.alloc(DType::I32, {10 * tokens}, 16);
        out.token_to_pos      = arena.alloc(DType::I32, {10 * tokens}, 16);
        if (tokens >= 512) {
            out.staged_down   = arena.alloc(DType::BF16, {2'560, 10 * tokens}, 256);
        } else {
            out.down_intermediate = arena.alloc(DType::FP32, {2'560, 10, tokens}, 256);
        }
        out.task_counter      = arena.alloc(DType::I32, {4}, 16);
        out.act_codes         = arena.alloc(DType::U8, {1'280, tokens}, 256);
        out.act_scales        = arena.alloc(DType::U8, {160, tokens}, 256);
        out.down_act_codes    = arena.alloc(DType::U8, {320, 11 * tokens}, 256);
        out.down_act_scales   = arena.alloc(DType::U8, {40, 11 * tokens}, 256);
    }
    return out;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
