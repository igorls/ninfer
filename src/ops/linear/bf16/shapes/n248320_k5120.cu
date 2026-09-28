#include "ops/linear/bf16/bf16_shapes.h"

// A full BF16 output head over the Qwen3.5 vocabulary (a checkpoint that keeps its source head in
// BF16). The unified schedules are instantiated per K and token tile, not per N, so the
// [14336,5120] routes serve it without separate instances. Every column extent reads the whole
// 2.5 GB matrix once, so decode and verification widths stay memory-bound.
namespace ninfer::ops::detail {

Bf16Launch select_bf16_n248320_k5120(std::int32_t tokens) {
    return select_bf16_n14336_k5120(tokens);
}

} // namespace ninfer::ops::detail
