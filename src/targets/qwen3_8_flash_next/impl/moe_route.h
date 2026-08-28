#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// Selects deterministic top-10 routes from the exact 512-way full softmax. scores additionally
// carries the independent shared-expert gate in row 512.
void flash_next_route_scores(const Tensor& scores, Tensor& ids, Tensor& alpha, Tensor& shared_scale,
                             cudaStream_t stream);

void flash_next_route(const Tensor& input, const Weight& router, const Weight& shared_gate,
                      Tensor& score_workspace, Tensor& ids, Tensor& alpha, Tensor& shared_scale,
                      cudaStream_t stream);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
