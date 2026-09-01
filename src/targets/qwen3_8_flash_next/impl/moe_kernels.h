#pragma once

#include "targets/qwen3_8_flash_next/impl/model_view.h"
#include "targets/qwen3_8_flash_next/impl/moe_workspace.h"

#include <cuda_runtime.h>

namespace ninfer::targets::qwen3_8_flash_next::detail {

void flash_next_moe_kernels_launch(const Tensor& input, const MoeWeights& weights,
                                   const FlashNextMoeWorkspace& workspace, Tensor& output,
                                   cudaStream_t stream);

void flash_next_moe_bf16_kernels_launch(const Tensor& input, const MoeBf16Weights& weights,
                                        const FlashNextMoeWorkspace& workspace, Tensor& output,
                                        cudaStream_t stream);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
