#pragma once

#include "core/tensor.h"
#include "targets/qwen3_8_flash_next/impl/hyper_workspace.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"

#include <cuda_runtime.h>

namespace ninfer::targets::qwen3_8_flash_next::detail {

void flash_next_hyper_prepare_launch(const Tensor& hidden, const HyperConnectionWeights& weights,
                                     FlashNextHyperWorkspace& scratch, Tensor& block_input,
                                     cudaStream_t stream);
void flash_next_hyper_mix_launch(const Tensor& hidden, const HyperMixerWeights& weights,
                                 FlashNextHyperWorkspace& scratch, Tensor& block_input,
                                 cudaStream_t stream);
void flash_next_hyper_inject_launch(const Tensor& block_output, const Tensor& injection,
                                    Tensor& hidden, cudaStream_t stream);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
