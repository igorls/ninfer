#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::targets::qwen3_8_flash_next::detail {

void flash_next_mtp_stem_combine_and_repeat_launch(const Tensor& emb_proj, const Tensor& hid_proj,
                                                  Tensor* trunk_sum, Tensor& mtp_hyper_hidden,
                                                  cudaStream_t stream);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
