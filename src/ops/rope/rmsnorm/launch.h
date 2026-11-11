#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void rmsnorm_rope_gated_d256_launch(const Tensor& projected, const Tensor& positions,
                                    const Tensor& q_norm_weight, const Tensor& k_norm_weight,
                                    Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                    std::int32_t tokens, cudaStream_t stream);

} // namespace ninfer::ops::detail
