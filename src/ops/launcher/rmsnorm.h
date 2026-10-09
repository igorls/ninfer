#pragma once

// ninfer::ops::detail - private launch prototype for rmsnorm.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Weight offset and gate of one RMSNorm contract; the gated forms read z.
enum class RmsNormForm : std::uint8_t {
    Plain,
    UnitOffset,
    SiluGated,
    SigmoidGated,
};

void rmsnorm_launch(const Tensor& x, const Tensor& weight, float eps, RmsNormForm form,
                    const Tensor* z, Tensor& out, std::int32_t multiprocessor_count,
                    cudaStream_t stream);

} // namespace ninfer::ops::detail
