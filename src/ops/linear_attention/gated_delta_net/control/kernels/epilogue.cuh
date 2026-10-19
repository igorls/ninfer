#include "ops/common/numeric/math.cuh"
#pragma once
#include "ops/common/numeric/convert.cuh"
#include <cuda_bf16.h>

namespace ninfer::ops::detail::gdn_control {
__device__ __forceinline__ float control_value(float v) { return v; }
__device__ __forceinline__ float control_value(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float decay(float value, float a_log, float bias) {
    return -expf(a_log) * common::softplus(value + bias);
}
} // namespace ninfer::ops::detail::gdn_control
