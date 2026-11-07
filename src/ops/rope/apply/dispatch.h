#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

inline constexpr int kYarnR64Pairs = 32;

// Host-resolved YaRN angular frequency of every R64 rotary pair and the output magnitude.
struct YarnR64Table {
    double frequency[kYarnR64Pairs];
    float magnitude;
};

void rope_dispatch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                   CudaExecutionView execution);

void rope_dispatch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                   CudaExecutionView execution);

// Text D256/R64, positions [T] or [T,3]; `k` may be null for the single-tensor form.
void rope_yarn_dispatch(const Tensor& positions, const YarnR64Table& table, Tensor& q, Tensor* k,
                        CudaExecutionView execution);

} // namespace ninfer::ops::detail
