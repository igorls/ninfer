#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// T <= kHyperDecodeTokens takes the CUDA-core decode chain; wider T the tensor-core route.
inline constexpr std::int32_t kHyperDecodeTokens = 8;
inline constexpr int kHyperDownSplits            = 4;

struct HyperScratch {
    void* normalized = nullptr; // BF16 [10240,T]
    void* low_rank   = nullptr; // BF16 [320,T]
    float* partials  = nullptr; // FP32 [kHyperDownSplits,320,T] (tensor-core route)
    void* up         = nullptr; // BF16 [10240,T] (tensor-core route)
    std::size_t partial_bytes = 0;
};

// inject may be null: the mixer form has no injection rows or output.
void hyper_mix_launch(const void* hidden, const void* norm, const void* down, const void* up,
                      const void* inject, void* block_input, float* injection,
                      const HyperScratch& scratch, std::int32_t tokens, cudaStream_t stream);

void hyper_inject_launch(const void* block_output, const float* injection, void* hidden,
                         std::int32_t tokens, cudaStream_t stream);

} // namespace ninfer::ops::detail
