#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

#include "ops/softmax_attention/selected_block/types.h"

namespace ninfer::ops::detail {

// Visible-token ordinal -> cache position: selected blocks first, then the causal tail.
__device__ __forceinline__ int selected_token(int ordinal, int count, int complete,
                                              const std::int32_t* selected) {
    return ordinal < count * 4 ? selected[ordinal / 4] * 4 + (ordinal & 3)
                               : complete * 4 + ordinal - count * 4;
}

// Page-major row index of (token, kv head) through one block-table row.
__device__ __forceinline__ std::int64_t selected_kv_row(const std::int32_t* table_row, int token,
                                                         int kv_head) {
    const int page = table_row[token / kSelectedPageTokens];
    return (static_cast<std::int64_t>(page) * kSelectedKvHeads + kv_head) * kSelectedPageTokens +
           token % kSelectedPageTokens;
}

__device__ __forceinline__ float fp8_value(std::uint8_t code) {
    __nv_fp8_e4m3 value;
    value.__x = code;
    return static_cast<float>(value);
}

} // namespace ninfer::ops::detail
