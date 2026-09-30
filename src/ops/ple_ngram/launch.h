#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct PleWeights {
    const void* query_norm;
    const void* key_norm;
    const void* conv_norm;
    const void* conv_weight;
};

struct PleScratch {
    float* gates;     // FP32 [4,columns]: sigmoid of each stream gate
    void* normalized; // BF16 [10240,columns]
};

void ple_decode_launch(const void* codes, const void* scales, void* out, std::int64_t rows,
                       cudaStream_t stream);

void ple_gate_launch(const void* hidden, const void* key, const void* value,
                     const PleWeights& weights, const PleScratch& scratch, std::int32_t columns,
                     cudaStream_t stream);

void ple_conv_launch(const PleWeights& weights, const PleScratch& scratch, const void* value,
                     const void* state_in,
                     void* state_out, void* out, std::int32_t columns, cudaStream_t stream);

void ple_snapshot_launch(const PleWeights& weights, const PleScratch& scratch, const void* value,
                         void* states,
                         const std::int32_t* valid_columns, const std::int32_t* initial_slots,
                         const std::int32_t* snapshot_base_slots, void* out, std::int32_t width,
                         std::int32_t batch, cudaStream_t stream);

} // namespace ninfer::ops::detail
