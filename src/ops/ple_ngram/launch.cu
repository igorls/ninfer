#include "ops/ple_ngram/launch.h"

#include "core/device.h"
#include "ops/ple_ngram/kernel.cuh"

namespace ninfer::ops::detail {
namespace {

const __nv_bfloat16* bf16(const void* pointer) {
    return static_cast<const __nv_bfloat16*>(pointer);
}

__nv_bfloat16* bf16(void* pointer) { return static_cast<__nv_bfloat16*>(pointer); }

} // namespace

void ple_decode_launch(const void* codes, const void* scales, void* out, std::int64_t rows,
                       cudaStream_t stream) {
    // Codes, scales and output are row-major with no padding, so group g of row r is the
    // (r * 10 + g)-th group of every plane.
    const std::int64_t groups = rows * (kPleRowWidth / 16);
    ple_decode_kernel<<<static_cast<unsigned>((groups + 255) / 256), 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(codes), static_cast<const __half*>(scales),
        static_cast<__nv_bfloat16*>(out), groups);
    CUDA_CHECK(cudaGetLastError());
}

void ple_gate_launch(const void* hidden, const void* key, const void* value,
                     const PleWeights& weights, const PleScratch& scratch, std::int32_t columns,
                     cudaStream_t stream) {
    // grid.y bounds one launch; wider calls advance the column origin.
    constexpr std::int32_t kMaximumColumns = 65535;
    for (std::int32_t begin = 0; begin < columns; begin += kMaximumColumns) {
        const std::int32_t count = columns - begin < kMaximumColumns ? columns - begin
                                                                     : kMaximumColumns;
        const auto wide   = static_cast<std::int64_t>(begin) * kPleWidth;
        const auto narrow = static_cast<std::int64_t>(begin) * kPleStreamWidth;
        ple_gate_norm_kernel<<<dim3(kPleStreams, count), kPleGateThreads, 0, stream>>>(
            bf16(hidden) + wide, bf16(key) + wide, bf16(value) + narrow, bf16(weights.query_norm),
            bf16(weights.key_norm), bf16(weights.conv_norm),
            scratch.gates + static_cast<std::int64_t>(begin) * kPleStreams,
            bf16(scratch.normalized) + wide);
        CUDA_CHECK(cudaGetLastError());
    }
}

void ple_conv_launch(const PleWeights& weights, const PleScratch& scratch, const void* value,
                     const void* state_in,
                     void* state_out, void* out, std::int32_t columns, cudaStream_t stream) {
    ple_conv_kernel<<<static_cast<unsigned>(columns) * kPleConvBlocks, kPleConvThreads, 0,
                      stream>>>(scratch.gates, bf16(value), bf16(scratch.normalized),
                                bf16(weights.conv_weight), bf16(state_in), bf16(out));
    CUDA_CHECK(cudaGetLastError());
    ple_state_kernel<<<kPleConvBlocks, kPleConvThreads, 0, stream>>>(
        bf16(scratch.normalized), bf16(state_in), bf16(state_out), columns);
    CUDA_CHECK(cudaGetLastError());
}

void ple_snapshot_launch(const PleWeights& weights, const PleScratch& scratch, const void* value,
                         void* states,
                         const std::int32_t* valid_columns, const std::int32_t* initial_slots,
                         const std::int32_t* snapshot_base_slots, void* out, std::int32_t width,
                         std::int32_t batch, cudaStream_t stream) {
    ple_snapshot_kernel<<<dim3(kPleConvBlocks, batch), kPleConvThreads, 0, stream>>>(
        scratch.gates, bf16(value), bf16(scratch.normalized), bf16(weights.conv_weight),
        bf16(states),
        valid_columns, initial_slots, snapshot_base_slots, bf16(out), width);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
