#include "targets/qwen3_8_flash_next/impl/load/quantize_output_head.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

constexpr int kThreads = 256;

__global__ __launch_bounds__(kThreads, 2) void quantize_row_kernel(
    const __nv_bfloat16* __restrict__ input, std::uint8_t* __restrict__ codes,
    float* __restrict__ scales) {
    static_assert((kOutputHeadColumns % (kThreads * 2)) == 0);
    constexpr int pairs_per_row    = kOutputHeadColumns / 2;
    constexpr int pairs_per_thread = pairs_per_row / kThreads;
    constexpr int warps            = kThreads / 32;
    __shared__ float warp_maxima[warps];
    __shared__ float row_scale;

    const int row  = static_cast<int>(blockIdx.x);
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const auto* input_pairs =
        reinterpret_cast<const std::uint32_t*>(input + static_cast<std::int64_t>(row) * kOutputHeadColumns);
    auto* output_pairs =
        reinterpret_cast<std::uint16_t*>(codes + static_cast<std::int64_t>(row) * kOutputHeadColumns);

    float2 values[pairs_per_thread];
    float maximum = 0.0F;
#pragma unroll
    for (int item = 0; item < pairs_per_thread; ++item) {
        const int pair = tid + item * kThreads;
        values[item]   = ops::bf16x2_bits_to_float2(input_pairs[pair]);
        maximum        = fmaxf(maximum, fabsf(values[item].x));
        maximum        = fmaxf(maximum, fabsf(values[item].y));
    }
    maximum = ops::warp_max(maximum);
    if (lane == 0) { warp_maxima[warp] = maximum; }
    __syncthreads();
    if (warp == 0) {
        maximum = lane < warps ? warp_maxima[lane] : 0.0F;
        maximum = ops::warp_max(maximum);
        if (lane == 0) { row_scale = maximum > 0.0F ? maximum / 448.0F : 0.0F; }
    }
    __syncthreads();

    const float scale   = row_scale;
    const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
#pragma unroll
    for (int item = 0; item < pairs_per_thread; ++item) {
        const int pair      = tid + item * kThreads;
        const float2 scaled = make_float2(values[item].x * inverse, values[item].y * inverse);
        output_pairs[pair]  = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
    }
    if (tid == 0) { scales[row] = scale; }
}

Weight make_fp8_view(void* payload, std::size_t payload_bytes) {
    const std::uint64_t codes        = static_cast<std::uint64_t>(kOutputHeadRows) * kOutputHeadColumns;
    const std::uint64_t scale_offset = (codes + 255U) & ~std::uint64_t{255U};
    auto* bytes                      = static_cast<std::byte*>(payload);
    const std::int64_t scale_stride  = static_cast<std::int64_t>(kOutputHeadRows) * 4;
    Weight out{};
    out.payload         = payload;
    out.payload_bytes   = payload_bytes;
    out.qdata           = payload;
    out.scales          = bytes + scale_offset;
    out.qtype           = QType::FP8_E4M3FN_ROW_F32S;
    out.layout          = QuantLayout::RowScale;
    out.scale_dtype     = DType::FP32;
    out.n               = kOutputHeadRows;
    out.k               = kOutputHeadColumns;
    out.group           = kOutputHeadColumns;
    out.group_size      = static_cast<std::uint32_t>(kOutputHeadColumns);
    out.ndim            = 2;
    out.shape[0]        = kOutputHeadRows;
    out.shape[1]        = kOutputHeadColumns;
    out.padded_shape[0] = kOutputHeadRows;
    out.padded_shape[1] = kOutputHeadColumns;
    out.scale_ne[0]     = kOutputHeadRows;
    out.scale_nb[0]     = 4;
    out.scale_nb[1]     = scale_stride;
    out.scale_nb[2]     = scale_stride;
    out.scale_nb[3]     = scale_stride;
    return out;
}

} // namespace

std::size_t flash_next_fp8_output_head_payload_bytes() {
    const std::uint64_t codes        = static_cast<std::uint64_t>(kOutputHeadRows) * kOutputHeadColumns;
    const std::uint64_t scale_offset = (codes + 255U) & ~std::uint64_t{255U};
    return static_cast<std::size_t>(scale_offset + static_cast<std::uint64_t>(kOutputHeadRows) * 4);
}

void quantize_bf16_output_head_to_fp8_e4m3_row_f32s(const Weight& bf16_head, DeviceBuffer& payload,
                                                    Weight& fp8_head, cudaStream_t stream) {
    if (bf16_head.qtype != QType::BF16_CTRL || bf16_head.n != kOutputHeadRows ||
        bf16_head.k != kOutputHeadColumns || bf16_head.qdata == nullptr) {
        throw std::invalid_argument("Flash-Next FP8 output-head quantize received an invalid BF16 view");
    }
    const std::size_t bytes = flash_next_fp8_output_head_payload_bytes();
    if (payload.p == nullptr || payload.bytes < bytes) {
        throw std::invalid_argument("Flash-Next FP8 output-head payload is too small");
    }
    const std::uint64_t codes        = static_cast<std::uint64_t>(kOutputHeadRows) * kOutputHeadColumns;
    const std::uint64_t scale_offset = (codes + 255U) & ~std::uint64_t{255U};
    auto* code_ptr                   = static_cast<std::uint8_t*>(payload.p);
    auto* scale_ptr = reinterpret_cast<float*>(static_cast<std::byte*>(payload.p) + scale_offset);
    quantize_row_kernel<<<kOutputHeadRows, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bf16_head.qdata), code_ptr, scale_ptr);
    CUDA_CHECK(cudaGetLastError());
    fp8_head = make_fp8_view(payload.p, payload.bytes);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
