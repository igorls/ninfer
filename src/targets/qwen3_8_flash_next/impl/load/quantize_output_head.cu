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

__global__ void gather_rows_kernel(
    const __nv_bfloat16* __restrict__ src,
    const std::int32_t* __restrict__ token_ids,
    __nv_bfloat16* __restrict__ dst,
    int rows, int cols) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const int src_token = token_ids[row];
    const auto* src_row = reinterpret_cast<const uint4*>(src + static_cast<std::int64_t>(src_token) * cols);
    auto* dst_row = reinterpret_cast<uint4*>(dst + static_cast<std::int64_t>(row) * cols);
    const int vectors = cols / 8; // 2560 / 8 = 320 uint4s
    for (int i = static_cast<int>(threadIdx.x); i < vectors; i += static_cast<int>(blockDim.x)) {
        dst_row[i] = src_row[i];
    }
}

Weight make_fp8_view_sized(void* payload, std::size_t payload_bytes, std::int32_t rows, std::int32_t cols) {
    const std::uint64_t codes        = static_cast<std::uint64_t>(rows) * cols;
    const std::uint64_t scale_offset = (codes + 255U) & ~std::uint64_t{255U};
    auto* bytes                      = static_cast<std::byte*>(payload);
    const std::int64_t scale_stride  = static_cast<std::int64_t>(rows) * 4;
    Weight out{};
    out.payload         = payload;
    out.payload_bytes   = payload_bytes;
    out.qdata           = payload;
    out.scales          = bytes + scale_offset;
    out.qtype           = QType::FP8_E4M3FN_ROW_F32S;
    out.layout          = QuantLayout::RowScale;
    out.scale_dtype     = DType::FP32;
    out.n               = rows;
    out.k               = cols;
    out.group           = cols;
    out.group_size      = static_cast<std::uint32_t>(cols);
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = cols;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = cols;
    out.scale_ne[0]     = rows;
    out.scale_nb[0]     = 4;
    out.scale_nb[1]     = scale_stride;
    out.scale_nb[2]     = scale_stride;
    out.scale_nb[3]     = scale_stride;
    return out;
}

Weight make_fp8_view(void* payload, std::size_t payload_bytes) {
    return make_fp8_view_sized(payload, payload_bytes, kOutputHeadRows, kOutputHeadColumns);
}

} // namespace

std::size_t flash_next_fp8_output_head_payload_bytes() {
    return flash_next_fp8_head_payload_bytes(kOutputHeadRows, kOutputHeadColumns);
}

std::size_t flash_next_fp8_head_payload_bytes(std::int32_t rows, std::int32_t cols) {
    const std::uint64_t codes        = static_cast<std::uint64_t>(rows) * cols;
    const std::uint64_t scale_offset = (codes + 255U) & ~std::uint64_t{255U};
    return static_cast<std::size_t>(scale_offset + static_cast<std::uint64_t>(rows) * 4);
}

void quantize_bf16_output_head_to_fp8_e4m3_row_f32s(const Weight& bf16_head, DeviceBuffer& payload,
                                                    Weight& fp8_head, cudaStream_t stream) {
    quantize_bf16_head_to_fp8_e4m3_row_f32s(bf16_head, payload, fp8_head, kOutputHeadRows,
                                            kOutputHeadColumns, stream);
}

void quantize_bf16_head_to_fp8_e4m3_row_f32s(const Weight& bf16_head, DeviceBuffer& payload,
                                             Weight& fp8_head, std::int32_t rows,
                                             std::int32_t cols, cudaStream_t stream) {
    if (bf16_head.qtype != QType::BF16_CTRL || bf16_head.n != rows ||
        bf16_head.k != cols || bf16_head.qdata == nullptr) {
        throw std::invalid_argument("Flash-Next FP8 head quantize received an invalid BF16 view");
    }
    const std::size_t bytes = flash_next_fp8_head_payload_bytes(rows, cols);
    if (payload.p == nullptr || payload.bytes < bytes) {
        throw std::invalid_argument("Flash-Next FP8 head payload is too small");
    }
    const std::uint64_t codes        = static_cast<std::uint64_t>(rows) * cols;
    const std::uint64_t scale_offset = (codes + 255U) & ~std::uint64_t{255U};
    auto* code_ptr                   = static_cast<std::uint8_t*>(payload.p);
    auto* scale_ptr = reinterpret_cast<float*>(static_cast<std::byte*>(payload.p) + scale_offset);
    quantize_row_kernel<<<rows, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bf16_head.qdata), code_ptr, scale_ptr);
    CUDA_CHECK(cudaGetLastError());
    fp8_head = make_fp8_view_sized(payload.p, payload.bytes, rows, cols);
}

void gather_head_rows_bf16(const Weight& src_head, const std::int32_t* token_ids,
                           std::int32_t rows, std::int32_t cols,
                           DeviceBuffer& dst_payload, Weight& dst_head, cudaStream_t stream) {
    if (src_head.qtype != QType::BF16_CTRL || src_head.qdata == nullptr || token_ids == nullptr) {
        throw std::invalid_argument("gather_head_rows_bf16 received invalid inputs");
    }
    const std::size_t required_bytes = static_cast<std::size_t>(rows) * cols * sizeof(std::uint16_t);
    if (dst_payload.p == nullptr || dst_payload.bytes < required_bytes) {
        throw std::invalid_argument("gather_head_rows_bf16 dst_payload is too small");
    }
    gather_rows_kernel<<<rows, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(src_head.qdata), token_ids,
        static_cast<__nv_bfloat16*>(dst_payload.p), rows, cols);
    CUDA_CHECK(cudaGetLastError());

    Weight out{};
    out.payload         = dst_payload.p;
    out.payload_bytes   = dst_payload.bytes;
    out.qdata           = dst_payload.p;
    out.qtype           = QType::BF16_CTRL;
    out.layout          = QuantLayout::Contiguous;
    out.n               = rows;
    out.k               = cols;
    out.group           = cols;
    out.group_size      = static_cast<std::uint32_t>(cols);
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = cols;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = cols;
    dst_head = out;
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
