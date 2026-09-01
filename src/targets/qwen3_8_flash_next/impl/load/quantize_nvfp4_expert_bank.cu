#include "targets/qwen3_8_flash_next/impl/load/quantize_nvfp4_expert_bank.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

constexpr int kThreads = 256;

__global__ __launch_bounds__(kThreads, 2) void quantize_bf16_expert_bank_kernel(
    const __nv_bfloat16* __restrict__ input, std::uint8_t* __restrict__ code_plane,
    std::uint8_t* __restrict__ scale_plane, float* __restrict__ weight_divisors,
    int experts, int rows, int columns, int groups_per_row, int k_tiles, std::int64_t total_groups) {

    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_groups) { return; }

    const int group_in_row = static_cast<int>(idx % groups_per_row);
    const std::int64_t row_flat = idx / groups_per_row;
    const int row          = static_cast<int>(row_flat % rows);
    const int expert       = static_cast<int>(row_flat / rows);

    const auto* src = input + (static_cast<std::int64_t>(expert) * rows + row) * columns + group_in_row * 16;
    const uint4 packed0 = *reinterpret_cast<const uint4*>(src);
    const uint4 packed1 = *reinterpret_cast<const uint4*>(src + 8);

    const std::uint32_t represented[8] = {
        packed0.x, packed0.y, packed0.z, packed0.w,
        packed1.x, packed1.y, packed1.z, packed1.w,
    };

    float2 values[8];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = ops::bf16x2_bits_to_float2(represented[pair]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    const float scale_unencoded = __fdiv_rn(max_abs, 6.0F);
    const std::uint8_t scale    = __nv_cvt_float_to_fp8(scale_unencoded, __NV_SATFINITE, __NV_E4M3);

    std::uint32_t codes_lo = 0;
    std::uint32_t codes_hi = 0;
    if (scale != 0) {
        const float decoded_scale = ops::detail::decode_nvfp4_e4m3(scale);
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            values[pair].x = __fdiv_rn(values[pair].x, decoded_scale);
            values[pair].y = __fdiv_rn(values[pair].y, decoded_scale);
        }
        ops::detail::pack_nvfp4_e2m1x16(values, codes_lo, codes_hi);
    }

    // 1. Write 8 code bytes (16 nibbles)
    const std::size_t code_offset =
        (static_cast<std::size_t>(expert) * rows + row) * (columns / 2) + group_in_row * 8;
    *reinterpret_cast<uint2*>(code_plane + code_offset) = make_uint2(codes_lo, codes_hi);

    // 2. Write swizzled scale byte
    const int m_tile    = row / 128;
    const int row_inner = row % 128;
    const int row_mod32 = row_inner & 31;
    const int row_quart = row_inner >> 5;
    const int k_tile    = group_in_row >> 2;
    const int k_mod4    = group_in_row & 3;
    const std::size_t expert_scale_offset =
        static_cast<std::size_t>(m_tile * k_tiles + k_tile) * 512 +
        row_mod32 * 16 + row_quart * 4 + k_mod4;
    const std::size_t scale_offset =
        static_cast<std::size_t>(expert) * (static_cast<std::size_t>(rows) * groups_per_row) + expert_scale_offset;
    scale_plane[scale_offset] = scale;

    // 3. Set expert divisor
    if (row == 0 && group_in_row == 0) {
        weight_divisors[expert] = 1.0F;
    }
}

} // namespace

std::size_t flash_next_nvfp4_expert_bank_payload_bytes(std::int32_t experts, std::int32_t rows,
                                                       std::int32_t columns) {
    if (experts <= 0 || rows <= 0 || columns <= 0 || (rows % 128) != 0 || (columns % 64) != 0) {
        throw std::invalid_argument("Invalid NVFP4 expert bank dimensions");
    }
    const std::uint64_t elements             = static_cast<std::uint64_t>(experts) * rows * columns;
    const std::uint64_t code_plane_bytes     = elements / 2;
    const std::uint64_t scale_plane_offset   = (code_plane_bytes + 255U) & ~std::uint64_t{255U};
    const std::uint64_t scale_plane_bytes    = elements / 16;
    const std::uint64_t weight_divisor_offset = scale_plane_offset + scale_plane_bytes;
    const std::uint64_t weight_divisor_bytes  = static_cast<std::uint64_t>(experts) * sizeof(float);
    return static_cast<std::size_t>(weight_divisor_offset + weight_divisor_bytes);
}

void quantize_bf16_expert_bank_to_nvfp4(const void* bf16_data, void* nvfp4_payload,
                                        std::int32_t experts, std::int32_t rows,
                                        std::int32_t columns, cudaStream_t stream) {
    if (bf16_data == nullptr || nvfp4_payload == nullptr || experts <= 0 || rows <= 0 ||
        columns <= 0 || (rows % 128) != 0 || (columns % 64) != 0) {
        throw std::invalid_argument("Invalid arguments to quantize_bf16_expert_bank_to_nvfp4");
    }

    const std::uint64_t elements             = static_cast<std::uint64_t>(experts) * rows * columns;
    const std::uint64_t code_plane_bytes     = elements / 2;
    const std::uint64_t scale_plane_offset   = (code_plane_bytes + 255U) & ~std::uint64_t{255U};
    const std::uint64_t scale_plane_bytes    = elements / 16;
    const std::uint64_t weight_divisor_offset = scale_plane_offset + scale_plane_bytes;

    auto* code_plane       = static_cast<std::uint8_t*>(nvfp4_payload);
    auto* scale_plane      = static_cast<std::uint8_t*>(nvfp4_payload) + scale_plane_offset;
    auto* weight_divisors  = reinterpret_cast<float*>(static_cast<std::byte*>(nvfp4_payload) + weight_divisor_offset);

    const int groups_per_row = columns / 16;
    const int k_tiles        = columns / 64;
    const std::int64_t total_groups = static_cast<std::int64_t>(experts) * rows * groups_per_row;
    const int blocks = static_cast<int>((total_groups + kThreads - 1) / kThreads);

    quantize_bf16_expert_bank_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bf16_data), code_plane, scale_plane, weight_divisors,
        experts, rows, columns, groups_per_row, k_tiles, total_groups);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void populate_synthetic_bf16_kernel(__nv_bfloat16* data, std::int64_t total, float scale) {
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) { return; }
    const float v = scale * sinf(static_cast<float>(idx % 317 + 1));
    data[idx]     = __float2bfloat16(v);
}

void populate_synthetic_bf16_bank(void* ptr, std::size_t elements, float scale,
                                  cudaStream_t stream) {
    if (ptr == nullptr || elements == 0) { return; }
    const int blocks = static_cast<int>((elements + kThreads - 1) / kThreads);
    populate_synthetic_bf16_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(ptr), static_cast<std::int64_t>(elements), scale);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void populate_constant_bf16_kernel(__nv_bfloat16* data, std::int64_t total, float val) {
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) { return; }
    data[idx] = __float2bfloat16(val);
}

void populate_constant_bf16_bank(void* ptr, std::size_t elements, float val,
                                 cudaStream_t stream) {
    if (ptr == nullptr || elements == 0) { return; }
    const int blocks = static_cast<int>((elements + kThreads - 1) / kThreads);
    populate_constant_bf16_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(ptr), static_cast<std::int64_t>(elements), val);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
