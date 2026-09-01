#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "targets/qwen3_8_flash_next/impl/load/quantize_output_head.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

std::uint16_t float_to_bf16(float value) {
    const __nv_bfloat16 raw = __float2bfloat16_rn(value);
    return *reinterpret_cast<const std::uint16_t*>(&raw);
}

float bf16_to_float(std::uint16_t bits) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bits));
}

ninfer::Weight make_bf16_head(void* data) {
    constexpr std::int32_t rows    = 248'320;
    constexpr std::int32_t columns = 2'560;
    ninfer::Weight out{};
    out.payload         = data;
    out.payload_bytes   = static_cast<std::uint64_t>(rows) * columns * 2;
    out.qdata           = data;
    out.qtype           = ninfer::QType::BF16_CTRL;
    out.layout          = ninfer::QuantLayout::Contiguous;
    out.n               = rows;
    out.k               = columns;
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    return out;
}

void fill_patterned_bf16(ninfer::DeviceBuffer& buffer, std::uint32_t seed) {
    constexpr std::int32_t rows    = 248'320;
    constexpr std::int32_t columns = 2'560;
    std::vector<std::uint16_t> host(static_cast<std::size_t>(rows) * columns);
    for (std::size_t i = 0; i < host.size(); ++i) {
        std::uint32_t x = static_cast<std::uint32_t>(i) * 0x9E3779B9u ^ seed;
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        const float value =
            static_cast<float>(static_cast<std::int32_t>(x)) * (1.0F / 2147483648.0F);
        host[i] = float_to_bf16(value);
    }
    buffer.copy_from_host(host.data(), host.size() * sizeof(std::uint16_t));
}

int argmax_column(const std::vector<std::uint16_t>& logits, std::int32_t rows, std::int32_t token) {
    int best      = 0;
    float best_v  = bf16_to_float(logits[static_cast<std::size_t>(token) * rows]);
    for (std::int32_t row = 1; row < rows; ++row) {
        const float v = bf16_to_float(logits[static_cast<std::size_t>(token) * rows + row]);
        if (v > best_v) {
            best_v = v;
            best   = row;
        }
    }
    return best;
}

} // namespace

int main() {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || device_count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_error);
    ninfer::DeviceContext device(0);

    constexpr std::int32_t rows    = 248'320;
    constexpr std::int32_t columns = 2'560;
    constexpr std::int32_t tokens  = 8;
    ninfer::DeviceBuffer bf16_storage(static_cast<std::size_t>(rows) * columns * 2);
    fill_patterned_bf16(bf16_storage, 20260901U);
    CUDA_CHECK(cudaDeviceSynchronize());
    ninfer::Weight bf16_head = make_bf16_head(bf16_storage.p);

    ninfer::DeviceBuffer fp8_storage(flash_next_fp8_output_head_payload_bytes());
    ninfer::Weight fp8_head{};
    quantize_bf16_output_head_to_fp8_e4m3_row_f32s(bf16_head, fp8_storage, fp8_head, device.stream);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (fp8_head.qtype != ninfer::QType::FP8_E4M3FN_ROW_F32S || fp8_head.n != rows ||
        fp8_head.k != columns || fp8_head.scale_dtype != ninfer::DType::FP32) {
        std::cerr << "FAIL: quantized output head is not FP8_E4M3FN_ROW_F32S [248320,2560]\n";
        return 1;
    }

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-0.5F, 0.5F);
    constexpr int kBatches          = 512;
    constexpr int kColumns          = kBatches * tokens;
    std::int32_t flips              = 0;
    double logit_diff_sq            = 0.0;
    double logit_base_sq            = 0.0;
    int nonfinite                   = 0;
    ninfer::DeviceBuffer hidden_dev(static_cast<std::size_t>(columns) * tokens * 2);
    ninfer::DeviceBuffer bf16_logits_dev(static_cast<std::size_t>(rows) * tokens * 2);
    ninfer::DeviceBuffer fp8_logits_dev(static_cast<std::size_t>(rows) * tokens * 2);
    ninfer::WorkspaceArena workspace(256);
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(columns) * tokens);
    std::vector<std::uint16_t> bf16_logits(static_cast<std::size_t>(rows) * tokens);
    std::vector<std::uint16_t> fp8_logits(static_cast<std::size_t>(rows) * tokens);

    for (int batch = 0; batch < kBatches; ++batch) {
        for (auto& v : hidden) { v = float_to_bf16(dist(rng)); }
        hidden_dev.copy_from_host(hidden.data(), hidden.size() * sizeof(std::uint16_t));
        CUDA_CHECK(cudaDeviceSynchronize());
        ninfer::Tensor hidden_view(hidden_dev.p, ninfer::DType::BF16, {columns, tokens});
        ninfer::Tensor bf16_out(bf16_logits_dev.p, ninfer::DType::BF16, {rows, tokens});
        ninfer::Tensor fp8_out(fp8_logits_dev.p, ninfer::DType::BF16, {rows, tokens});
        ninfer::ops::linear(hidden_view, bf16_head, bf16_out, ninfer::ops::LinearPolicy::A16Only,
                            workspace, device.stream);
        ninfer::ops::linear(hidden_view, fp8_head, fp8_out, ninfer::ops::LinearPolicy::A16Only,
                            workspace, device.stream);
        CUDA_CHECK(cudaDeviceSynchronize());
        bf16_logits_dev.copy_to_host(bf16_logits.data(), bf16_logits.size() * sizeof(std::uint16_t));
        fp8_logits_dev.copy_to_host(fp8_logits.data(), fp8_logits.size() * sizeof(std::uint16_t));
        for (std::int32_t t = 0; t < tokens; ++t) {
            if (argmax_column(bf16_logits, rows, t) != argmax_column(fp8_logits, rows, t)) {
                ++flips;
            }
            for (std::int32_t row = 0; row < rows; ++row) {
                const float a =
                    bf16_to_float(bf16_logits[static_cast<std::size_t>(t) * rows + row]);
                const float b = bf16_to_float(fp8_logits[static_cast<std::size_t>(t) * rows + row]);
                if (!std::isfinite(a) || !std::isfinite(b)) { ++nonfinite; }
                logit_base_sq += static_cast<double>(a) * a;
                const double d = static_cast<double>(a) - b;
                logit_diff_sq += d * d;
            }
        }
    }

    if (nonfinite > 0 || logit_base_sq <= 0.0) {
        std::cerr << "FAIL: BF16 vs FP8 comparison was vacuous or non-finite base_sq="
                  << logit_base_sq << " nonfinite=" << nonfinite << "\n";
        return 1;
    }
    const double rel_l2 = std::sqrt(logit_diff_sq / logit_base_sq);
    std::cout << "G2 argmax samples=" << kColumns << " flips=" << flips
              << " flip_rate=" << (static_cast<double>(flips) / kColumns) << " logits_rel_L2=" << rel_l2
              << " base_sq=" << logit_base_sq << "\n";

    constexpr int kWarmup = 5;
    constexpr int kIters  = 20;
    auto time_linear      = [&](ninfer::Weight& weight, std::int32_t t) {
        ninfer::Tensor hidden_view(hidden_dev.p, ninfer::DType::BF16, {columns, t});
        ninfer::Tensor out_view(bf16_logits_dev.p, ninfer::DType::BF16, {rows, t});
        for (int i = 0; i < kWarmup; ++i) {
            ninfer::ops::linear(hidden_view, weight, out_view, ninfer::ops::LinearPolicy::A16Only,
                                workspace, device.stream);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start = nullptr;
        cudaEvent_t stop  = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, device.stream));
        for (int i = 0; i < kIters; ++i) {
            ninfer::ops::linear(hidden_view, weight, out_view, ninfer::ops::LinearPolicy::A16Only,
                                workspace, device.stream);
        }
        CUDA_CHECK(cudaEventRecord(stop, device.stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return ms * 1000.0F / static_cast<float>(kIters);
    };

    const float us_bf16_c1 = time_linear(bf16_head, 1);
    const float us_fp8_c1  = time_linear(fp8_head, 1);
    const float us_bf16_c8 = time_linear(bf16_head, 8);
    const float us_fp8_c8  = time_linear(fp8_head, 8);
    std::cout << "G2 head kernel c=1 BF16=" << us_bf16_c1 << " us  FP8=" << us_fp8_c1 << " us\n";
    std::cout << "G2 head kernel c=8 BF16=" << us_bf16_c8 << " us  FP8=" << us_fp8_c8 << " us\n";
    std::cout << "PASS: test_output_head_fp8\n";
    return 0;
}
