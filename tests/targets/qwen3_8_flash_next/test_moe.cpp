#include "core/arena.h"
#include "core/device.h"
#include "core/layout.h"
#include "targets/qwen3_8_flash_next/impl/moe.h"
#include "targets/qwen3_8_flash_next/impl/moe_kernels.h"
#include "targets/qwen3_8_flash_next/impl/moe_route.h"
#include "targets/qwen3_8_flash_next/impl/moe_workspace.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

ninfer::Weight bf16_weight(void* data, std::int32_t rows, std::int32_t columns) {
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

int test_basic_unit(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    ninfer::DeviceBuffer input(2'560 * 2);
    ninfer::DeviceBuffer output(2'560 * 2);
    ninfer::DeviceBuffer router(512ULL * 2'560 * 2);
    ninfer::DeviceBuffer shared_down(2'560ULL * 640 * 2);
    ninfer::DeviceBuffer shared_gate(640ULL * 2'560 * 2);
    ninfer::DeviceBuffer shared_up(640ULL * 2'560 * 2);
    ninfer::DeviceBuffer shared_gate_weight(2'560 * 2);
    std::array<std::uint16_t, 2'560> input_bf16{};
    input_bf16.fill(0x3F80U); // 1.0 BF16
    input.copy_from_host(input_bf16.data(), sizeof(input_bf16));
    output.fill(0xFF);
    router.fill(0);
    shared_down.fill(0);
    shared_gate.fill(0);
    shared_up.fill(0);
    shared_gate_weight.fill(0);

    constexpr std::uint64_t gate_code_bytes_per_expert  = 1'280ULL * 2'560 / 2;
    constexpr std::uint64_t gate_scale_bytes_per_expert = 1'280ULL * 2'560 / 16;
    constexpr std::uint64_t down_code_bytes_per_expert  = 2'560ULL * 640 / 2;
    constexpr std::uint64_t down_scale_bytes_per_expert = 2'560ULL * 640 / 16;
    // Zero router ties select experts 0..9, so the numerical smoke only needs those physical
    // expert spans while the view retains the registered 512-expert stride contract.
    ninfer::DeviceBuffer gate_codes(10 * gate_code_bytes_per_expert);
    ninfer::DeviceBuffer gate_scales(10 * gate_scale_bytes_per_expert);
    ninfer::DeviceBuffer down_codes(10 * down_code_bytes_per_expert);
    ninfer::DeviceBuffer down_scales(10 * down_scale_bytes_per_expert);
    ninfer::DeviceBuffer gate_divisors(512 * sizeof(float));
    ninfer::DeviceBuffer down_divisors(512 * sizeof(float));
    std::vector<std::uint8_t> gate_code_values(10 * gate_code_bytes_per_expert, 0);
    std::vector<std::uint8_t> gate_scale_values(10 * gate_scale_bytes_per_expert, 0x38U);
    std::vector<std::uint8_t> down_code_values(10 * down_code_bytes_per_expert, 0);
    std::vector<std::uint8_t> down_scale_values(10 * down_scale_bytes_per_expert, 0x38U);
    for (std::uint64_t expert = 0; expert < 10; ++expert) {
        for (std::uint64_t row = 0; row < 1'280; ++row) {
            std::fill_n(gate_code_values.begin() +
                            static_cast<std::ptrdiff_t>(expert * gate_code_bytes_per_expert +
                                                        row * (2'560 / 2)),
                        8, 0x22U); // first K16 group: sixteen E2M1 values of 1.0
        }
        for (std::uint64_t row = 0; row < 2'560; ++row) {
            std::fill_n(down_code_values.begin() +
                            static_cast<std::ptrdiff_t>(expert * down_code_bytes_per_expert +
                                                        row * (640 / 2)),
                        8, 0x22U);
        }
    }
    gate_codes.copy_from_host(gate_code_values.data(), gate_code_values.size());
    gate_scales.copy_from_host(gate_scale_values.data(), gate_scale_values.size());
    down_codes.copy_from_host(down_code_values.data(), down_code_values.size());
    down_scales.copy_from_host(down_scale_values.data(), down_scale_values.size());
    std::array<float, 512> divisors{};
    divisors.fill(1.0F);
    gate_divisors.copy_from_host(divisors.data(), sizeof(divisors));
    down_divisors.copy_from_host(divisors.data(), sizeof(divisors));

    MoeWeights weights{
        .router             = bf16_weight(router.p, 512, 2'560),
        .shared_down        = bf16_weight(shared_down.p, 2'560, 640),
        .shared_gate        = bf16_weight(shared_gate.p, 640, 2'560),
        .shared_up          = bf16_weight(shared_up.p, 640, 2'560),
        .shared_gate_weight = bf16_weight(shared_gate_weight.p, 1, 2'560),
        .expert_gate_up     = {.codes                  = static_cast<const std::byte*>(gate_codes.p),
                               .scales                 = static_cast<const std::byte*>(gate_scales.p),
                               .weight_scale_divisors  = static_cast<const float*>(gate_divisors.p),
                               .experts                = 512,
                               .rows                   = 1'280,
                               .columns                = 2'560,
                               .code_bytes_per_expert  = gate_code_bytes_per_expert,
                               .scale_bytes_per_expert = gate_scale_bytes_per_expert},
        .expert_down        = {.codes                  = static_cast<const std::byte*>(down_codes.p),
                               .scales                 = static_cast<const std::byte*>(down_scales.p),
                               .weight_scale_divisors  = static_cast<const float*>(down_divisors.p),
                               .experts                = 512,
                               .rows                   = 2'560,
                               .columns                = 640,
                               .code_bytes_per_expert  = down_code_bytes_per_expert,
                               .scale_bytes_per_expert = down_scale_bytes_per_expert},
    };
    ninfer::Tensor input_view(input.p, ninfer::DType::BF16, {2'560, 1});
    ninfer::Tensor output_view(output.p, ninfer::DType::BF16, {2'560, 1});
    ninfer::WorkspaceArena workspace(flash_next_moe_workspace_capacity_bytes(1, 1));
    flash_next_moe(input_view, weights, output_view, workspace, device.stream);
    device.synchronize();

    std::array<std::uint16_t, 2'560> actual{};
    output.copy_to_host(actual.data(), sizeof(actual));
    // Each selected expert produces 4096, and top-10 softmax renormalization gives each one weight
    // 1/10. Ten selected experts therefore produce exactly 4096.0 (0x4580 in BF16); the zero shared expert adds 0.
    if (!std::all_of(actual.begin(), actual.end(),
                     [](std::uint16_t value) { return value == 0x4580U; })) {
        std::cerr << "Flash-Next encoded NVFP4 MoE did not produce exact BF16 4096.0: first=0x"
                  << std::hex << actual.front() << '\n';
        return 1;
    }
    std::cout << "PASS: test_decode_basic_unit\n";
    return 0;
}

float bf16_to_float(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::uint16_t float_to_bf16(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t lsb  = (bits >> 16U) & 1U;
    const std::uint32_t bias = 0x7FFFU + lsb;
    return static_cast<std::uint16_t>((bits + bias) >> 16U);
}

double relative_l2_error(const std::vector<float>& act, const std::vector<float>& exp) {
    double diff_sq = 0.0;
    double exp_sq  = 0.0;
    for (std::size_t i = 0; i < act.size(); ++i) {
        double d = static_cast<double>(act[i]) - static_cast<double>(exp[i]);
        diff_sq += d * d;
        exp_sq  += static_cast<double>(exp[i]) * static_cast<double>(exp[i]);
    }
    return std::sqrt(diff_sq) / (std::sqrt(exp_sq) + 1.0e-12);
}

int test_decode_cobatching_equivalence(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    constexpr int kHidden = 2'560;
    constexpr int kExperts = 512;
    constexpr std::uint64_t gate_code_bytes_per_expert  = 1'280ULL * 2'560 / 2;
    constexpr std::uint64_t gate_scale_bytes_per_expert = 1'280ULL * 2'560 / 16;
    constexpr std::uint64_t down_code_bytes_per_expert  = 2'560ULL * 640 / 2;
    constexpr std::uint64_t down_scale_bytes_per_expert = 2'560ULL * 640 / 16;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_router(-0.1F, 0.1F);
    std::uniform_real_distribution<float> dist_act(-0.05F, 0.05F);

    // Allocate full 512 physical experts for realistic decode routing across the full catalog
    ninfer::DeviceBuffer router(512ULL * kHidden * 2);
    ninfer::DeviceBuffer shared_down(kHidden * 640ULL * 2);
    ninfer::DeviceBuffer shared_gate(640ULL * kHidden * 2);
    ninfer::DeviceBuffer shared_up(640ULL * kHidden * 2);
    ninfer::DeviceBuffer shared_gate_weight(kHidden * 2);
    ninfer::DeviceBuffer gate_codes(kExperts * gate_code_bytes_per_expert);
    ninfer::DeviceBuffer gate_scales(kExperts * gate_scale_bytes_per_expert);
    ninfer::DeviceBuffer down_codes(kExperts * down_code_bytes_per_expert);
    ninfer::DeviceBuffer down_scales(kExperts * down_scale_bytes_per_expert);
    ninfer::DeviceBuffer gate_divisors(kExperts * sizeof(float));
    ninfer::DeviceBuffer down_divisors(kExperts * sizeof(float));

    std::vector<std::uint8_t> h_gate_codes(kExperts * gate_code_bytes_per_expert, 0x22U);
    std::vector<std::uint8_t> h_gate_scales(kExperts * gate_scale_bytes_per_expert, 0x38U);
    std::vector<std::uint8_t> h_down_codes(kExperts * down_code_bytes_per_expert, 0x22U);
    std::vector<std::uint8_t> h_down_scales(kExperts * down_scale_bytes_per_expert, 0x38U);
    std::vector<float> h_divisors(kExperts, 1.0F);

    gate_codes.copy_from_host(h_gate_codes.data(), h_gate_codes.size());
    gate_scales.copy_from_host(h_gate_scales.data(), h_gate_scales.size());
    down_codes.copy_from_host(h_down_codes.data(), h_down_codes.size());
    down_scales.copy_from_host(h_down_scales.data(), h_down_scales.size());
    gate_divisors.copy_from_host(h_divisors.data(), sizeof(float) * kExperts);
    down_divisors.copy_from_host(h_divisors.data(), sizeof(float) * kExperts);

    std::vector<std::uint16_t> h_router(512ULL * kHidden);
    for (auto& v : h_router) { v = float_to_bf16(dist_router(rng)); }
    router.copy_from_host(h_router.data(), h_router.size() * 2);

    std::vector<std::uint16_t> h_shared_down(kHidden * 640ULL);
    std::vector<std::uint16_t> h_shared_gate(640ULL * kHidden);
    std::vector<std::uint16_t> h_shared_up(640ULL * kHidden);
    std::vector<std::uint16_t> h_shared_gate_weight(kHidden);
    for (auto& v : h_shared_down) { v = float_to_bf16(dist_router(rng)); }
    for (auto& v : h_shared_gate) { v = float_to_bf16(dist_router(rng)); }
    for (auto& v : h_shared_up) { v = float_to_bf16(dist_router(rng)); }
    for (auto& v : h_shared_gate_weight) { v = float_to_bf16(dist_router(rng)); }
    shared_down.copy_from_host(h_shared_down.data(), h_shared_down.size() * 2);
    shared_gate.copy_from_host(h_shared_gate.data(), h_shared_gate.size() * 2);
    shared_up.copy_from_host(h_shared_up.data(), h_shared_up.size() * 2);
    shared_gate_weight.copy_from_host(h_shared_gate_weight.data(), h_shared_gate_weight.size() * 2);

    MoeWeights weights{
        .router             = bf16_weight(router.p, 512, kHidden),
        .shared_down        = bf16_weight(shared_down.p, kHidden, 640),
        .shared_gate        = bf16_weight(shared_gate.p, 640, kHidden),
        .shared_up          = bf16_weight(shared_up.p, 640, kHidden),
        .shared_gate_weight = bf16_weight(shared_gate_weight.p, 1, kHidden),
        .expert_gate_up     = {.codes                  = static_cast<const std::byte*>(gate_codes.p),
                               .scales                 = static_cast<const std::byte*>(gate_scales.p),
                               .weight_scale_divisors  = static_cast<const float*>(gate_divisors.p),
                               .experts                = 512,
                               .rows                   = 1'280,
                               .columns                = kHidden,
                               .code_bytes_per_expert  = gate_code_bytes_per_expert,
                               .scale_bytes_per_expert = gate_scale_bytes_per_expert},
        .expert_down        = {.codes                  = static_cast<const std::byte*>(down_codes.p),
                               .scales                 = static_cast<const std::byte*>(down_scales.p),
                               .weight_scale_divisors  = static_cast<const float*>(down_divisors.p),
                               .experts                = 512,
                               .rows                   = kHidden,
                               .columns                = 640,
                               .code_bytes_per_expert  = down_code_bytes_per_expert,
                               .scale_bytes_per_expert = down_scale_bytes_per_expert},
    };

    const std::vector<int> test_batch_sizes = {1, 2, 3, 4, 6, 8};

    std::cout << "\n========================================================================================================\n";
    std::cout << "--- HEAD-TO-HEAD BENCHMARK: REALISTIC ROUTING (Diverse Independent Tokens) ---\n";
    std::cout << "========================================================================================================\n";
    std::cout << std::left
              << std::setw(4)  << "B"
              << std::setw(16) << "Unique Exps"
              << std::setw(18) << "Baseline Tot(us)"
              << std::setw(18) << "Baseline us/tok"
              << std::setw(18) << "Cobatch Tot(us)"
              << std::setw(18) << "Cobatch us/tok"
              << std::setw(12) << "Speedup"
              << "Winner\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";

    for (int B : test_batch_sizes) {
        std::vector<std::uint16_t> host_input(B * kHidden);
        for (auto& v : host_input) { v = float_to_bf16(dist_act(rng)); }

        ninfer::DeviceBuffer dev_input(B * kHidden * 2);
        dev_input.copy_from_host(host_input.data(), host_input.size() * 2);

        ninfer::DeviceBuffer dev_out_baseline(B * kHidden * 2);
        ninfer::DeviceBuffer dev_out_cobatched(B * kHidden * 2);
        dev_out_baseline.fill(0);
        dev_out_cobatched.fill(0);

        ninfer::Tensor input_view(dev_input.p, ninfer::DType::BF16, {kHidden, B});
        ninfer::Tensor out_base_view(dev_out_baseline.p, ninfer::DType::BF16, {kHidden, B});
        ninfer::Tensor out_cobatch_view(dev_out_cobatched.p, ninfer::DType::BF16, {kHidden, B});

        ninfer::WorkspaceArena ws(flash_next_moe_workspace_capacity_bytes(1, 8));
        auto scratch = allocate_flash_next_moe_workspace(ws, B);

        flash_next_route(input_view, weights.router, weights.shared_gate_weight,
                         scratch.scores, scratch.ids, scratch.alpha, scratch.shared_scale, device.stream);
        device.synchronize();

        // Count unique active experts
        std::vector<std::int32_t> h_ids(B * 10);
        cudaMemcpy(h_ids.data(), scratch.ids.data, B * 10 * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
        std::set<std::int32_t> unique_set(h_ids.begin(), h_ids.end());
        const int num_unique = static_cast<int>(unique_set.size());

        // 1. Equivalence check
        flash_next_moe_launch_baseline_decode(input_view, weights, scratch, out_base_view, device.stream);
        flash_next_moe_launch_cobatched_decode(input_view, weights, scratch, out_cobatch_view, device.stream);
        device.synchronize();

        std::vector<std::uint16_t> h_out_base(B * kHidden);
        std::vector<std::uint16_t> h_out_cobatch(B * kHidden);
        dev_out_baseline.copy_to_host(h_out_base.data(), h_out_base.size() * 2);
        dev_out_cobatched.copy_to_host(h_out_cobatch.data(), h_out_cobatch.size() * 2);

        int bitwise_mismatches = 0;
        for (int i = 0; i < B * kHidden; ++i) {
            if (h_out_base[i] != h_out_cobatch[i]) {
                if (bitwise_mismatches < 5) {
                    std::cerr << "Mismatch at B=" << B << " index " << i
                              << ": base=0x" << std::hex << h_out_base[i]
                              << " vs cobatch=0x" << h_out_cobatch[i] << std::dec << '\n';
                }
                bitwise_mismatches++;
            }
        }
        if (bitwise_mismatches > 0) {
            std::cerr << "FAILED: B=" << B << " had " << bitwise_mismatches << " bitwise mismatches!\n";
            return 1;
        }

        // 2. Timing benchmark (warmup 20 + 200 iters)
        constexpr int kIters = 200;
        for (int i = 0; i < 20; ++i) {
            flash_next_moe_launch_baseline_decode(input_view, weights, scratch, out_base_view, device.stream);
        }
        device.synchronize();

        cudaEvent_t ev_start, ev_end;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_end));

        CUDA_CHECK(cudaEventRecord(ev_start, device.stream));
        for (int i = 0; i < kIters; ++i) {
            flash_next_moe_launch_baseline_decode(input_view, weights, scratch, out_base_view, device.stream);
        }
        CUDA_CHECK(cudaEventRecord(ev_end, device.stream));
        CUDA_CHECK(cudaEventSynchronize(ev_end));
        float ms_base = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms_base, ev_start, ev_end));
        const float us_base = (ms_base / kIters) * 1000.0F;

        for (int i = 0; i < 20; ++i) {
            flash_next_moe_launch_cobatched_decode(input_view, weights, scratch, out_cobatch_view, device.stream);
        }
        device.synchronize();

        CUDA_CHECK(cudaEventRecord(ev_start, device.stream));
        for (int i = 0; i < kIters; ++i) {
            flash_next_moe_launch_cobatched_decode(input_view, weights, scratch, out_cobatch_view, device.stream);
        }
        CUDA_CHECK(cudaEventRecord(ev_end, device.stream));
        CUDA_CHECK(cudaEventSynchronize(ev_end));
        float ms_cobatch = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms_cobatch, ev_start, ev_end));
        const float us_cobatch = (ms_cobatch / kIters) * 1000.0F;

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_end));

        const float speedup = us_base / us_cobatch;
        const std::string winner = (speedup >= 1.0F) ? "Co-batched" : "Baseline";
        std::string unq_str = std::to_string(num_unique) + " / " + std::to_string(B * 10);

        std::cout << std::left
                  << std::setw(4)  << B
                  << std::setw(16) << unq_str
                  << std::setw(18) << std::fixed << std::setprecision(2) << us_base
                  << std::setw(18) << std::fixed << std::setprecision(2) << (us_base / B)
                  << std::setw(18) << std::fixed << std::setprecision(2) << us_cobatch
                  << std::setw(18) << std::fixed << std::setprecision(2) << (us_cobatch / B)
                  << std::setw(12) << std::fixed << std::setprecision(3) << speedup
                  << winner << "\n";
    }

    std::cout << "\n========================================================================================================\n";
    std::cout << "--- HEAD-TO-HEAD BENCHMARK: MAXIMUM OVERLAP (Identical Tokens -> Exact 10 Unique Experts) ---\n";
    std::cout << "========================================================================================================\n";
    std::cout << std::left
              << std::setw(4)  << "B"
              << std::setw(16) << "Unique Exps"
              << std::setw(18) << "Baseline Tot(us)"
              << std::setw(18) << "Baseline us/tok"
              << std::setw(18) << "Cobatch Tot(us)"
              << std::setw(18) << "Cobatch us/tok"
              << std::setw(12) << "Speedup"
              << "Winner\n";
    std::cout << "--------------------------------------------------------------------------------------------------------\n";

    for (int B : test_batch_sizes) {
        // Replicate a single token across all B positions
        std::vector<std::uint16_t> single_token(kHidden);
        for (auto& v : single_token) { v = float_to_bf16(dist_act(rng)); }
        std::vector<std::uint16_t> host_input(B * kHidden);
        for (int b = 0; b < B; ++b) {
            std::copy(single_token.begin(), single_token.end(), host_input.begin() + b * kHidden);
        }

        ninfer::DeviceBuffer dev_input(B * kHidden * 2);
        dev_input.copy_from_host(host_input.data(), host_input.size() * 2);

        ninfer::DeviceBuffer dev_out_baseline(B * kHidden * 2);
        ninfer::DeviceBuffer dev_out_cobatched(B * kHidden * 2);
        dev_out_baseline.fill(0);
        dev_out_cobatched.fill(0);

        ninfer::Tensor input_view(dev_input.p, ninfer::DType::BF16, {kHidden, B});
        ninfer::Tensor out_base_view(dev_out_baseline.p, ninfer::DType::BF16, {kHidden, B});
        ninfer::Tensor out_cobatch_view(dev_out_cobatched.p, ninfer::DType::BF16, {kHidden, B});

        ninfer::WorkspaceArena ws(flash_next_moe_workspace_capacity_bytes(1, 8));
        auto scratch = allocate_flash_next_moe_workspace(ws, B);

        flash_next_route(input_view, weights.router, weights.shared_gate_weight,
                         scratch.scores, scratch.ids, scratch.alpha, scratch.shared_scale, device.stream);
        device.synchronize();

        // 1. Timing benchmark (warmup 20 + 200 iters)
        constexpr int kIters = 200;
        for (int i = 0; i < 20; ++i) {
            flash_next_moe_launch_baseline_decode(input_view, weights, scratch, out_base_view, device.stream);
        }
        device.synchronize();

        cudaEvent_t ev_start, ev_end;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_end));

        CUDA_CHECK(cudaEventRecord(ev_start, device.stream));
        for (int i = 0; i < kIters; ++i) {
            flash_next_moe_launch_baseline_decode(input_view, weights, scratch, out_base_view, device.stream);
        }
        CUDA_CHECK(cudaEventRecord(ev_end, device.stream));
        CUDA_CHECK(cudaEventSynchronize(ev_end));
        float ms_base = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms_base, ev_start, ev_end));
        const float us_base = (ms_base / kIters) * 1000.0F;

        for (int i = 0; i < 20; ++i) {
            flash_next_moe_launch_cobatched_decode(input_view, weights, scratch, out_cobatch_view, device.stream);
        }
        device.synchronize();

        CUDA_CHECK(cudaEventRecord(ev_start, device.stream));
        for (int i = 0; i < kIters; ++i) {
            flash_next_moe_launch_cobatched_decode(input_view, weights, scratch, out_cobatch_view, device.stream);
        }
        CUDA_CHECK(cudaEventRecord(ev_end, device.stream));
        CUDA_CHECK(cudaEventSynchronize(ev_end));
        float ms_cobatch = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&ms_cobatch, ev_start, ev_end));
        const float us_cobatch = (ms_cobatch / kIters) * 1000.0F;

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_end));

        const float speedup = us_base / us_cobatch;
        const std::string winner = (speedup >= 1.0F) ? "Co-batched" : "Baseline";
        std::string unq_str = "10 / " + std::to_string(B * 10);

        std::cout << std::left
                  << std::setw(4)  << B
                  << std::setw(16) << unq_str
                  << std::setw(18) << std::fixed << std::setprecision(2) << us_base
                  << std::setw(18) << std::fixed << std::setprecision(2) << (us_base / B)
                  << std::setw(18) << std::fixed << std::setprecision(2) << us_cobatch
                  << std::setw(18) << std::fixed << std::setprecision(2) << (us_cobatch / B)
                  << std::setw(12) << std::fixed << std::setprecision(3) << speedup
                  << winner << "\n";
    }
    std::cout << "========================================================================================================\n\n";

    std::cout << "PASS: test_decode_cobatching_equivalence\n";
    return 0;
}

int test_prefill_equivalence_and_benchmark(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist_router(-0.1F, 0.1F);
    std::uniform_real_distribution<float> dist_act(-0.05F, 0.05F);

    constexpr std::uint64_t gate_code_bytes_per_expert  = 1'280ULL * 2'560 / 2;
    constexpr std::uint64_t gate_scale_bytes_per_expert = 1'280ULL * 2'560 / 16;
    constexpr std::uint64_t down_code_bytes_per_expert  = 2'560ULL * 640 / 2;
    constexpr std::uint64_t down_scale_bytes_per_expert = 2'560ULL * 640 / 16;

    // Allocate 512 physical experts for prefill test
    ninfer::DeviceBuffer gate_codes(512 * gate_code_bytes_per_expert);
    ninfer::DeviceBuffer gate_scales(512 * gate_scale_bytes_per_expert);
    ninfer::DeviceBuffer down_codes(512 * down_code_bytes_per_expert);
    ninfer::DeviceBuffer down_scales(512 * down_scale_bytes_per_expert);
    ninfer::DeviceBuffer gate_divisors(512 * sizeof(float));
    ninfer::DeviceBuffer down_divisors(512 * sizeof(float));

    std::vector<std::uint8_t> h_gate_codes(512 * gate_code_bytes_per_expert, 0x22U);
    std::vector<std::uint8_t> h_gate_scales(512 * gate_scale_bytes_per_expert, 0x38U);
    std::vector<std::uint8_t> h_down_codes(512 * down_code_bytes_per_expert, 0x22U);
    std::vector<std::uint8_t> h_down_scales(512 * down_scale_bytes_per_expert, 0x38U);
    std::vector<float> h_divisors(512, 1.0F);

    gate_codes.copy_from_host(h_gate_codes.data(), h_gate_codes.size());
    gate_scales.copy_from_host(h_gate_scales.data(), h_gate_scales.size());
    down_codes.copy_from_host(h_down_codes.data(), h_down_codes.size());
    down_scales.copy_from_host(h_down_scales.data(), h_down_scales.size());
    gate_divisors.copy_from_host(h_divisors.data(), sizeof(float) * 512);
    down_divisors.copy_from_host(h_divisors.data(), sizeof(float) * 512);

    ninfer::DeviceBuffer d_router(512ULL * 2'560 * 2);
    ninfer::DeviceBuffer d_shared_down(2'560ULL * 640 * 2);
    ninfer::DeviceBuffer d_shared_gate(640ULL * 2'560 * 2);
    ninfer::DeviceBuffer d_shared_up(640ULL * 2'560 * 2);
    ninfer::DeviceBuffer d_shared_gate_weight(2'560 * 2);

    std::vector<std::uint16_t> h_router(512ULL * 2'560);
    for (auto& v : h_router) { v = float_to_bf16(dist_router(rng)); }
    d_router.copy_from_host(h_router.data(), h_router.size() * 2);

    std::vector<std::uint16_t> h_shared_down(2'560ULL * 640);
    for (auto& v : h_shared_down) { v = float_to_bf16(dist_router(rng)); }
    d_shared_down.copy_from_host(h_shared_down.data(), h_shared_down.size() * 2);

    std::vector<std::uint16_t> h_shared_gate(640ULL * 2'560);
    for (auto& v : h_shared_gate) { v = float_to_bf16(dist_router(rng)); }
    d_shared_gate.copy_from_host(h_shared_gate.data(), h_shared_gate.size() * 2);

    std::vector<std::uint16_t> h_shared_up(640ULL * 2'560);
    for (auto& v : h_shared_up) { v = float_to_bf16(dist_router(rng)); }
    d_shared_up.copy_from_host(h_shared_up.data(), h_shared_up.size() * 2);

    std::vector<std::uint16_t> h_shared_gate_weight(2'560);
    for (auto& v : h_shared_gate_weight) { v = float_to_bf16(dist_router(rng)); }
    d_shared_gate_weight.copy_from_host(h_shared_gate_weight.data(), h_shared_gate_weight.size() * 2);

    MoeWeights weights{
        .router             = bf16_weight(d_router.p, 512, 2'560),
        .shared_down        = bf16_weight(d_shared_down.p, 2'560, 640),
        .shared_gate        = bf16_weight(d_shared_gate.p, 640, 2'560),
        .shared_up          = bf16_weight(d_shared_up.p, 640, 2'560),
        .shared_gate_weight = bf16_weight(d_shared_gate_weight.p, 1, 2'560),
        .expert_gate_up     = {.codes                  = static_cast<const std::byte*>(gate_codes.p),
                               .scales                 = static_cast<const std::byte*>(gate_scales.p),
                               .weight_scale_divisors  = static_cast<const float*>(gate_divisors.p),
                               .experts                = 512,
                               .rows                   = 1'280,
                               .columns                = 2'560,
                               .code_bytes_per_expert  = gate_code_bytes_per_expert,
                               .scale_bytes_per_expert = gate_scale_bytes_per_expert},
        .expert_down        = {.codes                  = static_cast<const std::byte*>(down_codes.p),
                               .scales                 = static_cast<const std::byte*>(down_scales.p),
                               .weight_scale_divisors  = static_cast<const float*>(down_divisors.p),
                               .experts                = 512,
                               .rows                   = 2'560,
                               .columns                = 640,
                               .code_bytes_per_expert  = down_code_bytes_per_expert,
                               .scale_bytes_per_expert = down_scale_bytes_per_expert},
    };

    double max_rel_l2 = 0.0;
    float time_128_us = 0.0F;
    float time_512_us = 0.0F;
    float time_2048_us = 0.0F;

    for (int tokens : {128, 512, 2048}) {
        std::vector<std::uint16_t> h_input(static_cast<std::size_t>(tokens) * 2'560);
        for (auto& v : h_input) { v = float_to_bf16(dist_act(rng)); }

        ninfer::DeviceBuffer d_input(h_input.size() * 2);
        d_input.copy_from_host(h_input.data(), h_input.size() * 2);
        ninfer::DeviceBuffer d_out_prefill(h_input.size() * 2);
        ninfer::DeviceBuffer d_out_decode(h_input.size() * 2);

        ninfer::Tensor in_view(d_input.p, ninfer::DType::BF16, {2'560, tokens});
        ninfer::Tensor out_prefill_view(d_out_prefill.p, ninfer::DType::BF16, {2'560, tokens});
        ninfer::Tensor out_decode_view(d_out_decode.p, ninfer::DType::BF16, {2'560, tokens});

        ninfer::WorkspaceArena ws_prefill(flash_next_moe_workspace_capacity_bytes(1, tokens));
        ninfer::WorkspaceArena ws_decode(flash_next_moe_workspace_capacity_bytes(1, 8));

        // 1. Run grouped prefill
        flash_next_moe(in_view, weights, out_prefill_view, ws_prefill, device.stream);
        device.synchronize();

        // 2. Run decode reference in batches of 8 tokens (decode path T <= 8)
        for (int t = 0; t < tokens; t += 8) {
            const int cur_t = std::min(8, tokens - t);
            ninfer::Tensor in_t(static_cast<__nv_bfloat16*>(d_input.p) + static_cast<std::int64_t>(t) * 2'560,
                                ninfer::DType::BF16, {2'560, cur_t});
            ninfer::Tensor out_t(static_cast<__nv_bfloat16*>(d_out_decode.p) + static_cast<std::int64_t>(t) * 2'560,
                                 ninfer::DType::BF16, {2'560, cur_t});
            flash_next_moe(in_t, weights, out_t, ws_decode, device.stream);
        }
        device.synchronize();

        std::vector<std::uint16_t> act_prefill_bf(h_input.size());
        std::vector<std::uint16_t> act_decode_bf(h_input.size());
        d_out_prefill.copy_to_host(act_prefill_bf.data(), act_prefill_bf.size() * 2);
        d_out_decode.copy_to_host(act_decode_bf.data(), act_decode_bf.size() * 2);

        std::vector<float> act_prefill_f(h_input.size());
        std::vector<float> act_decode_f(h_input.size());
        for (std::size_t i = 0; i < h_input.size(); ++i) {
            act_prefill_f[i] = bf16_to_float(act_prefill_bf[i]);
            act_decode_f[i]  = bf16_to_float(act_decode_bf[i]);
        }

        int nan_count = 0;
        for (std::size_t i = 0; i < act_prefill_f.size(); ++i) {
            if (std::isnan(act_prefill_f[i]) || std::isinf(act_prefill_f[i])) {
                nan_count++;
            }
        }
        if (nan_count > 0) {
            std::cerr << "FAILED: Prefill output contains " << nan_count << " non-finite values!\n";
            return 1;
        }

        double err = relative_l2_error(act_prefill_f, act_decode_f);
        max_rel_l2 = std::max(max_rel_l2, err);
        std::cout << "  Tokens T=" << tokens << " Prefill vs Decode Rel-L2 Error: "
                  << std::scientific << std::setprecision(6) << err << "\n" << std::flush;

        if (tokens < 512) {
            // T < 512 runs SIMT path: should match decode reference tightly
            if (err > 1.0e-3) {
                std::cerr << "FAILED: SIMT prefill T=" << tokens << " rel-L2 error exceeded tolerance 1e-3: " << err << "\n";
                return 1;
            }
        } else {
            // T >= 512 runs Native NVFP4 MMA path: dynamic W4A4 quant introduces ~11-14% Rel-L2 difference
            if (err > 0.20) {
                std::cerr << "FAILED: MMA prefill T=" << tokens << " rel-L2 error exceeded tolerance 0.20: " << err << "\n";
                return 1;
            }
        }

        // Detailed breakdown
        cudaEvent_t ev_start, ev_route, ev_end;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_route);
        cudaEventCreate(&ev_end);

        const auto scope = ws_prefill.scope();
        FlashNextMoeWorkspace scratch = allocate_flash_next_moe_workspace(ws_prefill, tokens);

        flash_next_route(in_view, weights.router, weights.shared_gate_weight, scratch.scores, scratch.ids,
                         scratch.alpha, scratch.shared_scale, device.stream);
        flash_next_moe_kernels_launch(in_view, weights, scratch, out_prefill_view, device.stream);
        device.synchronize();

        if (tokens >= 512) {
            const std::size_t id_bytes =
                static_cast<std::size_t>(tokens) * 10U * sizeof(std::int32_t);
            std::vector<std::int32_t> ids_old(static_cast<std::size_t>(tokens) * 10U);
            std::vector<std::int32_t> ids_new(ids_old.size());
#ifdef _WIN32
            (void)_putenv_s("NINFER_FLASH_NEXT_MOE_SHARED_MMA", "0");
#else
            (void)setenv("NINFER_FLASH_NEXT_MOE_SHARED_MMA", "0", 1);
#endif
            flash_next_route(in_view, weights.router, weights.shared_gate_weight, scratch.scores,
                             scratch.ids, scratch.alpha, scratch.shared_scale, device.stream);
            device.synchronize();
            CUDA_CHECK(cudaMemcpy(ids_old.data(), scratch.ids.data, id_bytes,
                                  cudaMemcpyDeviceToHost));
#ifdef _WIN32
            (void)_putenv_s("NINFER_FLASH_NEXT_MOE_SHARED_MMA", "1");
#else
            (void)setenv("NINFER_FLASH_NEXT_MOE_SHARED_MMA", "1", 1);
#endif
            flash_next_route(in_view, weights.router, weights.shared_gate_weight, scratch.scores,
                             scratch.ids, scratch.alpha, scratch.shared_scale, device.stream);
            device.synchronize();
            CUDA_CHECK(cudaMemcpy(ids_new.data(), scratch.ids.data, id_bytes,
                                  cudaMemcpyDeviceToHost));
            int nonzero = 0;
            int flips   = 0;
            for (std::size_t i = 0; i < ids_old.size(); ++i) {
                if (ids_old[i] != 0) { ++nonzero; }
                if (ids_old[i] != ids_new[i]) { ++flips; }
            }
            if (nonzero == 0) {
                std::cerr << "FAILED: route-id compare vacuous at T=" << tokens << "\n";
                return 1;
            }
            if (flips != 0) {
                std::cerr << "FAILED: MMA router flipped " << flips << " of " << ids_old.size()
                          << " top-10 slots at T=" << tokens << " (nonzero=" << nonzero << ")\n";
                return 1;
            }
            std::cout << "  Tokens T=" << tokens
                      << " router ids bitwise identical vs scalar (slots=" << ids_old.size()
                      << " nonzero=" << nonzero << ")\n";
        }

        const int kIters = (tokens >= 2048) ? 10 : 50;
        cudaEventRecord(ev_start, device.stream);
        for (int i = 0; i < kIters; ++i) {
            flash_next_route(in_view, weights.router, weights.shared_gate_weight, scratch.scores, scratch.ids,
                             scratch.alpha, scratch.shared_scale, device.stream);
        }
        cudaEventRecord(ev_route, device.stream);
        for (int i = 0; i < kIters; ++i) {
            flash_next_moe_kernels_launch(in_view, weights, scratch, out_prefill_view, device.stream);
        }
        cudaEventRecord(ev_end, device.stream);
        cudaEventSynchronize(ev_end);

        float route_ms = 0.0f, compute_ms = 0.0f;
        cudaEventElapsedTime(&route_ms, ev_start, ev_route);
        cudaEventElapsedTime(&compute_ms, ev_route, ev_end);

        float avg_route_us = (route_ms / kIters) * 1000.0f;
        float avg_comp_us  = (compute_ms / kIters) * 1000.0f;
        if (tokens == 128) time_128_us = avg_comp_us;
        if (tokens == 512) time_512_us = avg_comp_us;
        if (tokens == 2048) time_2048_us = avg_comp_us;

        std::cout << "  Tokens T=" << tokens << " Timing Breakdown:\n"
                  << "    Router Time : " << avg_route_us << " us\n"
                  << "    MoE GEMMs   : " << avg_comp_us  << " us\n"
                  << "    Total Layer : " << avg_route_us + avg_comp_us << " us\n";

        if (tokens == 2048) {
            cudaEvent_t ev0, ev1, ev2, ev3, ev4, ev5, ev6;
            cudaEventCreate(&ev0); cudaEventCreate(&ev1); cudaEventCreate(&ev2);
            cudaEventCreate(&ev3); cudaEventCreate(&ev4); cudaEventCreate(&ev5);
            cudaEventCreate(&ev6);

            cudaEventRecord(ev0, device.stream);
            for (int i = 0; i < kIters; ++i) {
                // 1. Build groups
                flash_next_moe_kernels_launch(in_view, weights, scratch, out_prefill_view, device.stream);
            }
            cudaEventRecord(ev6, device.stream);
            cudaEventSynchronize(ev6);
        }
    }

    const float scaling_factor_512  = time_512_us / (time_128_us + 1e-6F);
    const float scaling_factor_2048 = time_2048_us / (time_128_us + 1e-6F);
    const float total_128_ms   = (time_128_us * 48.0F) / 1000.0F;
    const float total_2048_ms  = (time_2048_us * 48.0F) / 1000.0F;

    std::cout << "\n=== Flash-Next Prefill MoE Summary ===\n";
    std::cout << "  T=128 48-layer MoE time  : " << total_128_ms << " ms\n";
    std::cout << "  T=2048 48-layer MoE time : " << total_2048_ms << " ms\n";
    std::cout << "  T=512 vs T=128 scaling   : " << scaling_factor_512 << "x\n";
    std::cout << "  T=2048 vs T=128 scaling  : " << scaling_factor_2048 << "x\n";
    std::cout << "======================================\n" << std::flush;

    std::cout << "PASS: test_prefill_equivalence_and_benchmark\n";
    return 0;
}

} // namespace

// Sequence 14 envelope guard: the MoE workspace envelope is computed once at the chunk capacity,
// but a tail chunk of 9..511 tokens takes the SIMT arm and carves an FP32 [2560, 10, T]
// intermediate instead of the BF16 [2560, 10 * T] staging. Every SIMT tail must fit inside the
// envelope at every capacity. Non-vacuous: at capacity 512 the unpadded staging (26.2 MB) is
// smaller than the T=511 FP32 intermediate (52.3 MB), so this test fails without the pad.
int test_prefill_workspace_envelope_covers_simt_tail() {
    using ninfer::targets::qwen3_8_flash_next::detail::allocate_flash_next_moe_workspace;
    using ninfer::targets::qwen3_8_flash_next::detail::flash_next_moe_workspace_capacity_bytes;
    const std::array<std::int32_t, 5> capacities = {512, 768, 1024, 2048, 4096};
    const std::array<std::int32_t, 4> tails      = {9, 256, 511, 512};
    int checked = 0;
    for (const std::int32_t capacity : capacities) {
        const std::size_t envelope = flash_next_moe_workspace_capacity_bytes(1, capacity);
        for (const std::int32_t tail : tails) {
            const std::int32_t tokens = std::min(tail, capacity);
            ninfer::WorkspaceLayoutBuilder layout;
            (void)allocate_flash_next_moe_workspace(layout, tokens);
            const std::size_t need = layout.peak_bytes(256);
            if (need > envelope) {
                std::cerr << "FAILED: MoE workspace envelope for capacity " << capacity << " ("
                          << envelope << " bytes) does not cover a " << tokens
                          << "-token chunk (" << need << " bytes)\n";
                return 1;
            }
            ++checked;
        }
    }
    if (checked != 20) {
        std::cerr << "FAILED: envelope test checked " << checked << " cases, expected 20\n";
        return 1;
    }
    std::cout << "  envelope covers SIMT tails at capacities 512..4096 (" << checked << " cases)\n";
    return 0;
}

int main() {
    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || device_count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_error);

    ninfer::DeviceContext device(0);

    if (test_basic_unit(device) != 0) {
        std::cerr << "FAILED: test_basic_unit\n";
        return 1;
    }
    std::cout << "PASS: test_decode_basic_unit\n";

    if (test_decode_cobatching_equivalence(device) != 0) {
        std::cerr << "FAILED: test_decode_cobatching_equivalence\n";
        return 1;
    }

    if (test_prefill_workspace_envelope_covers_simt_tail() != 0) {
        std::cerr << "FAILED: test_prefill_workspace_envelope_covers_simt_tail\n";
        return 1;
    }
    std::cout << "PASS: test_prefill_workspace_envelope_covers_simt_tail\n";
    if (test_prefill_equivalence_and_benchmark(device) != 0) {
        std::cerr << "FAILED: test_prefill_equivalence_and_benchmark\n";
        return 1;
    }

    std::cout << "OK Flash-Next MoE\n";
    return 0;
}
