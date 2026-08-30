#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/moe.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
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
    return 0;
}
