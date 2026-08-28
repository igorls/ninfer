#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/hyper_connection.h"

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
    ninfer::DeviceBuffer hidden(10'240 * 2);
    ninfer::DeviceBuffer norm(10'240 * 2);
    ninfer::DeviceBuffer down(320ULL * 10'240 * 2);
    ninfer::DeviceBuffer up(10'240ULL * 320 * 2);
    ninfer::DeviceBuffer inject(4ULL * 10'240 * 2);
    ninfer::DeviceBuffer block_input(2'560 * 2);
    ninfer::DeviceBuffer block_output(2'560 * 2);
    std::array<std::uint16_t, 10'240> hidden_values{};
    hidden_values.fill(0x3F80U); // 1.0
    hidden.copy_from_host(hidden_values.data(), sizeof(hidden_values));
    norm.fill(0);
    down.fill(0);
    up.fill(0);
    inject.fill(0);
    std::array<std::uint16_t, 2'560> output_values{};
    output_values.fill(0x4000U); // 2.0
    block_output.copy_from_host(output_values.data(), sizeof(output_values));

    HyperConnectionWeights weights{
        .block_inject   = bf16_weight(inject.p, 4, 10'240),
        .norm           = ninfer::Tensor(norm.p, ninfer::DType::BF16, {10'240}),
        .input_mix_down = bf16_weight(down.p, 320, 10'240),
        .input_mix_up   = bf16_weight(up.p, 10'240, 320),
    };
    ninfer::WorkspaceArena workspace(flash_next_hyper_workspace_capacity_bytes(1, 1));
    auto scope                      = workspace.scope();
    FlashNextHyperWorkspace scratch = allocate_flash_next_hyper_workspace(workspace, 1);
    ninfer::Tensor hidden_view(hidden.p, ninfer::DType::BF16, {10'240, 1});
    ninfer::Tensor input_view(block_input.p, ninfer::DType::BF16, {2'560, 1});
    ninfer::Tensor output_view(block_output.p, ninfer::DType::BF16, {2'560, 1});
    flash_next_hyper_prepare(hidden_view, weights, scratch, input_view, device.stream);
    flash_next_hyper_inject(output_view, scratch.injection, hidden_view, device.stream);
    device.synchronize();

    std::array<std::uint16_t, 2'560> actual_input{};
    block_input.copy_to_host(actual_input.data(), sizeof(actual_input));
    if (!std::all_of(actual_input.begin(), actual_input.end(),
                     [](std::uint16_t value) { return value == 0x3F00U; })) {
        std::cerr << "Flash-Next zero-mix hyper input was not exact BF16 0.5\n";
        return 1;
    }
    hidden.copy_to_host(hidden_values.data(), sizeof(hidden_values));
    if (!std::all_of(hidden_values.begin(), hidden_values.end(),
                     [](std::uint16_t value) { return value == 0x4040U; })) {
        std::cerr << "Flash-Next unit injection did not produce exact BF16 3.0\n";
        return 1;
    }

    hidden_values.fill(0x3F80U);
    hidden.copy_from_host(hidden_values.data(), sizeof(hidden_values));
    const HyperMixerWeights mixer{
        .norm           = weights.norm,
        .input_mix_down = weights.input_mix_down,
        .input_mix_up   = weights.input_mix_up,
    };
    flash_next_hyper_mix(hidden_view, mixer, scratch, input_view, device.stream);
    device.synchronize();
    block_input.copy_to_host(actual_input.data(), sizeof(actual_input));
    if (!std::all_of(actual_input.begin(), actual_input.end(),
                     [](std::uint16_t value) { return value == 0x3F00U; })) {
        std::cerr << "Flash-Next final hyper mixer was not exact BF16 0.5\n";
        return 1;
    }
    return 0;
}
