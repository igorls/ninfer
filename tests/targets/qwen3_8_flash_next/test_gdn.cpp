#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/gdn.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
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

std::uint64_t fp8_payload_bytes(std::int32_t rows, std::int32_t columns) {
    const std::uint64_t codes = static_cast<std::uint64_t>(rows) * columns;
    return ((codes + 255U) & ~std::uint64_t{255U}) + static_cast<std::uint64_t>(rows) * 4;
}

float bf16_to_float(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

ninfer::Weight fp8_f32_weight(ninfer::DeviceBuffer& storage, std::int32_t rows,
                              std::int32_t columns) {
    const std::uint64_t codes        = static_cast<std::uint64_t>(rows) * columns;
    const std::uint64_t scale_offset = (codes + 255U) & ~std::uint64_t{255U};
    storage.fill(0);
    std::vector<float> scales(static_cast<std::size_t>(rows), 1.0F);
    storage.copy_from_host(scales.data(), scales.size() * sizeof(float), scale_offset);

    auto* payload = static_cast<std::byte*>(storage.p);
    ninfer::Weight out{};
    out.payload         = payload;
    out.payload_bytes   = storage.bytes;
    out.qdata           = payload;
    out.scales          = payload + scale_offset;
    out.qtype           = ninfer::QType::FP8_E4M3FN_ROW_F32S;
    out.layout          = ninfer::QuantLayout::RowScale;
    out.scale_dtype     = ninfer::DType::FP32;
    out.group_size      = static_cast<std::uint32_t>(columns);
    out.group           = columns;
    out.n               = rows;
    out.k               = columns;
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    out.scale_ne[0]     = rows;
    out.scale_nb[0]     = 4;
    out.scale_nb[1]     = static_cast<std::int64_t>(rows) * 4;
    out.scale_nb[2]     = out.scale_nb[1];
    out.scale_nb[3]     = out.scale_nb[1];
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
    ninfer::DeviceBuffer a_log(48 * 2);
    ninfer::DeviceBuffer convolution(10'240ULL * 4 * 2);
    ninfer::DeviceBuffer dt_bias(48 * 2);
    ninfer::DeviceBuffer a_b_projection(96ULL * 2'560 * 2);
    ninfer::DeviceBuffer norm(128 * 2);
    ninfer::DeviceBuffer qkvz(fp8_payload_bytes(16'384, 2'560));
    ninfer::DeviceBuffer output_projection(fp8_payload_bytes(2'560, 6'144));
    ninfer::DeviceBuffer source_slots(sizeof(std::int32_t));
    ninfer::DeviceBuffer destination_slots(sizeof(std::int32_t));
    ninfer::DeviceBuffer convolution_states(10'240ULL * 3 * 2 * 2);
    ninfer::DeviceBuffer ssm_states(128ULL * 128 * 48 * 2 * sizeof(float));

    input.fill(0);
    output.fill(0xFF);
    a_log.fill(0);
    convolution.fill(0);
    dt_bias.fill(0);
    a_b_projection.fill(0);
    norm.fill(0);
    ssm_states.fill(0);
    const std::int32_t source      = 0;
    const std::int32_t destination = 1;
    source_slots.copy_from_host(&source, sizeof(source));
    destination_slots.copy_from_host(&destination, sizeof(destination));

    std::vector<std::uint16_t> convolution_state_values(10'240ULL * 3 * 2, 0x7FC1U);
    std::fill_n(convolution_state_values.begin(), 10'240, 0x3F80U);          // source oldest: 1
    std::fill_n(convolution_state_values.begin() + 10'240, 10'240, 0x4000U); // source: 2
    std::fill_n(convolution_state_values.begin() + 20'480, 10'240, 0x4040U); // source: 3
    convolution_states.copy_from_host(convolution_state_values.data(),
                                      convolution_state_values.size() * sizeof(std::uint16_t));

    GdnWeights weights{
        .a_log             = ninfer::Tensor(a_log.p, ninfer::DType::BF16, {48}),
        .convolution       = ninfer::Tensor(convolution.p, ninfer::DType::BF16, {10'240, 4}),
        .dt_bias           = ninfer::Tensor(dt_bias.p, ninfer::DType::BF16, {48}),
        .a_b_projection    = bf16_weight(a_b_projection.p, 96, 2'560),
        .norm              = ninfer::Tensor(norm.p, ninfer::DType::BF16, {128}),
        .query_key_value_z = fp8_f32_weight(qkvz, 16'384, 2'560),
        .output            = fp8_f32_weight(output_projection, 2'560, 6'144),
    };
    ninfer::Tensor input_view(input.p, ninfer::DType::BF16, {2'560, 1});
    ninfer::Tensor output_view(output.p, ninfer::DType::BF16, {2'560, 1});
    ninfer::Tensor source_view(source_slots.p, ninfer::DType::I32, {1});
    ninfer::Tensor destination_view(destination_slots.p, ninfer::DType::I32, {1});
    ninfer::Tensor convolution_state_view(convolution_states.p, ninfer::DType::BF16,
                                          {10'240, 3, 2});
    ninfer::Tensor ssm_state_view(ssm_states.p, ninfer::DType::FP32, {128, 128, 48, 2});
    ninfer::WorkspaceArena workspace(flash_next_gdn_workspace_capacity_bytes(1, 1));
    flash_next_gdn_decode(input_view, weights, source_view, destination_view,
                          convolution_state_view, ssm_state_view, workspace, output_view,
                          device.stream);
    device.synchronize();

    std::array<std::uint16_t, 2'560> actual_output{};
    output.copy_to_host(actual_output.data(), sizeof(actual_output));
    if (!std::all_of(actual_output.begin(), actual_output.end(),
                     [](std::uint16_t value) { return value == 0; })) {
        std::cerr << "Flash-Next zero GDN did not produce exact BF16 zero\n";
        return 1;
    }

    convolution_states.copy_to_host(convolution_state_values.data(),
                                    convolution_state_values.size() * sizeof(std::uint16_t));
    const auto destination_begin = convolution_state_values.begin() + 30'720;
    if (!std::all_of(destination_begin, destination_begin + 10'240,
                     [](std::uint16_t value) { return value == 0x4000U; }) ||
        !std::all_of(destination_begin + 10'240, destination_begin + 20'480,
                     [](std::uint16_t value) { return value == 0x4040U; }) ||
        !std::all_of(destination_begin + 20'480, destination_begin + 30'720,
                     [](std::uint16_t value) { return value == 0; })) {
        std::cerr << "Flash-Next GDN did not publish the four-tap convolution state transition\n";
        return 1;
    }

    // Exercise the complete nonzero boundary. Each projection row reads input column zero with
    // exact FP8 1.0; the causal current tap is BF16 1.0. The resulting represented q/k/v is
    // BF16 SiLU(1), so the independently evaluated first state element is beta*v*k_normalized.
    std::array<std::uint16_t, 2'560> nonzero_input{};
    nonzero_input.front() = 0x3F80U;
    input.copy_from_host(nonzero_input.data(), sizeof(nonzero_input));
    output.fill(0xFF);
    std::array<std::uint16_t, 128> norm_values{};
    norm_values.fill(0x3F80U); // RMSNormGated owns a direct, ones-initialized weight.
    norm.copy_from_host(norm_values.data(), sizeof(norm_values));
    convolution_states.fill(0);
    ssm_states.fill(0);
    std::vector<std::uint16_t> convolution_values(10'240ULL * 4, 0);
    std::fill(convolution_values.begin() + 30'720, convolution_values.end(), 0x3F80U);
    convolution.copy_from_host(convolution_values.data(),
                               convolution_values.size() * sizeof(std::uint16_t));
    std::vector<std::uint8_t> qkvz_codes(16'384ULL * 2'560, 0);
    for (std::int32_t row = 0; row < 16'384; ++row) {
        qkvz_codes[static_cast<std::size_t>(row) * 2'560] = 0x38U; // E4M3 1.0
    }
    qkvz.copy_from_host(qkvz_codes.data(), qkvz_codes.size());
    std::vector<std::uint8_t> output_codes(2'560ULL * 6'144, 0);
    for (std::int32_t row = 0; row < 2'560; ++row) {
        output_codes[static_cast<std::size_t>(row) * 6'144] = 0x38U;
    }
    output_projection.copy_from_host(output_codes.data(), output_codes.size());

    flash_next_gdn_decode(input_view, weights, source_view, destination_view,
                          convolution_state_view, ssm_state_view, workspace, output_view,
                          device.stream);
    device.synchronize();

    float actual_state                     = 0.0F;
    constexpr std::size_t state_slot_bytes = 128ULL * 128 * 48 * sizeof(float);
    ssm_states.copy_to_host(&actual_state, sizeof(actual_state), state_slot_bytes);
    const float activated      = 1.0F / (1.0F + std::exp(-1.0F));
    const float normalized_key = activated / std::sqrt(128.0F * activated * activated + 1.0e-6F);
    const float expected_state = 0.5F * activated * normalized_key;
    if (std::abs(actual_state - expected_state) > 2.0e-4F) {
        std::cerr << "Flash-Next GDN recurrent state disagreed with the independent formula: "
                  << actual_state << " vs " << expected_state << '\n';
        return 1;
    }

    output.copy_to_host(actual_output.data(), sizeof(actual_output));
    const float first_output = bf16_to_float(actual_output.front());
    if (first_output < 0.70F || first_output > 0.75F ||
        !std::all_of(actual_output.begin(), actual_output.end(),
                     [&](std::uint16_t value) { return value == actual_output.front(); })) {
        std::cerr << "Flash-Next nonzero GDN output was outside its represented gated-norm range: "
                  << first_output << '\n';
        return 1;
    }
    return 0;
}
