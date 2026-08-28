#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/moe_route.h"

#include <cuda_runtime.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

std::uint16_t to_bf16(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
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
    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || device_count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_error);

    constexpr int tokens = 2;
    std::array<float, 513 * tokens> scores{};
    for (int expert = 10; expert < 20; ++expert) {
        scores[expert] = static_cast<float>(expert - 9);
    }
    scores[512] = 2.0F;
    // Token one is an all-zero tie: deterministic order must select lower expert IDs.
    scores[513 + 512] = 0.0F;

    ninfer::DeviceContext device(0);
    ninfer::DeviceBuffer score_device(sizeof(scores));
    ninfer::DeviceBuffer id_device(sizeof(std::int32_t) * 10 * tokens);
    ninfer::DeviceBuffer alpha_device(sizeof(float) * 10 * tokens);
    ninfer::DeviceBuffer shared_device(sizeof(float) * tokens);
    score_device.copy_from_host(scores.data(), sizeof(scores));
    ninfer::Tensor score_view(score_device.p, ninfer::DType::FP32, {513, tokens});
    ninfer::Tensor id_view(id_device.p, ninfer::DType::I32, {10, tokens});
    ninfer::Tensor alpha_view(alpha_device.p, ninfer::DType::FP32, {10, tokens});
    ninfer::Tensor shared_view(shared_device.p, ninfer::DType::FP32, {tokens});
    ninfer::targets::qwen3_8_flash_next::detail::flash_next_route_scores(
        score_view, id_view, alpha_view, shared_view, device.stream);
    device.synchronize();

    std::array<std::int32_t, 10 * tokens> ids{};
    std::array<float, 10 * tokens> alpha{};
    std::array<float, tokens> shared{};
    id_device.copy_to_host(ids.data(), sizeof(ids));
    alpha_device.copy_to_host(alpha.data(), sizeof(alpha));
    shared_device.copy_to_host(shared.data(), sizeof(shared));

    float denominator = 502.0F;
    for (int logit = 1; logit <= 10; ++logit) {
        denominator += std::exp(static_cast<float>(logit));
    }
    for (int rank = 0; rank < 10; ++rank) {
        if (ids[rank] != 19 - rank ||
            std::abs(alpha[rank] - std::exp(static_cast<float>(10 - rank)) / denominator) > 2e-6F) {
            std::cerr << "Flash-Next route changed full-softmax top-10 semantics\n";
            return 1;
        }
        if (ids[10 + rank] != rank || std::abs(alpha[10 + rank] - 1.0F / 512.0F) > 1e-7F) {
            std::cerr << "Flash-Next route changed deterministic tie semantics\n";
            return 1;
        }
    }
    if (std::abs(shared[0] - 1.0F / (1.0F + std::exp(-2.0F))) > 1e-7F ||
        std::abs(shared[1] - 0.5F) > 1e-7F) {
        std::cerr << "Flash-Next route changed the independent shared-expert gate\n";
        return 1;
    }

    std::vector<std::uint16_t> input(2'560 * tokens);
    std::vector<std::uint16_t> router(512 * 2'560);
    std::vector<std::uint16_t> shared_gate(2'560);
    input[0]       = to_bf16(1.0F);
    shared_gate[0] = to_bf16(2.0F);
    for (int expert = 10; expert < 20; ++expert) {
        router[expert * 2'560] = to_bf16(static_cast<float>(expert - 9));
    }
    ninfer::DeviceBuffer input_device(input.size() * 2);
    ninfer::DeviceBuffer router_device(router.size() * 2);
    ninfer::DeviceBuffer gate_device(shared_gate.size() * 2);
    ninfer::DeviceBuffer score_workspace(sizeof(scores));
    input_device.copy_from_host(input.data(), input.size() * 2);
    router_device.copy_from_host(router.data(), router.size() * 2);
    gate_device.copy_from_host(shared_gate.data(), shared_gate.size() * 2);
    ninfer::Tensor input_view(input_device.p, ninfer::DType::BF16, {2'560, tokens});
    ninfer::Tensor score_workspace_view(score_workspace.p, ninfer::DType::FP32, {513, tokens});
    const ninfer::Weight router_view = bf16_weight(router_device.p, 512, 2'560);
    const ninfer::Weight gate_view   = bf16_weight(gate_device.p, 1, 2'560);
    ninfer::targets::qwen3_8_flash_next::detail::flash_next_route(
        input_view, router_view, gate_view, score_workspace_view, id_view, alpha_view, shared_view,
        device.stream);
    device.synchronize();
    std::array<std::int32_t, 10 * tokens> projected_ids{};
    std::array<float, 10 * tokens> projected_alpha{};
    std::array<float, tokens> projected_shared{};
    id_device.copy_to_host(projected_ids.data(), sizeof(projected_ids));
    alpha_device.copy_to_host(projected_alpha.data(), sizeof(projected_alpha));
    shared_device.copy_to_host(projected_shared.data(), sizeof(projected_shared));
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if (projected_ids[index] != ids[index] ||
            std::abs(projected_alpha[index] - alpha[index]) > 2e-6F) {
            std::cerr << "Flash-Next BF16 router projection changed route selection\n";
            return 1;
        }
    }
    if (std::abs(projected_shared[0] - shared[0]) > 1e-7F ||
        std::abs(projected_shared[1] - shared[1]) > 1e-7F) {
        std::cerr << "Flash-Next BF16 shared gate projection changed routing\n";
        return 1;
    }
    return 0;
}
