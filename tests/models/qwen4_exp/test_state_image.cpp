#include "core/device.h"
#include "models/qwen4_exp/state/state_image.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
namespace q4 = ninfer::models::qwen4_exp;

std::vector<ninfer::Tensor> components(q4::StateImageDevicePool& pool, int slot,
                                       std::uint32_t indexer_layers) {
    std::vector<ninfer::Tensor> tensors;
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        tensors.push_back(pool.linear().conv_slot(layer, slot));
        tensors.push_back(pool.linear().recurrent_slot(layer, slot));
    }
    for (std::uint32_t layer = 0; layer < indexer_layers; ++layer) {
        const auto indexer = pool.indexer(layer);
        tensors.push_back(indexer.raw_keys.slice(2, slot, 1));
        tensors.push_back(indexer.raw_positions.slice(2, slot, 1));
    }
    tensors.push_back(pool.ple_history_slot(slot));
    tensors.push_back(pool.continuation_hidden_slot(slot));
    tensors.push_back(pool.continuation_positions_slot(slot));
    return tensors;
}

std::vector<std::uint8_t> pattern(std::size_t bytes, unsigned seed) {
    std::vector<std::uint8_t> out(bytes);
    for (std::size_t i = 0; i < bytes; ++i) {
        out[i] = static_cast<std::uint8_t>((i * 17U + (i >> 8U) + seed * 29U) & 255U);
    }
    return out;
}

void fill(q4::StateImageDevicePool& pool, int slot, unsigned seed) {
    for (const auto& tensor : components(pool, slot, 2)) {
        const auto bytes = pattern(tensor.bytes(), seed++);
        CUDA_CHECK(cudaMemcpy(tensor.data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
    }
}

void check(q4::StateImageDevicePool& pool, int slot, unsigned seed, bool zero = false) {
    for (const auto& tensor : components(pool, slot, 2)) {
        std::vector<std::uint8_t> actual(tensor.bytes());
        CUDA_CHECK(cudaMemcpy(actual.data(), tensor.data, actual.size(), cudaMemcpyDeviceToHost));
        const auto expected =
            zero ? std::vector<std::uint8_t>(actual.size()) : pattern(actual.size(), seed++);
        if (actual != expected) {
            throw std::runtime_error("StateImage transfer changed a component or another slot");
        }
    }
}

void round_trip(ninfer::DeviceContext& device, bool real_shape) {
    q4::StateImageSpec spec{
        .linear      = {.layers         = 2,
                        .conv_channels  = real_shape ? 10240 : 5,
                        .conv_width     = 3,
                        .value_heads    = real_shape ? 48 : 2,
                        .value_head_dim = real_shape ? 128 : 4,
                        .key_head_dim   = real_shape ? 128 : 3,
                        .slot_count     = 4,
                        .conv_dtype     = ninfer::DType::BF16},
        .token_mixer = {.indexer_layers  = 2,
                        .indexer_dim     = real_shape ? 128 : 7,
                        .indexer_block   = 4,
                        .ple_channels    = real_shape ? 10240 : 11,
                        .ple_history     = 9,
                        .transient_slots = 8},
        .hidden      = real_shape ? 10240 : 13,
    };
    ninfer::LayoutBuilder builder;
    const auto layout = q4::plan_state_image_device_pool(builder, spec);
    ninfer::DeviceBuffer backing(builder.finish(256));
    q4::StateImageDevicePool pool({backing.p, backing.bytes}, layout);
    ninfer::PinnedHostBuffer host(layout.host.image_bytes);
    auto* host_bytes = static_cast<std::byte*>(host.data());

    for (int slot = 0; slot < 4; ++slot) { fill(pool, slot, 20U * (slot + 1)); }
    pool.copy_to_host(0, {host_bytes, &layout.host}, device.stream);
    device.synchronize();
    // Destroy every Device copy of the saved state before restoring into a different slot.
    pool.zero_slot(0, device.stream);
    pool.copy_from_host({host_bytes, &layout.host}, 2, device.stream);
    pool.copy_slot(2, 3, device.stream);
    device.synchronize();
    check(pool, 0, 0, true);
    check(pool, 1, 40);
    check(pool, 2, 20);
    check(pool, 3, 20);

    auto wrong_layout = layout.host;
    ++wrong_layout.spec.token_mixer.ple_history;
    bool rejected = false;
    try {
        pool.copy_from_host({host_bytes, &wrong_layout}, 1, device.stream);
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) { throw std::runtime_error("incompatible Host state was accepted"); }
    device.synchronize();
    check(pool, 1, 40);
    pool.zero_all(device.stream);
    device.synchronize();
    for (int slot = 0; slot < 4; ++slot) { check(pool, slot, 0, true); }
}
} // namespace

int main() {
    int count         = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        CUDA_CHECK(status);
        ninfer::DeviceContext device(0);
        round_trip(device, false);
        round_trip(device, true);
        std::cout << "PASS: complete StateImage transfer, restore, isolation and reset\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
