#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/ple_pipeline.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
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

    constexpr std::uint64_t rows         = 1;
    constexpr std::uint64_t width        = 160;
    constexpr std::uint64_t scale_offset = 256;
    std::vector<std::byte> encoded(scale_offset + (width / 16) * 2, std::byte{0});
    std::fill_n(encoded.begin(), width / 2, std::byte{0x88});
    for (std::uint8_t index = 0; index < 8; ++index) {
        encoded[index] = static_cast<std::byte>(index * 2 | ((index * 2 + 1) << 4));
    }
    constexpr std::uint16_t half_point_five = 0x3800;
    for (std::size_t offset = scale_offset; offset < encoded.size(); offset += 2) {
        std::memcpy(encoded.data() + offset, &half_point_five, sizeof(half_point_five));
    }
    PleTableView table;
    for (PleShardView& shard : table.shards) { shard = make_ple_shard_view(encoded, rows, width); }

    std::array<std::array<std::int64_t, 16>, 1> indices{};
    for (std::size_t head = 0; head < 16; ++head) {
        indices[0][head] = static_cast<std::int64_t>(head * kPleRowsPerShard);
    }

    ninfer::DeviceContext device(0);
    ninfer::DeviceBuffer output(2 * 2'560);
    PleGatherPipeline pipeline(table, device, 1);
    auto ticket = pipeline.prepare(indices);
    ninfer::Tensor output_view(output.p, ninfer::DType::BF16, {2'560, 1});
    pipeline.enqueue_copy(std::move(ticket), output_view);
    device.synchronize();

    std::array<std::uint16_t, 2'560> actual{};
    output.copy_to_host(actual.data(), sizeof(actual));
    constexpr std::array<std::uint16_t, 16> expected = {
        0xC080, 0xC060, 0xC040, 0xC020, 0xC000, 0xBFC0, 0xBF80, 0xBF00,
        0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4020, 0x4040, 0x4060,
    };
    for (std::size_t head = 0; head < 16; ++head) {
        for (std::size_t column = 0; column < width; ++column) {
            const std::uint16_t wanted = column < expected.size() ? expected[column] : 0;
            if (actual[head * width + column] != wanted) {
                std::cerr << "asynchronous PLE gather/copy changed represented values\n";
                return 1;
            }
        }
    }
    return 0;
}
