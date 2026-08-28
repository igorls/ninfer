#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "targets/qwen3_8_flash_next/impl/qsa_indexer.h"

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

    constexpr std::int32_t maximum_blocks = 513;
    constexpr std::int32_t logical_pages  = 9;
    ninfer::DeviceContext device(0);
    ninfer::DeviceBuffer input(2'560ULL * 8 * 2);
    ninfer::DeviceBuffer small_t_output(640ULL * 8 * 2);
    ninfer::DeviceBuffer projection(640ULL * 2'560 * 2);
    ninfer::DeviceBuffer query_norm(128 * 2);
    ninfer::DeviceBuffer key_norm(128 * 2);
    ninfer::DeviceBuffer block_keys(128ULL * 64 * logical_pages * 2);
    ninfer::DeviceBuffer block_tables(logical_pages * sizeof(std::int32_t));
    ninfer::DeviceBuffer raw_keys(128ULL * 4 * 2);
    ninfer::DeviceBuffer raw_positions(3ULL * 4 * sizeof(std::int32_t));
    ninfer::DeviceBuffer token_index(sizeof(std::int32_t));
    ninfer::DeviceBuffer mrope_positions(3 * sizeof(std::int32_t));
    ninfer::DeviceBuffer table_row(sizeof(std::int32_t));
    ninfer::DeviceBuffer state_slot(sizeof(std::int32_t));
    ninfer::DeviceBuffer selected_blocks(512 * sizeof(std::int32_t));
    ninfer::DeviceBuffer selected_count(sizeof(std::int32_t));

    std::vector<std::uint16_t> input_values(2'560ULL * 8, 0);
    for (std::int32_t token = 0; token < 8; ++token) {
        input_values[static_cast<std::size_t>(token) * 2'560] = 0x3F80U;
    }
    input.copy_from_host(input_values.data(), input_values.size() * sizeof(std::uint16_t));
    std::vector<std::uint16_t> projection_values(640ULL * 2'560, 0);
    for (std::int32_t row = 0; row < 640; ++row) {
        projection_values[static_cast<std::size_t>(row) * 2'560] = 0x3F80U;
    }
    projection.copy_from_host(projection_values.data(),
                              projection_values.size() * sizeof(std::uint16_t));
    query_norm.fill(0);
    key_norm.fill(0);
    block_keys.fill(0);
    raw_keys.fill(0);
    raw_positions.fill(0);
    std::array<std::int32_t, logical_pages> page_ids{};
    for (std::int32_t page = 0; page < logical_pages; ++page) { page_ids[page] = page; }
    block_tables.copy_from_host(page_ids.data(), sizeof(page_ids));
    constexpr std::array<std::int32_t, 3> zero_positions{};
    mrope_positions.copy_from_host(zero_positions.data(), sizeof(zero_positions));
    constexpr std::int32_t zero = 0;
    table_row.copy_from_host(&zero, sizeof(zero));
    state_slot.copy_from_host(&zero, sizeof(zero));

    AttentionWeights weights{};
    weights.indexer_query_key  = bf16_weight(projection.p, 640, 2'560);
    weights.indexer_query_norm = ninfer::Tensor(query_norm.p, ninfer::DType::BF16, {128});
    weights.indexer_key_norm   = ninfer::Tensor(key_norm.p, ninfer::DType::BF16, {128});
    QsaIndexerCacheView cache{
        .block_keys   = ninfer::Tensor(block_keys.p, ninfer::DType::BF16, {128, 64, logical_pages}),
        .block_tables = ninfer::Tensor(block_tables.p, ninfer::DType::I32, {logical_pages, 1}),
        .raw_keys     = ninfer::Tensor(raw_keys.p, ninfer::DType::BF16, {128, 4, 1}),
        .raw_positions = ninfer::Tensor(raw_positions.p, ninfer::DType::I32, {3, 4, 1}),
    };
    ninfer::Tensor input_view(input.p, ninfer::DType::BF16, {2'560, 1});
    ninfer::Tensor token_view(token_index.p, ninfer::DType::I32, {1});
    ninfer::Tensor position_view(mrope_positions.p, ninfer::DType::I32, {1, 3});
    ninfer::Tensor table_row_view(table_row.p, ninfer::DType::I32, {1});
    ninfer::Tensor state_slot_view(state_slot.p, ninfer::DType::I32, {1});
    ninfer::Tensor selected_view(selected_blocks.p, ninfer::DType::I32, {512, 1});
    ninfer::Tensor count_view(selected_count.p, ninfer::DType::I32, {1});
    ninfer::WorkspaceArena workspace(
        flash_next_qsa_indexer_workspace_capacity_bytes(maximum_blocks, 1));

    token_index.copy_from_host(&zero, sizeof(zero));
    flash_next_qsa_indexer_decode(input_view, weights, token_view, position_view, table_row_view,
                                  state_slot_view, state_slot_view, cache, maximum_blocks, 0,
                                  workspace, selected_view, count_view, device.stream);
    device.synchronize();
    std::int32_t actual_count = -1;
    selected_count.copy_to_host(&actual_count, sizeof(actual_count));
    if (actual_count != 0) {
        std::cerr << "Flash-Next QSA indexer selected a complete block for active_blocks=0\n";
        return 1;
    }

    constexpr std::int32_t final_token = 2'051; // completes block 512, yielding 513 blocks
    token_index.copy_from_host(&final_token, sizeof(final_token));
    flash_next_qsa_indexer_decode(input_view, weights, token_view, position_view, table_row_view,
                                  state_slot_view, state_slot_view, cache, maximum_blocks,
                                  maximum_blocks, workspace, selected_view, count_view,
                                  device.stream);
    device.synchronize();
    selected_count.copy_to_host(&actual_count, sizeof(actual_count));
    std::array<std::int32_t, 512> actual_ids{};
    selected_blocks.copy_to_host(actual_ids.data(), sizeof(actual_ids));
    if (actual_count != 512 || actual_ids.front() != 512) {
        std::cerr << "Flash-Next QSA indexer did not prioritize the positive 513th block\n";
        return 1;
    }
    for (std::int32_t index = 1; index < 512; ++index) {
        if (actual_ids[static_cast<std::size_t>(index)] != index - 1) {
            std::cerr << "Flash-Next QSA stable top-k tie order changed at " << index << '\n';
            return 1;
        }
    }
    std::array<std::uint16_t, 128> compressed_key{};
    constexpr std::size_t block_512_byte_offset = 8ULL * 128 * 64 * 2;
    block_keys.copy_to_host(compressed_key.data(), sizeof(compressed_key), block_512_byte_offset);
    if (!std::all_of(compressed_key.begin(), compressed_key.end(),
                     [](std::uint16_t value) { return value == 0x3F80U; })) {
        std::cerr << "Flash-Next QSA four-token compressed key was not exact BF16 one\n";
        return 1;
    }

    ninfer::Tensor small_t_input(input.p, ninfer::DType::BF16, {2'560, 8});
    ninfer::Tensor small_t_output_view(small_t_output.p, ninfer::DType::BF16, {640, 8});
    ninfer::WorkspaceArena empty_workspace(1);
    if (ninfer::ops::linear_workspace_capacity_bytes(
            ninfer::QType::BF16_CTRL, 640, 2'560, ninfer::ops::LinearPolicy::A16Only, 1, 8) != 0) {
        std::cerr << "Flash-Next QSA BF16 indexer unexpectedly required workspace\n";
        return 1;
    }
    ninfer::ops::linear(small_t_input, weights.indexer_query_key, small_t_output_view,
                        ninfer::ops::LinearPolicy::A16Only, empty_workspace, device.stream);
    device.synchronize();
    std::vector<std::uint16_t> actual_small_t(640ULL * 8);
    small_t_output.copy_to_host(actual_small_t.data(),
                                actual_small_t.size() * sizeof(std::uint16_t));
    if (!std::all_of(actual_small_t.begin(), actual_small_t.end(),
                     [](std::uint16_t value) { return value == 0x3F80U; })) {
        std::cerr << "Flash-Next QSA BF16 T=8 projection was not exact BF16 one\n";
        return 1;
    }
    return 0;
}
