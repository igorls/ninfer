#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "targets/qwen3_8_flash_next/impl/qsa_attention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

ninfer::Weight fp8_f32_weight(ninfer::DeviceBuffer& storage, std::int32_t rows,
                              std::int32_t columns) {
    const std::uint64_t codes        = static_cast<std::uint64_t>(rows) * columns;
    const std::uint64_t scale_offset = (codes + 255U) & ~std::uint64_t{255U};
    auto* payload                    = static_cast<std::byte*>(storage.p);
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

std::uint64_t fp8_payload_bytes(std::int32_t rows, std::int32_t columns) {
    const std::uint64_t codes = static_cast<std::uint64_t>(rows) * columns;
    return ((codes + 255U) & ~std::uint64_t{255U}) + static_cast<std::uint64_t>(rows) * 4;
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
    constexpr std::int32_t batch          = 1;
    constexpr std::int32_t physical_pages = 2;
    constexpr std::int32_t logical_pages  = 2;
    constexpr std::int32_t input_dim      = 2'560;
    constexpr std::int32_t proj_rows      = 13'312;
    constexpr std::int32_t out_rows       = 2'560;
    constexpr std::int32_t out_cols       = 6'144;

    ninfer::DeviceBuffer input(input_dim * batch * sizeof(std::uint16_t));
    ninfer::DeviceBuffer output(input_dim * batch * sizeof(std::uint16_t));
    ninfer::DeviceBuffer query_norm(256 * sizeof(std::uint16_t));
    ninfer::DeviceBuffer key_norm(256 * sizeof(std::uint16_t));
    ninfer::DeviceBuffer qgkv_storage(fp8_payload_bytes(proj_rows, input_dim));
    ninfer::DeviceBuffer out_storage(fp8_payload_bytes(out_rows, out_cols));
    ninfer::DeviceBuffer token_indices(batch * sizeof(std::int32_t));
    ninfer::DeviceBuffer mrope_positions(batch * 3 * sizeof(std::int32_t));
    ninfer::DeviceBuffer table_rows(batch * sizeof(std::int32_t));
    ninfer::DeviceBuffer selected_blocks(512 * batch * sizeof(std::int32_t));
    ninfer::DeviceBuffer selected_counts(batch * sizeof(std::int32_t));
    ninfer::DeviceBuffer key_pages(256ULL * 64 * 2 * physical_pages * sizeof(std::uint16_t));
    ninfer::DeviceBuffer value_pages(256ULL * 64 * 2 * physical_pages * sizeof(std::uint16_t));
    ninfer::DeviceBuffer block_tables(logical_pages * sizeof(std::int32_t));

    // 1. Input: first BF16 value 1.0 (0x3F80), remaining 0
    std::vector<std::uint16_t> host_input(input_dim * batch, 0);
    host_input[0] = 0x3F80U;
    input.copy_from_host(host_input.data(), host_input.size() * sizeof(std::uint16_t));
    output.fill(0);

    // 2. Query & Key norm: zero norm deltas
    query_norm.fill(0);
    key_norm.fill(0);

    // 3. QGKV FP8 weight:
    // Rows: per query head [q256, gate256] for 24 heads (12288 rows),
    // then key [2, 256] (rows 12288..12799), then value [2, 256] (rows 12800..13311).
    // q, k, v rows have W[row, 0] = 0x38 (1.0 in FP8 E4M3); gate rows have W[row, 0] = 0x00 (0.0).
    // All other column weights are 0x00. All row scales are 1.0F.
    const std::uint64_t qgkv_codes        = static_cast<std::uint64_t>(proj_rows) * input_dim;
    const std::uint64_t qgkv_scale_offset = (qgkv_codes + 255U) & ~std::uint64_t{255U};
    std::vector<std::uint8_t> host_qgkv_codes(qgkv_codes, 0);
    for (std::int32_t head = 0; head < 24; ++head) {
        for (std::int32_t dim = 0; dim < 256; ++dim) {
            const std::int32_t q_row                                     = head * 512 + dim;
            host_qgkv_codes[static_cast<std::size_t>(q_row) * input_dim] = 0x38U;
            // gate row is head * 512 + 256 + dim, stays 0x00
        }
    }
    for (std::int32_t r = 12'288; r < 13'312; ++r) {
        host_qgkv_codes[static_cast<std::size_t>(r) * input_dim] = 0x38U;
    }
    std::vector<float> host_qgkv_scales(proj_rows, 1.0F);
    qgkv_storage.fill(0);
    qgkv_storage.copy_from_host(host_qgkv_codes.data(), host_qgkv_codes.size());
    qgkv_storage.copy_from_host(host_qgkv_scales.data(), host_qgkv_scales.size() * sizeof(float),
                                qgkv_scale_offset);

    // 4. Output FP8 weight [2560, 6144]:
    // Map the first gated attended channel (channel 0) to every output row:
    // W[row, 0] = 0x38 (1.0 in FP8 E4M3), W[row, col > 0] = 0. Scale = 1.0F.
    const std::uint64_t out_codes        = static_cast<std::uint64_t>(out_rows) * out_cols;
    const std::uint64_t out_scale_offset = (out_codes + 255U) & ~std::uint64_t{255U};
    std::vector<std::uint8_t> host_out_codes(out_codes, 0);
    for (std::int32_t r = 0; r < out_rows; ++r) {
        host_out_codes[static_cast<std::size_t>(r) * out_cols] = 0x38U;
    }
    std::vector<float> host_out_scales(out_rows, 1.0F);
    out_storage.fill(0);
    out_storage.copy_from_host(host_out_codes.data(), host_out_codes.size());
    out_storage.copy_from_host(host_out_scales.data(), host_out_scales.size() * sizeof(float),
                               out_scale_offset);

    // 5. Positions, token index, table rows, selected blocks
    token_indices.fill(0); // token 0
    mrope_positions.fill(0);
    table_rows.fill(0);
    selected_blocks.fill(0);
    selected_counts.fill(0);

    // 6. Cache pages & block tables
    key_pages.fill(0);
    value_pages.fill(0);
    std::array<std::int32_t, logical_pages> host_block_tables = {0, 1};
    block_tables.copy_from_host(host_block_tables.data(), sizeof(host_block_tables));

    // Construct AttentionWeights & CacheView
    AttentionWeights weights{};
    weights.query_norm           = ninfer::Tensor(query_norm.p, ninfer::DType::BF16, {256});
    weights.key_norm             = ninfer::Tensor(key_norm.p, ninfer::DType::BF16, {256});
    weights.query_gate_key_value = fp8_f32_weight(qgkv_storage, proj_rows, input_dim);
    weights.output               = fp8_f32_weight(out_storage, out_rows, out_cols);

    QsaAttentionCacheView cache{
        .key_pages = ninfer::Tensor(key_pages.p, ninfer::DType::BF16, {256, 64, 2, physical_pages}),
        .value_pages =
            ninfer::Tensor(value_pages.p, ninfer::DType::BF16, {256, 64, 2, physical_pages}),
        .block_tables = ninfer::Tensor(block_tables.p, ninfer::DType::I32, {logical_pages, 1}),
    };

    ninfer::Tensor input_tensor(input.p, ninfer::DType::BF16, {input_dim, batch});
    ninfer::Tensor output_tensor(output.p, ninfer::DType::BF16, {input_dim, batch});
    ninfer::Tensor token_indices_tensor(token_indices.p, ninfer::DType::I32, {batch});
    ninfer::Tensor mrope_positions_tensor(mrope_positions.p, ninfer::DType::I32, {batch, 3});
    ninfer::Tensor table_rows_tensor(table_rows.p, ninfer::DType::I32, {batch});
    ninfer::Tensor selected_blocks_tensor(selected_blocks.p, ninfer::DType::I32, {512, batch});
    ninfer::Tensor selected_counts_tensor(selected_counts.p, ninfer::DType::I32, {batch});

    // Execute QSA attention decode
    ninfer::WorkspaceArena workspace(flash_next_qsa_attention_workspace_capacity_bytes(batch));
    flash_next_qsa_attention_decode(input_tensor, weights, token_indices_tensor,
                                    mrope_positions_tensor, table_rows_tensor,
                                    selected_blocks_tensor, selected_counts_tensor, cache,
                                    workspace, output_tensor, device.stream);
    device.synchronize();

    // Verify output: every output BF16 must be 0.5 (0x3F00)
    std::vector<std::uint16_t> host_output(input_dim * batch);
    output.copy_to_host(host_output.data(), host_output.size() * sizeof(std::uint16_t));
    int failures = 0;
    for (std::size_t i = 0; i < host_output.size(); ++i) {
        if (host_output[i] != 0x3F00U) {
            std::cerr << "Mismatch at output[" << i << "]: expected 0x3F00 (0.5), got 0x"
                      << std::hex << host_output[i] << std::dec << "\n";
            failures++;
            if (failures > 10) break;
        }
    }

    // Verify appended key and value cache:
    // Physical page 0, token 0: both KV heads, 256 dims must be 0x3F80 (1.0)
    std::vector<std::uint16_t> host_key_pages(256ULL * 64 * 2 * physical_pages);
    std::vector<std::uint16_t> host_value_pages(256ULL * 64 * 2 * physical_pages);
    key_pages.copy_to_host(host_key_pages.data(), host_key_pages.size() * sizeof(std::uint16_t));
    value_pages.copy_to_host(host_value_pages.data(),
                             host_value_pages.size() * sizeof(std::uint16_t));

    for (int kv_head = 0; kv_head < 2; ++kv_head) {
        for (int dim = 0; dim < 256; ++dim) {
            // Layout: [256, 64, 2, physical_pages]
            // index: ((page * 2 + head) * 64 + token) * 256 + dim
            const std::size_t idx = ((0 * 2 + kv_head) * 64 + 0) * 256 + dim;
            if (host_key_pages[idx] != 0x3F80U) {
                std::cerr << "Key cache mismatch at head " << kv_head << ", dim " << dim
                          << ": expected 0x3F80 (1.0), got 0x" << std::hex << host_key_pages[idx]
                          << std::dec << "\n";
                failures++;
                if (failures > 20) break;
            }
            if (host_value_pages[idx] != 0x3F80U) {
                std::cerr << "Value cache mismatch at head " << kv_head << ", dim " << dim
                          << ": expected 0x3F80 (1.0), got 0x" << std::hex << host_value_pages[idx]
                          << std::dec << "\n";
                failures++;
                if (failures > 20) break;
            }
        }
    }

    // Verify untouched cache entries remain 0
    for (std::size_t i = 0; i < host_key_pages.size(); ++i) {
        // page 0 token 0 is indices in [0, 256) for head 0 and [64*256, 64*256+256) for head 1
        const bool is_head0_tok0 = (i < 256);
        const bool is_head1_tok0 = (i >= 64 * 256 && i < 64 * 256 + 256);
        if (!is_head0_tok0 && !is_head1_tok0) {
            if (host_key_pages[i] != 0) {
                std::cerr << "Untouched key cache dirty at " << i << ": 0x" << std::hex
                          << host_key_pages[i] << std::dec << "\n";
                failures++;
                break;
            }
            if (host_value_pages[i] != 0) {
                std::cerr << "Untouched value cache dirty at " << i << ": 0x" << std::hex
                          << host_value_pages[i] << std::dec << "\n";
                failures++;
                break;
            }
        }
    }

    if (failures != 0) {
        std::cerr << "FAIL: " << failures << " errors in test_qsa_attention\n";
        return 1;
    }

    std::cout << "PASS: test_qsa_attention\n";
    return 0;
}
