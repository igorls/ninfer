#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "targets/qwen3_8_flash_next/impl/qsa_attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
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

bool test_prefill_vs_decode_equivalence(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t physical_pages = 64;
    constexpr std::int32_t logical_pages  = 64;
    constexpr std::int32_t input_dim      = 2'560;
    constexpr std::int32_t proj_rows      = 13'312;
    constexpr std::int32_t out_rows       = 2'560;
    constexpr std::int32_t out_cols       = 6'144;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const std::uint64_t qgkv_codes        = static_cast<std::uint64_t>(proj_rows) * input_dim;
    const std::uint64_t qgkv_scale_offset = (qgkv_codes + 255U) & ~std::uint64_t{255U};
    std::vector<std::uint8_t> host_qgkv_codes(qgkv_codes);
    for (auto& v : host_qgkv_codes) { v = static_cast<std::uint8_t>(rng() & 0x7F); }
    std::vector<float> host_qgkv_scales(proj_rows, 0.05f);

    ninfer::DeviceBuffer qgkv_storage(fp8_payload_bytes(proj_rows, input_dim));
    qgkv_storage.copy_from_host(host_qgkv_codes.data(), host_qgkv_codes.size());
    qgkv_storage.copy_from_host(host_qgkv_scales.data(), host_qgkv_scales.size() * sizeof(float),
                                qgkv_scale_offset);

    const std::uint64_t out_codes        = static_cast<std::uint64_t>(out_rows) * out_cols;
    const std::uint64_t out_scale_offset = (out_codes + 255U) & ~std::uint64_t{255U};
    std::vector<std::uint8_t> host_out_codes(out_codes);
    for (auto& v : host_out_codes) { v = static_cast<std::uint8_t>(rng() & 0x7F); }
    std::vector<float> host_out_scales(out_rows, 0.05f);

    ninfer::DeviceBuffer out_storage(fp8_payload_bytes(out_rows, out_cols));
    out_storage.copy_from_host(host_out_codes.data(), host_out_codes.size());
    out_storage.copy_from_host(host_out_scales.data(), host_out_scales.size() * sizeof(float),
                               out_scale_offset);

    ninfer::DeviceBuffer query_norm(256 * sizeof(std::uint16_t));
    ninfer::DeviceBuffer key_norm(256 * sizeof(std::uint16_t));
    std::vector<std::uint16_t> host_qnorm(256), host_knorm(256);
    for (auto& v : host_qnorm) { v = __float2bfloat16_rn(dist(rng) * 0.1f); }
    for (auto& v : host_knorm) { v = __float2bfloat16_rn(dist(rng) * 0.1f); }
    query_norm.copy_from_host(host_qnorm.data(), host_qnorm.size() * 2);
    key_norm.copy_from_host(host_knorm.data(), host_knorm.size() * 2);

    AttentionWeights weights{};
    weights.query_norm           = ninfer::Tensor(query_norm.p, ninfer::DType::BF16, {256});
    weights.key_norm             = ninfer::Tensor(key_norm.p, ninfer::DType::BF16, {256});
    weights.query_gate_key_value = fp8_f32_weight(qgkv_storage, proj_rows, input_dim);
    weights.output               = fp8_f32_weight(out_storage, out_rows, out_cols);

    const std::vector<std::int32_t> test_t = {1, 2, 4, 8, 16, 64};
    for (std::int32_t T : test_t) {
        const std::int32_t first_token_index = 64;

        std::vector<std::uint16_t> host_input(static_cast<std::size_t>(input_dim) * T);
        for (auto& v : host_input) { v = __float2bfloat16_rn(dist(rng)); }

        std::vector<std::int32_t> host_pos(3 * T);
        for (std::int32_t t = 0; t < T; ++t) {
            host_pos[0 * T + t] = first_token_index + t;
            host_pos[1 * T + t] = first_token_index + t;
            host_pos[2 * T + t] = first_token_index + t;
        }

        std::vector<std::int32_t> host_selected_blocks(512 * T, -1);
        std::vector<std::int32_t> host_selected_counts(T);
        for (std::int32_t t = 0; t < T; ++t) {
            const int complete_blocks = (first_token_index + t + 1) / 4;
            const int count           = std::min(complete_blocks, 512);
            host_selected_counts[t]   = count;
            for (int k = 0; k < count; ++k) { host_selected_blocks[t * 512 + k] = k; }
        }

        ninfer::DeviceBuffer key_pages_a(256ULL * 64 * 2 * physical_pages * 2);
        ninfer::DeviceBuffer value_pages_a(256ULL * 64 * 2 * physical_pages * 2);
        ninfer::DeviceBuffer block_tables(logical_pages * sizeof(std::int32_t));
        key_pages_a.fill(0);
        value_pages_a.fill(0);
        std::array<std::int32_t, logical_pages> page_ids{};
        for (std::int32_t p = 0; p < logical_pages; ++p) { page_ids[p] = p; }
        block_tables.copy_from_host(page_ids.data(), sizeof(page_ids));

        QsaAttentionCacheView cache_a{
            .key_pages =
                ninfer::Tensor(key_pages_a.p, ninfer::DType::BF16, {256, 64, 2, physical_pages}),
            .value_pages =
                ninfer::Tensor(value_pages_a.p, ninfer::DType::BF16, {256, 64, 2, physical_pages}),
            .block_tables = ninfer::Tensor(block_tables.p, ninfer::DType::I32, {logical_pages, 1}),
        };

        ninfer::DeviceBuffer key_pages_b(256ULL * 64 * 2 * physical_pages * 2);
        ninfer::DeviceBuffer value_pages_b(256ULL * 64 * 2 * physical_pages * 2);
        key_pages_b.fill(0);
        value_pages_b.fill(0);
        QsaAttentionCacheView cache_b{
            .key_pages =
                ninfer::Tensor(key_pages_b.p, ninfer::DType::BF16, {256, 64, 2, physical_pages}),
            .value_pages =
                ninfer::Tensor(value_pages_b.p, ninfer::DType::BF16, {256, 64, 2, physical_pages}),
            .block_tables = ninfer::Tensor(block_tables.p, ninfer::DType::I32, {logical_pages, 1}),
        };

        ninfer::DeviceBuffer input_dev(static_cast<std::size_t>(input_dim) * T * 2);
        input_dev.copy_from_host(host_input.data(), host_input.size() * 2);

        // Path A: Sequential decode
        std::vector<std::uint16_t> seq_output(static_cast<std::size_t>(input_dim) * T);
        ninfer::DeviceBuffer token_buf(sizeof(std::int32_t));
        ninfer::DeviceBuffer pos_buf(3 * sizeof(std::int32_t));
        ninfer::DeviceBuffer row_buf(sizeof(std::int32_t));
        ninfer::DeviceBuffer sel_buf(512 * sizeof(std::int32_t));
        ninfer::DeviceBuffer cnt_buf(sizeof(std::int32_t));
        ninfer::DeviceBuffer out_slice(input_dim * sizeof(std::uint16_t));
        std::int32_t zero = 0;
        row_buf.copy_from_host(&zero, sizeof(zero));

        ninfer::WorkspaceArena ws_decode(flash_next_qsa_attention_workspace_capacity_bytes(1));

        for (std::int32_t t = 0; t < T; ++t) {
            std::int32_t tok_idx = first_token_index + t;
            std::array<std::int32_t, 3> pos_t = {host_pos[0 * T + t], host_pos[1 * T + t],
                                                 host_pos[2 * T + t]};
            token_buf.copy_from_host(&tok_idx, sizeof(tok_idx));
            pos_buf.copy_from_host(pos_t.data(), sizeof(pos_t));
            sel_buf.copy_from_host(&host_selected_blocks[t * 512], 512 * sizeof(std::int32_t));
            cnt_buf.copy_from_host(&host_selected_counts[t], sizeof(std::int32_t));

            ninfer::Tensor in_t(
                static_cast<std::uint16_t*>(input_dev.p) + static_cast<std::size_t>(t) * input_dim,
                ninfer::DType::BF16, {input_dim, 1});
            ninfer::Tensor out_t(out_slice.p, ninfer::DType::BF16, {input_dim, 1});
            ninfer::Tensor tok_t(token_buf.p, ninfer::DType::I32, {1});
            ninfer::Tensor pos_t_view(pos_buf.p, ninfer::DType::I32, {1, 3});
            ninfer::Tensor row_view(row_buf.p, ninfer::DType::I32, {1});
            ninfer::Tensor sel_view(sel_buf.p, ninfer::DType::I32, {512, 1});
            ninfer::Tensor cnt_view(cnt_buf.p, ninfer::DType::I32, {1});

            flash_next_qsa_attention_decode(in_t, weights, tok_t, pos_t_view, row_view, sel_view,
                                            cnt_view, cache_a, ws_decode, out_t, device.stream);
            device.synchronize();
            out_slice.copy_to_host(&seq_output[static_cast<std::size_t>(t) * input_dim],
                                   input_dim * 2);
        }

        // Path B: Chunked prefill
        ninfer::DeviceBuffer dev_indices(T * sizeof(std::int32_t));
        std::vector<std::int32_t> host_indices(T);
        for (std::int32_t t = 0; t < T; ++t) { host_indices[t] = first_token_index + t; }
        dev_indices.copy_from_host(host_indices.data(), T * sizeof(std::int32_t));

        ninfer::DeviceBuffer dev_mrope_pos(3 * T * sizeof(std::int32_t));
        dev_mrope_pos.copy_from_host(host_pos.data(), host_pos.size() * sizeof(std::int32_t));

        ninfer::DeviceBuffer dev_sel(512 * T * sizeof(std::int32_t));
        ninfer::DeviceBuffer dev_cnt(T * sizeof(std::int32_t));
        dev_sel.copy_from_host(host_selected_blocks.data(),
                               host_selected_blocks.size() * sizeof(std::int32_t));
        dev_cnt.copy_from_host(host_selected_counts.data(),
                               host_selected_counts.size() * sizeof(std::int32_t));

        ninfer::DeviceBuffer chunk_out(static_cast<std::size_t>(input_dim) * T * 2);
        ninfer::Tensor chunk_in(input_dev.p, ninfer::DType::BF16, {input_dim, T});
        ninfer::Tensor chunk_out_view(chunk_out.p, ninfer::DType::BF16, {input_dim, T});
        ninfer::Tensor chunk_idx(dev_indices.p, ninfer::DType::I32, {T});
        ninfer::Tensor chunk_pos(dev_mrope_pos.p, ninfer::DType::I32, {T, 3});
        ninfer::Tensor chunk_sel(dev_sel.p, ninfer::DType::I32, {512, T});
        ninfer::Tensor chunk_cnt(dev_cnt.p, ninfer::DType::I32, {T});

        ninfer::WorkspaceArena ws_prefill(flash_next_qsa_attention_workspace_capacity_bytes(T));
        flash_next_qsa_attention_prefill_chunk(chunk_in, weights, chunk_idx, chunk_pos, 0,
                                               chunk_sel, chunk_cnt, cache_b, ws_prefill,
                                               chunk_out_view, device.stream);
        device.synchronize();

        std::vector<std::uint16_t> chunk_output(static_cast<std::size_t>(input_dim) * T);
        chunk_out.copy_to_host(chunk_output.data(), chunk_output.size() * 2);

        // Compare KV cache entries
        std::vector<std::uint16_t> kp_a(256ULL * 64 * 2 * physical_pages);
        std::vector<std::uint16_t> kp_b(256ULL * 64 * 2 * physical_pages);
        key_pages_a.copy_to_host(kp_a.data(), kp_a.size() * 2);
        key_pages_b.copy_to_host(kp_b.data(), kp_b.size() * 2);
        for (std::size_t i = 0; i < kp_a.size(); ++i) {
            if (kp_a[i] != kp_b[i]) {
                std::cerr << "Key cache mismatch at T=" << T << ", idx=" << i << ": seq=0x"
                          << std::hex << kp_a[i] << ", chunk=0x" << kp_b[i] << std::dec << "\n";
                return false;
            }
        }

        // Compare output
        for (std::size_t i = 0; i < seq_output.size(); ++i) {
            float f_seq =
                __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&seq_output[i]));
            float f_chk =
                __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&chunk_output[i]));
            if (std::abs(f_seq - f_chk) > 1e-2f) {
                std::cerr << "Attended output mismatch at T=" << T << ", idx=" << i
                          << ": seq=" << f_seq << ", chunk=" << f_chk << "\n";
                return false;
            }
        }
    }
    return true;
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

    if (!test_prefill_vs_decode_equivalence(device)) {
        std::cerr << "FAIL: test_prefill_vs_decode_equivalence failed\n";
        return 1;
    }

    std::cout << "PASS: test_qsa_attention\n";
    return 0;
}
