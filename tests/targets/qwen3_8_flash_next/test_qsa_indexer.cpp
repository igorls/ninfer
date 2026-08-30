#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "targets/qwen3_8_flash_next/impl/qsa_indexer.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
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

bool test_block_publish_equivalence(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t maximum_blocks = 256;
    constexpr std::int32_t logical_pages  = 4;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::uint16_t> host_proj(640ULL * 2560);
    for (auto& v : host_proj) {
        float f = dist(rng);
        v       = __float2bfloat16_rn(f);
    }
    std::vector<std::uint16_t> host_qnorm(128), host_knorm(128);
    for (auto& v : host_qnorm) v = __float2bfloat16_rn(dist(rng) * 0.1f);
    for (auto& v : host_knorm) v = __float2bfloat16_rn(dist(rng) * 0.1f);

    ninfer::DeviceBuffer proj_buf(640ULL * 2560 * 2);
    ninfer::DeviceBuffer qnorm_buf(128 * 2);
    ninfer::DeviceBuffer knorm_buf(128 * 2);
    proj_buf.copy_from_host(host_proj.data(), host_proj.size() * 2);
    qnorm_buf.copy_from_host(host_qnorm.data(), host_qnorm.size() * 2);
    knorm_buf.copy_from_host(host_knorm.data(), host_knorm.size() * 2);

    AttentionWeights weights{};
    weights.indexer_query_key  = bf16_weight(proj_buf.p, 640, 2560);
    weights.indexer_query_norm = ninfer::Tensor(qnorm_buf.p, ninfer::DType::BF16, {128});
    weights.indexer_key_norm   = ninfer::Tensor(knorm_buf.p, ninfer::DType::BF16, {128});

    const std::vector<std::int32_t> test_t = {1, 2, 3, 4, 5, 7, 8, 16, 32, 64};
    for (std::int32_t T : test_t) {
        for (std::int32_t leftover_in = 0; leftover_in < 4; ++leftover_in) {
            const std::int32_t first_token_index = 100 + leftover_in;

            std::vector<std::uint16_t> host_input(static_cast<std::size_t>(2560) * T);
            for (auto& v : host_input) v = __float2bfloat16_rn(dist(rng));

            std::vector<std::int32_t> host_pos(3 * T);
            for (std::int32_t t = 0; t < T; ++t) {
                host_pos[0 * T + t] = first_token_index + t;
                host_pos[1 * T + t] = first_token_index + t;
                host_pos[2 * T + t] = first_token_index + t;
            }

            std::vector<std::uint16_t> init_raw_keys(128ULL * 4 * 2, 0);
            std::vector<std::int32_t> init_raw_pos(3ULL * 4 * 2, 0);
            for (std::int32_t s = 0; s < leftover_in; ++s) {
                for (int d = 0; d < 128; ++d) {
                    init_raw_keys[s * 128 + d] = __float2bfloat16_rn(dist(rng));
                }
                for (int d = 0; d < 3; ++d) {
                    init_raw_pos[s * 3 + d] = first_token_index - leftover_in + s;
                }
            }

            ninfer::DeviceBuffer block_keys_a(128ULL * 64 * logical_pages * 2);
            ninfer::DeviceBuffer block_tables_a(logical_pages * sizeof(std::int32_t));
            ninfer::DeviceBuffer raw_keys_a(128ULL * 4 * 2 * 2);
            ninfer::DeviceBuffer raw_positions_a(3ULL * 4 * 2 * sizeof(std::int32_t));
            block_keys_a.fill(0);
            raw_keys_a.copy_from_host(init_raw_keys.data(), init_raw_keys.size() * 2);
            raw_positions_a.copy_from_host(init_raw_pos.data(), init_raw_pos.size() * sizeof(std::int32_t));
            std::array<std::int32_t, logical_pages> page_ids{};
            for (std::int32_t p = 0; p < logical_pages; ++p) page_ids[p] = p;
            block_tables_a.copy_from_host(page_ids.data(), sizeof(page_ids));

            QsaIndexerCacheView cache_a{
                .block_keys   = ninfer::Tensor(block_keys_a.p, ninfer::DType::BF16, {128, 64, logical_pages}),
                .block_tables = ninfer::Tensor(block_tables_a.p, ninfer::DType::I32, {logical_pages, 1}),
                .raw_keys     = ninfer::Tensor(raw_keys_a.p, ninfer::DType::BF16, {128, 4, 2}),
                .raw_positions = ninfer::Tensor(raw_positions_a.p, ninfer::DType::I32, {3, 4, 2}),
            };

            ninfer::DeviceBuffer input_dev(static_cast<std::size_t>(2560) * T * 2);
            input_dev.copy_from_host(host_input.data(), host_input.size() * 2);

            ninfer::DeviceBuffer token_buf(sizeof(std::int32_t));
            ninfer::DeviceBuffer pos_buf(3 * sizeof(std::int32_t));
            ninfer::DeviceBuffer row_buf(sizeof(std::int32_t));
            ninfer::DeviceBuffer src_buf(sizeof(std::int32_t));
            ninfer::DeviceBuffer dst_buf(sizeof(std::int32_t));
            ninfer::DeviceBuffer sel_buf(512 * sizeof(std::int32_t));
            ninfer::DeviceBuffer cnt_buf(sizeof(std::int32_t));
            std::int32_t zero = 0;
            row_buf.copy_from_host(&zero, sizeof(zero));

            ninfer::WorkspaceArena ws_decode(flash_next_qsa_indexer_workspace_capacity_bytes(maximum_blocks, 1));

            for (std::int32_t t = 0; t < T; ++t) {
                // DeviceBuffer::copy_from_host is a legacy-stream cudaMemcpy: it does not order against
                // the non-blocking device.stream, so drain the previous iteration first.
                device.synchronize();
                std::int32_t tok_idx = first_token_index + t;
                std::array<std::int32_t, 3> pos_t = {host_pos[0 * T + t], host_pos[1 * T + t], host_pos[2 * T + t]};
                device.synchronize(); // legacy-stream copies do not order against device.stream
            token_buf.copy_from_host(&tok_idx, sizeof(tok_idx));
                pos_buf.copy_from_host(pos_t.data(), sizeof(pos_t));
                std::int32_t src = t % 2;
                std::int32_t dst = 1 - src;
                src_buf.copy_from_host(&src, sizeof(src));
                dst_buf.copy_from_host(&dst, sizeof(dst));

                ninfer::Tensor in_slice(static_cast<std::uint16_t*>(input_dev.p) + static_cast<std::size_t>(t) * 2560,
                                       ninfer::DType::BF16, {2560, 1});
                ninfer::Tensor tok_t(token_buf.p, ninfer::DType::I32, {1});
                ninfer::Tensor pos_t_view(pos_buf.p, ninfer::DType::I32, {1, 3});
                ninfer::Tensor row_view(row_buf.p, ninfer::DType::I32, {1});
                ninfer::Tensor src_view(src_buf.p, ninfer::DType::I32, {1});
                ninfer::Tensor dst_view(dst_buf.p, ninfer::DType::I32, {1});
                ninfer::Tensor sel_view(sel_buf.p, ninfer::DType::I32, {512, 1});
                ninfer::Tensor cnt_view(cnt_buf.p, ninfer::DType::I32, {1});

                flash_next_qsa_indexer_decode(in_slice, weights, tok_t, pos_t_view, row_view,
                                              src_view, dst_view, cache_a, maximum_blocks, maximum_blocks,
                                              ws_decode, sel_view, cnt_view, device.stream);
            }

            ninfer::DeviceBuffer block_keys_b(128ULL * 64 * logical_pages * 2);
            ninfer::DeviceBuffer block_tables_b(logical_pages * sizeof(std::int32_t));
            ninfer::DeviceBuffer raw_keys_b(128ULL * 4 * 2 * 2);
            ninfer::DeviceBuffer raw_positions_b(3ULL * 4 * 2 * sizeof(std::int32_t));
            block_keys_b.fill(0);
            raw_keys_b.copy_from_host(init_raw_keys.data(), init_raw_keys.size() * 2);
            raw_positions_b.copy_from_host(init_raw_pos.data(), init_raw_pos.size() * sizeof(std::int32_t));
            block_tables_b.copy_from_host(page_ids.data(), sizeof(page_ids));

            QsaIndexerCacheView cache_b{
                .block_keys   = ninfer::Tensor(block_keys_b.p, ninfer::DType::BF16, {128, 64, logical_pages}),
                .block_tables = ninfer::Tensor(block_tables_b.p, ninfer::DType::I32, {logical_pages, 1}),
                .raw_keys     = ninfer::Tensor(raw_keys_b.p, ninfer::DType::BF16, {128, 4, 2}),
                .raw_positions = ninfer::Tensor(raw_positions_b.p, ninfer::DType::I32, {3, 4, 2}),
            };

            ninfer::DeviceBuffer dev_indices(T * sizeof(std::int32_t));
            std::vector<std::int32_t> host_indices(T);
            for (std::int32_t t = 0; t < T; ++t) host_indices[t] = first_token_index + t;
            dev_indices.copy_from_host(host_indices.data(), T * sizeof(std::int32_t));

            ninfer::DeviceBuffer dev_mrope_pos(3 * T * sizeof(std::int32_t));
            dev_mrope_pos.copy_from_host(host_pos.data(), host_pos.size() * sizeof(std::int32_t));

            ninfer::DeviceBuffer chunk_sel_buf(512 * T * sizeof(std::int32_t));
            ninfer::DeviceBuffer chunk_cnt_buf(T * sizeof(std::int32_t));

            ninfer::Tensor chunk_in(input_dev.p, ninfer::DType::BF16, {2560, T});
            ninfer::Tensor chunk_idx(dev_indices.p, ninfer::DType::I32, {T});
            ninfer::Tensor chunk_pos(dev_mrope_pos.p, ninfer::DType::I32, {T, 3});
            ninfer::Tensor chunk_sel(chunk_sel_buf.p, ninfer::DType::I32, {512, T});
            ninfer::Tensor chunk_cnt(chunk_cnt_buf.p, ninfer::DType::I32, {T});

            ninfer::WorkspaceArena ws_prefill(100ULL * 1024 * 1024);

            flash_next_qsa_indexer_prefill_chunk(
                chunk_in, weights, chunk_idx, chunk_pos, 0, 0, 1, cache_b, maximum_blocks,
                ws_prefill, chunk_sel, chunk_cnt, device.stream);
            device.synchronize();

            std::vector<std::uint16_t> keys_a(128ULL * 64 * logical_pages);
            std::vector<std::uint16_t> keys_b(128ULL * 64 * logical_pages);
            block_keys_a.copy_to_host(keys_a.data(), keys_a.size() * 2);
            block_keys_b.copy_to_host(keys_b.data(), keys_b.size() * 2);

            for (std::size_t i = 0; i < keys_a.size(); ++i) {
                float fa = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&keys_a[i]));
                float fb = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&keys_b[i]));
                if (std::abs(fa - fb) > 1e-4f) {
                    std::cerr << "Block keys mismatch at T=" << T << ", leftover_in=" << leftover_in
                              << ", idx " << i << ": seq=" << fa << " chunk=" << fb << "\n";
                    return false;
                }
            }

            const std::int32_t final_slot_a = (T % 2 == 1) ? 1 : 0;
            const std::int32_t leftover_out = (leftover_in + T) & 3;
            std::vector<std::uint16_t> rk_a(128ULL * 4 * 2);
            std::vector<std::uint16_t> rk_b(128ULL * 4 * 2);
            raw_keys_a.copy_to_host(rk_a.data(), rk_a.size() * 2);
            raw_keys_b.copy_to_host(rk_b.data(), rk_b.size() * 2);
            for (std::int32_t s = 0; s < leftover_out; ++s) {
                for (int d = 0; d < 128; ++d) {
                    const std::size_t idx_a = final_slot_a * 128 * 4 + s * 128 + d;
                    const std::size_t idx_b = 1 * 128 * 4 + s * 128 + d;
                    if (rk_a[idx_a] != rk_b[idx_b]) {
                        std::cerr << "Raw keys mismatch at T=" << T
                                  << ", leftover_in=" << leftover_in << ", slot=" << s
                                  << ", d=" << d << ": seq=0x" << std::hex << rk_a[idx_a]
                                  << " chunk=0x" << rk_b[idx_b] << std::dec << "\n";
                        return false;
                    }
                }
            }

            std::vector<std::int32_t> rp_a(3ULL * 4 * 2);
            std::vector<std::int32_t> rp_b(3ULL * 4 * 2);
            raw_positions_a.copy_to_host(rp_a.data(), rp_a.size() * sizeof(std::int32_t));
            raw_positions_b.copy_to_host(rp_b.data(), rp_b.size() * sizeof(std::int32_t));
            for (std::int32_t s = 0; s < leftover_out; ++s) {
                for (int d = 0; d < 3; ++d) {
                    const std::size_t idx_a = final_slot_a * 3 * 4 + s * 3 + d;
                    const std::size_t idx_b = 1 * 3 * 4 + s * 3 + d;
                    if (rp_a[idx_a] != rp_b[idx_b]) {
                        std::cerr << "Raw positions mismatch at T=" << T
                                  << ", leftover_in=" << leftover_in << ", slot=" << s
                                  << ", d=" << d << ": seq=" << rp_a[idx_a] << " chunk=" << rp_b[idx_b]
                                  << "\n";
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

bool test_selection_equivalence(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t maximum_blocks = 256;
    constexpr std::int32_t logical_pages  = 4;

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::uint16_t> host_proj(640ULL * 2560);
    for (auto& v : host_proj) v = __float2bfloat16_rn(dist(rng));
    std::vector<std::uint16_t> host_qnorm(128), host_knorm(128);
    for (auto& v : host_qnorm) v = __float2bfloat16_rn(dist(rng) * 0.1f);
    for (auto& v : host_knorm) v = __float2bfloat16_rn(dist(rng) * 0.1f);

    ninfer::DeviceBuffer proj_buf(640ULL * 2560 * 2);
    ninfer::DeviceBuffer qnorm_buf(128 * 2);
    ninfer::DeviceBuffer knorm_buf(128 * 2);
    proj_buf.copy_from_host(host_proj.data(), host_proj.size() * 2);
    qnorm_buf.copy_from_host(host_qnorm.data(), host_qnorm.size() * 2);
    knorm_buf.copy_from_host(host_knorm.data(), host_knorm.size() * 2);

    AttentionWeights weights{};
    weights.indexer_query_key  = bf16_weight(proj_buf.p, 640, 2560);
    weights.indexer_query_norm = ninfer::Tensor(qnorm_buf.p, ninfer::DType::BF16, {128});
    weights.indexer_key_norm   = ninfer::Tensor(knorm_buf.p, ninfer::DType::BF16, {128});

    const std::vector<std::int32_t> test_t = {1, 4, 8, 32, 128, 256};
    for (std::int32_t T : test_t) {
        const std::int32_t first_token_index = 40;

        std::vector<std::uint16_t> host_input(static_cast<std::size_t>(2560) * T);
        for (auto& v : host_input) v = __float2bfloat16_rn(dist(rng));

        std::vector<std::int32_t> host_pos(3 * T);
        for (std::int32_t t = 0; t < T; ++t) {
            host_pos[0 * T + t] = first_token_index + t;
            host_pos[1 * T + t] = first_token_index + t;
            host_pos[2 * T + t] = first_token_index + t;
        }

        std::vector<std::uint16_t> host_bkeys(128ULL * 64 * logical_pages);
        for (auto& v : host_bkeys) v = __float2bfloat16_rn(dist(rng));

        ninfer::DeviceBuffer block_keys_a(128ULL * 64 * logical_pages * 2);
        ninfer::DeviceBuffer block_keys_b(128ULL * 64 * logical_pages * 2);
        block_keys_a.copy_from_host(host_bkeys.data(), host_bkeys.size() * 2);
        block_keys_b.copy_from_host(host_bkeys.data(), host_bkeys.size() * 2);

        ninfer::DeviceBuffer block_tables(logical_pages * sizeof(std::int32_t));
        std::array<std::int32_t, logical_pages> page_ids{};
        for (std::int32_t p = 0; p < logical_pages; ++p) page_ids[p] = p;
        block_tables.copy_from_host(page_ids.data(), sizeof(page_ids));

        ninfer::DeviceBuffer raw_keys_a(128ULL * 4 * 2 * 2), raw_keys_b(128ULL * 4 * 2 * 2);
        ninfer::DeviceBuffer raw_pos_a(3ULL * 4 * 2 * sizeof(std::int32_t)), raw_pos_b(3ULL * 4 * 2 * sizeof(std::int32_t));
        raw_keys_a.fill(0); raw_keys_b.fill(0);
        raw_pos_a.fill(0); raw_pos_b.fill(0);

        QsaIndexerCacheView cache_a{
            .block_keys   = ninfer::Tensor(block_keys_a.p, ninfer::DType::BF16, {128, 64, logical_pages}),
            .block_tables = ninfer::Tensor(block_tables.p, ninfer::DType::I32, {logical_pages, 1}),
            .raw_keys     = ninfer::Tensor(raw_keys_a.p, ninfer::DType::BF16, {128, 4, 2}),
            .raw_positions = ninfer::Tensor(raw_pos_a.p, ninfer::DType::I32, {3, 4, 2}),
        };
        QsaIndexerCacheView cache_b{
            .block_keys   = ninfer::Tensor(block_keys_b.p, ninfer::DType::BF16, {128, 64, logical_pages}),
            .block_tables = ninfer::Tensor(block_tables.p, ninfer::DType::I32, {logical_pages, 1}),
            .raw_keys     = ninfer::Tensor(raw_keys_b.p, ninfer::DType::BF16, {128, 4, 2}),
            .raw_positions = ninfer::Tensor(raw_pos_b.p, ninfer::DType::I32, {3, 4, 2}),
        };

        ninfer::DeviceBuffer input_dev(static_cast<std::size_t>(2560) * T * 2);
        input_dev.copy_from_host(host_input.data(), host_input.size() * 2);

        std::vector<std::int32_t> seq_counts(T);
        std::vector<std::int32_t> seq_blocks(512 * T);

        ninfer::DeviceBuffer token_buf(sizeof(std::int32_t));
        ninfer::DeviceBuffer pos_buf(3 * sizeof(std::int32_t));
        ninfer::DeviceBuffer row_buf(sizeof(std::int32_t));
        ninfer::DeviceBuffer src_buf(sizeof(std::int32_t));
        ninfer::DeviceBuffer dst_buf(sizeof(std::int32_t));
        ninfer::DeviceBuffer sel_buf(512 * sizeof(std::int32_t));
        ninfer::DeviceBuffer cnt_buf(sizeof(std::int32_t));
        std::int32_t zero = 0;
        row_buf.copy_from_host(&zero, sizeof(zero));

        ninfer::WorkspaceArena ws_decode(flash_next_qsa_indexer_workspace_capacity_bytes(maximum_blocks, 1));

        for (std::int32_t t = 0; t < T; ++t) {
            std::int32_t tok_idx = first_token_index + t;
            std::array<std::int32_t, 3> pos_t = {host_pos[0 * T + t], host_pos[1 * T + t], host_pos[2 * T + t]};
            device.synchronize(); // legacy-stream copies do not order against device.stream
            token_buf.copy_from_host(&tok_idx, sizeof(tok_idx));
            pos_buf.copy_from_host(pos_t.data(), sizeof(pos_t));
            std::int32_t src = t % 2;
            std::int32_t dst = 1 - src;
            src_buf.copy_from_host(&src, sizeof(src));
            dst_buf.copy_from_host(&dst, sizeof(dst));

            ninfer::Tensor in_slice(static_cast<std::uint16_t*>(input_dev.p) + static_cast<std::size_t>(t) * 2560,
                                   ninfer::DType::BF16, {2560, 1});
            ninfer::Tensor tok_t(token_buf.p, ninfer::DType::I32, {1});
            ninfer::Tensor pos_t_view(pos_buf.p, ninfer::DType::I32, {1, 3});
            ninfer::Tensor row_view(row_buf.p, ninfer::DType::I32, {1});
            ninfer::Tensor src_view(src_buf.p, ninfer::DType::I32, {1});
            ninfer::Tensor dst_view(dst_buf.p, ninfer::DType::I32, {1});
            ninfer::Tensor sel_view(sel_buf.p, ninfer::DType::I32, {512, 1});
            ninfer::Tensor cnt_view(cnt_buf.p, ninfer::DType::I32, {1});

            flash_next_qsa_indexer_decode(in_slice, weights, tok_t, pos_t_view, row_view,
                                          src_view, dst_view, cache_a, maximum_blocks, maximum_blocks,
                                          ws_decode, sel_view, cnt_view, device.stream);
            device.synchronize();
            cnt_buf.copy_to_host(&seq_counts[t], sizeof(std::int32_t));
            sel_buf.copy_to_host(&seq_blocks[t * 512], 512 * sizeof(std::int32_t));
        }

        ninfer::DeviceBuffer dev_indices(T * sizeof(std::int32_t));
        std::vector<std::int32_t> host_indices(T);
        for (std::int32_t t = 0; t < T; ++t) host_indices[t] = first_token_index + t;
        dev_indices.copy_from_host(host_indices.data(), T * sizeof(std::int32_t));

        ninfer::DeviceBuffer dev_mrope_pos(3 * T * sizeof(std::int32_t));
        dev_mrope_pos.copy_from_host(host_pos.data(), host_pos.size() * sizeof(std::int32_t));

        ninfer::DeviceBuffer chunk_sel_buf(512 * T * sizeof(std::int32_t));
        ninfer::DeviceBuffer chunk_cnt_buf(T * sizeof(std::int32_t));

        ninfer::Tensor chunk_in(input_dev.p, ninfer::DType::BF16, {2560, T});
        ninfer::Tensor chunk_idx(dev_indices.p, ninfer::DType::I32, {T});
        ninfer::Tensor chunk_pos(dev_mrope_pos.p, ninfer::DType::I32, {T, 3});
        ninfer::Tensor chunk_sel(chunk_sel_buf.p, ninfer::DType::I32, {512, T});
        ninfer::Tensor chunk_cnt(chunk_cnt_buf.p, ninfer::DType::I32, {T});

        ninfer::WorkspaceArena ws_prefill(100ULL * 1024 * 1024);
        flash_next_qsa_indexer_prefill_chunk(
            chunk_in, weights, chunk_idx, chunk_pos, 0, 0, 1, cache_b, maximum_blocks,
            ws_prefill, chunk_sel, chunk_cnt, device.stream);
        device.synchronize();

        std::vector<std::int32_t> chunk_counts(T);
        std::vector<std::int32_t> chunk_blocks(512 * T);
        chunk_cnt_buf.copy_to_host(chunk_counts.data(), T * sizeof(std::int32_t));
        chunk_sel_buf.copy_to_host(chunk_blocks.data(), 512 * T * sizeof(std::int32_t));

        for (std::int32_t t = 0; t < T; ++t) {
            if (seq_counts[t] != chunk_counts[t]) {
                std::cerr << "Selection count mismatch at T=" << T << ", t=" << t
                          << ": seq=" << seq_counts[t] << ", chunk=" << chunk_counts[t] << "\n";
                return false;
            }
            const std::int32_t count = seq_counts[t];
            for (std::int32_t k = 0; k < count; ++k) {
                if (seq_blocks[t * 512 + k] != chunk_blocks[t * 512 + k]) {
                    std::cerr << "Selected block mismatch at T=" << T << ", t=" << t << ", rank=" << k
                              << ": seq=" << seq_blocks[t * 512 + k] << ", chunk=" << chunk_blocks[t * 512 + k] << "\n";
                    return false;
                }
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

    if (!test_block_publish_equivalence(device)) {
        std::cerr << "FAIL: test_block_publish_equivalence failed\n";
        return 1;
    }
    if (!test_selection_equivalence(device)) {
        std::cerr << "FAIL: test_selection_equivalence failed\n";
        return 1;
    }

    std::cout << "PASS: test_qsa_indexer\n";
    return 0;
}
