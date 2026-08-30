#include "core/arena.h"
#include "core/device.h"
#include "runtime/engine/context_cost.h"
#include "targets/qwen3_8_flash_next/impl/vision_adapter.h"
#include "targets/qwen3_8_flash_next/impl/vision_execute.h"
#include "targets/qwen3_8_flash_next/impl/text_decode.h"
#include "targets/qwen3_8_flash_next/impl/text_executor.h"
#include "targets/qwen3_8_flash_next/impl/program_impl.h"
#include "targets/qwen3_8_flash_next/impl/ple_table.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6/vision_control.h>
#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include <ninfer/targets/qwen3_vision/vision.h>

#include <cuda_runtime.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::targets::qwen3_8_flash_next;
using namespace ninfer::targets::qwen3_8_flash_next::detail;
using namespace ninfer::targets::qwen3_vision;

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

std::uint16_t float_to_bf16(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t lsb  = (bits >> 16U) & 1U;
    const std::uint32_t bias = 0x7FFFU + lsb;
    return static_cast<std::uint16_t>((bits + bias) >> 16U);
}

Weight make_bf16_weight(const void* data, std::int32_t rows, std::int32_t cols) {
    Weight w{};
    w.payload       = data;
    w.payload_bytes = static_cast<std::size_t>(rows) * cols * sizeof(std::uint16_t);
    w.qdata         = data;
    w.qtype         = QType::BF16_CTRL;
    w.layout        = QuantLayout::Contiguous;
    w.shape[0]      = rows;
    w.shape[1]      = cols;
    w.shape[2]      = 1;
    w.shape[3]      = 1;
    w.padded_shape[0] = rows;
    w.padded_shape[1] = cols;
    w.n             = rows;
    w.k             = cols;
    w.ndim          = 2;
    return w;
}

struct SyntheticFlashNextModel {
    ninfer::DeviceBuffer big_bf16_buf;
    ninfer::DeviceBuffer norm_bf16_buf;
    ninfer::DeviceBuffer gdn_a_log_buf;
    ninfer::DeviceBuffer gdn_dt_bias_buf;
    ninfer::DeviceBuffer gdn_conv_buf;
    ninfer::DeviceBuffer ple_conv_buf;
    ninfer::DeviceBuffer shared_gate_weight_buf;
    ninfer::DeviceBuffer inject_buf;

    ninfer::DeviceBuffer fp8_qkvz_buf;
    ninfer::DeviceBuffer fp8_qgkv_buf;
    ninfer::DeviceBuffer fp8_out_buf;

    ninfer::DeviceBuffer big_nvfp4_gate_codes_buf;
    ninfer::DeviceBuffer big_nvfp4_gate_scales_buf;
    ninfer::DeviceBuffer big_nvfp4_down_codes_buf;
    ninfer::DeviceBuffer big_nvfp4_down_scales_buf;
    ninfer::DeviceBuffer big_divisors_buf;

    std::vector<std::byte> ple_table_data;
    TextModelView view;
};

SyntheticFlashNextModel make_synthetic_model(ninfer::DeviceContext& device) {
    SyntheticFlashNextModel model;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_bf16(-0.02f, 0.02f);
    std::uniform_real_distribution<float> dist_norm(0.98f, 1.02f);

    constexpr std::uint64_t kOutputHeadBytes = 248'320ULL * 2'560 * 2;
    model.big_bf16_buf = ninfer::DeviceBuffer(kOutputHeadBytes);
    constexpr std::size_t kChunkFloats = 2'560 * 1024;
    std::vector<std::uint16_t> h_bf16(kChunkFloats);
    for (auto& v : h_bf16) { v = float_to_bf16(dist_bf16(rng)); }
    for (std::size_t off = 0; off < kOutputHeadBytes; off += h_bf16.size() * sizeof(std::uint16_t)) {
        std::size_t chunk = std::min<std::size_t>(h_bf16.size() * sizeof(std::uint16_t), kOutputHeadBytes - off);
        model.big_bf16_buf.copy_from_host(h_bf16.data(), chunk, off);
    }

    std::vector<std::uint16_t> h_norm(10'240);
    for (auto& v : h_norm) { v = float_to_bf16(dist_norm(rng)); }
    model.norm_bf16_buf = ninfer::DeviceBuffer(10'240 * sizeof(std::uint16_t));
    model.norm_bf16_buf.copy_from_host(h_norm.data(), h_norm.size() * sizeof(std::uint16_t));

    std::vector<std::uint16_t> h_a_log(48);
    for (auto& v : h_a_log) { v = float_to_bf16(-1.0f); }
    model.gdn_a_log_buf = ninfer::DeviceBuffer(48 * sizeof(std::uint16_t));
    model.gdn_a_log_buf.copy_from_host(h_a_log.data(), h_a_log.size() * sizeof(std::uint16_t));

    std::vector<std::uint16_t> h_dt_bias(48);
    for (auto& v : h_dt_bias) { v = float_to_bf16(0.05f); }
    model.gdn_dt_bias_buf = ninfer::DeviceBuffer(48 * sizeof(std::uint16_t));
    model.gdn_dt_bias_buf.copy_from_host(h_dt_bias.data(), h_dt_bias.size() * sizeof(std::uint16_t));

    std::vector<std::uint16_t> h_gdn_conv(10'240 * 4);
    for (auto& v : h_gdn_conv) { v = float_to_bf16(0.25f); }
    model.gdn_conv_buf = ninfer::DeviceBuffer(10'240 * 4 * sizeof(std::uint16_t));
    model.gdn_conv_buf.copy_from_host(h_gdn_conv.data(), h_gdn_conv.size() * sizeof(std::uint16_t));

    std::vector<std::uint16_t> h_ple_conv(10'240 * 4);
    for (int c = 0; c < 10'240; ++c) {
        h_ple_conv[0 * 10'240 + c] = float_to_bf16(0.25f);
        h_ple_conv[1 * 10'240 + c] = float_to_bf16(0.50f);
        h_ple_conv[2 * 10'240 + c] = float_to_bf16(0.75f);
        h_ple_conv[3 * 10'240 + c] = float_to_bf16(1.00f);
    }
    model.ple_conv_buf = ninfer::DeviceBuffer(10'240 * 4 * sizeof(std::uint16_t));
    model.ple_conv_buf.copy_from_host(h_ple_conv.data(), h_ple_conv.size() * sizeof(std::uint16_t));

    std::vector<std::uint16_t> h_sgw(2'560);
    for (auto& v : h_sgw) { v = float_to_bf16(0.1f); }
    model.shared_gate_weight_buf = ninfer::DeviceBuffer(2'560 * sizeof(std::uint16_t));
    model.shared_gate_weight_buf.copy_from_host(h_sgw.data(), h_sgw.size() * sizeof(std::uint16_t));

    std::vector<std::uint16_t> h_inject(4 * 10'240);
    for (auto& v : h_inject) { v = float_to_bf16(0.25f); }
    model.inject_buf = ninfer::DeviceBuffer(4 * 10'240 * sizeof(std::uint16_t));
    model.inject_buf.copy_from_host(h_inject.data(), h_inject.size() * sizeof(std::uint16_t));

    auto init_fp8_buf = [&](ninfer::DeviceBuffer& buf, std::int32_t rows, std::int32_t cols, float scale_val) {
        const std::uint64_t codes_bytes = static_cast<std::uint64_t>(rows) * cols;
        const std::uint64_t scale_off   = (codes_bytes + 255U) & ~255ULL;
        const std::uint64_t total_bytes = scale_off + static_cast<std::uint64_t>(rows) * sizeof(float);
        buf = ninfer::DeviceBuffer(total_bytes);

        std::vector<std::uint8_t> h_codes(codes_bytes);
        for (std::size_t i = 0; i < codes_bytes; ++i) {
            h_codes[i] = static_cast<std::uint8_t>(0x18 + (rng() % 32));
        }
        buf.copy_from_host(h_codes.data(), codes_bytes, 0);

        std::vector<float> h_scales(rows, scale_val);
        buf.copy_from_host(h_scales.data(), rows * sizeof(float), scale_off);
    };

    init_fp8_buf(model.fp8_qkvz_buf, 16'384, 2'560, 1.0f / std::sqrt(2'560.0f));
    init_fp8_buf(model.fp8_qgkv_buf, 13'312, 2'560, 1.0f / std::sqrt(2'560.0f));
    init_fp8_buf(model.fp8_out_buf, 2'560, 6'144, 1.0f / std::sqrt(6'144.0f));

    constexpr std::uint64_t gate_code_bytes_per_expert  = 1'280ULL * 2'560 / 2;
    constexpr std::uint64_t gate_scale_bytes_per_expert = 1'280ULL * 2'560 / 16;
    constexpr std::uint64_t down_code_bytes_per_expert  = 2'560ULL * 640 / 2;
    constexpr std::uint64_t down_scale_bytes_per_expert = 2'560ULL * 640 / 16;

    model.big_nvfp4_gate_codes_buf  = ninfer::DeviceBuffer(512 * gate_code_bytes_per_expert);
    model.big_nvfp4_gate_scales_buf = ninfer::DeviceBuffer(512 * gate_scale_bytes_per_expert);
    model.big_nvfp4_down_codes_buf  = ninfer::DeviceBuffer(512 * down_code_bytes_per_expert);
    model.big_nvfp4_down_scales_buf = ninfer::DeviceBuffer(512 * down_scale_bytes_per_expert);
    model.big_divisors_buf          = ninfer::DeviceBuffer(512 * sizeof(float));

    std::vector<std::uint8_t> h_fp4(1024 * 1024);
    for (auto& b : h_fp4) {
        const auto low  = static_cast<std::uint8_t>(1 + (rng() % 3));
        const auto high = static_cast<std::uint8_t>(1 + (rng() % 3));
        b = static_cast<std::uint8_t>((high << 4) | low);
    }
    for (std::size_t off = 0; off < model.big_nvfp4_gate_codes_buf.bytes; off += h_fp4.size()) {
        std::size_t chunk = std::min<std::size_t>(h_fp4.size(), model.big_nvfp4_gate_codes_buf.bytes - off);
        model.big_nvfp4_gate_codes_buf.copy_from_host(h_fp4.data(), chunk, off);
    }
    model.big_nvfp4_gate_scales_buf.fill(0x38);
    for (std::size_t off = 0; off < model.big_nvfp4_down_codes_buf.bytes; off += h_fp4.size()) {
        std::size_t chunk = std::min<std::size_t>(h_fp4.size(), model.big_nvfp4_down_codes_buf.bytes - off);
        model.big_nvfp4_down_codes_buf.copy_from_host(h_fp4.data(), chunk, off);
    }
    model.big_nvfp4_down_scales_buf.fill(0x38);

    std::vector<float> divisors(512, 1.0f);
    model.big_divisors_buf.copy_from_host(divisors.data(), divisors.size() * sizeof(float));

    constexpr std::uint64_t rows         = 1;
    constexpr std::uint64_t width        = 160;
    constexpr std::uint64_t scale_offset = 256;
    model.ple_table_data = std::vector<std::byte>(scale_offset + (width / 16) * 2, std::byte{0});
    for (std::size_t i = 0; i < width / 2; ++i) {
        model.ple_table_data[i] = static_cast<std::byte>(0x22 + (rng() % 16));
    }
    for (std::uint8_t index = 0; index < 8; ++index) {
        model.ple_table_data[index] = static_cast<std::byte>(index * 2 | ((index * 2 + 1) << 4));
    }
    constexpr std::uint16_t half_point_five = 0x3800;
    for (std::size_t offset = scale_offset; offset < model.ple_table_data.size(); offset += 2) {
        std::memcpy(model.ple_table_data.data() + offset, &half_point_five, sizeof(half_point_five));
    }
    for (PleShardView& shard : model.view.ple.table.shards) {
        shard = make_ple_shard_view(model.ple_table_data, rows, width);
    }

    auto make_bf16_weight_from = [](ninfer::DeviceBuffer& buf, std::int32_t rows, std::int32_t cols) {
        ninfer::Weight w{};
        w.payload         = buf.p;
        w.payload_bytes   = static_cast<std::uint64_t>(rows) * cols * 2;
        w.qdata           = buf.p;
        w.qtype           = ninfer::QType::BF16_CTRL;
        w.layout          = ninfer::QuantLayout::Contiguous;
        w.n               = rows;
        w.k               = cols;
        w.ndim            = 2;
        w.shape[0]        = rows;
        w.shape[1]        = cols;
        w.padded_shape[0] = rows;
        w.padded_shape[1] = cols;
        return w;
    };

    auto make_bf16_weight = [&](std::int32_t rows, std::int32_t cols) {
        return make_bf16_weight_from(model.big_bf16_buf, rows, cols);
    };

    auto make_fp8_weight = [](ninfer::DeviceBuffer& buf, std::int32_t rows, std::int32_t cols) {
        const std::uint64_t codes = static_cast<std::uint64_t>(rows) * cols;
        const std::uint64_t scale_off = (codes + 255U) & ~255ULL;
        ninfer::Weight w{};
        w.payload           = buf.p;
        w.payload_bytes     = buf.bytes;
        w.qdata             = buf.p;
        w.scales            = static_cast<const std::byte*>(buf.p) + scale_off;
        w.qtype             = ninfer::QType::FP8_E4M3FN_ROW_F32S;
        w.layout            = ninfer::QuantLayout::RowScale;
        w.scale_dtype       = ninfer::DType::FP32;
        w.group_size        = cols;
        w.group             = cols;
        w.n                 = rows;
        w.k                 = cols;
        w.ndim              = 2;
        w.shape[0]          = rows;
        w.shape[1]          = cols;
        w.shape[2]          = 1;
        w.shape[3]          = 1;
        w.padded_shape[0]   = rows;
        w.padded_shape[1]   = cols;
        w.padded_shape[2]   = 1;
        w.padded_shape[3]   = 1;
        w.scale_ne[0]       = rows;
        w.scale_ne[1]       = 1;
        w.scale_ne[2]       = 1;
        w.scale_ne[3]       = 1;
        w.scale_nb[0]       = 4;
        w.scale_nb[1]       = static_cast<std::int64_t>(rows) * 4;
        w.scale_nb[2]       = static_cast<std::int64_t>(rows) * 4;
        w.scale_nb[3]       = static_cast<std::int64_t>(rows) * 4;
        return w;
    };

    model.view.token_embedding = make_bf16_weight(248'320, 2'560);
    model.view.output_head     = make_bf16_weight(248'320, 2'560);

    model.view.ple.convolution      = ninfer::Tensor(model.ple_conv_buf.p, ninfer::DType::BF16, {10'240, 4});
    model.view.ple.key_projection   = make_bf16_weight(10'240, 2'560);
    model.view.ple.conv_norm        = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {10'240});
    model.view.ple.key_norm         = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {10'240});
    model.view.ple.query_norm       = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {10'240});
    model.view.ple.value_projection = make_bf16_weight(2'560, 2'560);

    model.view.final_mixer.norm           = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {10'240});
    model.view.final_mixer.input_mix_down = make_bf16_weight(320, 10'240);
    model.view.final_mixer.input_mix_up   = make_bf16_weight(10'240, 320);

    for (std::size_t l = 0; l < 48; ++l) {
        auto& layer = model.view.layers[l];
        layer.attention_hyper.block_inject   = make_bf16_weight_from(model.inject_buf, 4, 10'240);
        layer.attention_hyper.norm           = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {10'240});
        layer.attention_hyper.input_mix_down = make_bf16_weight(320, 10'240);
        layer.attention_hyper.input_mix_up   = make_bf16_weight(10'240, 320);

        layer.mlp_hyper.block_inject   = make_bf16_weight_from(model.inject_buf, 4, 10'240);
        layer.mlp_hyper.norm           = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {10'240});
        layer.mlp_hyper.input_mix_down = make_bf16_weight(320, 10'240);
        layer.mlp_hyper.input_mix_up   = make_bf16_weight(10'240, 320);

        layer.moe.router             = make_bf16_weight(512, 2'560);
        layer.moe.shared_down        = make_bf16_weight(2'560, 640);
        layer.moe.shared_gate        = make_bf16_weight(640, 2'560);
        layer.moe.shared_up          = make_bf16_weight(640, 2'560);
        layer.moe.shared_gate_weight = make_bf16_weight_from(model.shared_gate_weight_buf, 1, 2'560);
        layer.moe.expert_gate_up     = Nvfp4ExpertBankView{
            .codes                  = static_cast<const std::byte*>(model.big_nvfp4_gate_codes_buf.p),
            .scales                 = static_cast<const std::byte*>(model.big_nvfp4_gate_scales_buf.p),
            .weight_scale_divisors  = static_cast<const float*>(model.big_divisors_buf.p),
            .experts                = 512,
            .rows                   = 1'280,
            .columns                = 2'560,
            .code_bytes_per_expert  = gate_code_bytes_per_expert,
            .scale_bytes_per_expert = gate_scale_bytes_per_expert,
        };
        layer.moe.expert_down        = Nvfp4ExpertBankView{
            .codes                  = static_cast<const std::byte*>(model.big_nvfp4_down_codes_buf.p),
            .scales                 = static_cast<const std::byte*>(model.big_nvfp4_down_scales_buf.p),
            .weight_scale_divisors  = static_cast<const float*>(model.big_divisors_buf.p),
            .experts                = 512,
            .rows                   = 2'560,
            .columns                = 640,
            .code_bytes_per_expert  = down_code_bytes_per_expert,
            .scale_bytes_per_expert = down_scale_bytes_per_expert,
        };
    }

    for (std::size_t i = 0; i < kGdnLayers; ++i) {
        auto& gdn = model.view.gdn[i];
        gdn.a_log             = ninfer::Tensor(model.gdn_a_log_buf.p, ninfer::DType::BF16, {48});
        gdn.convolution       = ninfer::Tensor(model.gdn_conv_buf.p, ninfer::DType::BF16, {10'240, 4});
        gdn.dt_bias           = ninfer::Tensor(model.gdn_dt_bias_buf.p, ninfer::DType::BF16, {48});
        gdn.a_b_projection    = make_bf16_weight(96, 2'560);
        gdn.norm              = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {128});
        gdn.query_key_value_z = make_fp8_weight(model.fp8_qkvz_buf, 16'384, 2'560);
        gdn.output            = make_fp8_weight(model.fp8_out_buf, 2'560, 6'144);
    }

    for (std::size_t i = 0; i < kFullAttentionLayers; ++i) {
        auto& att = model.view.full_attention[i];
        att.indexer_query_key    = make_bf16_weight(640, 2'560);
        att.indexer_key_norm     = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {128});
        att.indexer_query_norm   = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {128});
        att.key_norm             = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {256});
        att.query_norm           = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {256});
        att.query_gate_key_value = make_fp8_weight(model.fp8_qgkv_buf, 13'312, 2'560);
        att.output               = make_fp8_weight(model.fp8_out_buf, 2'560, 6'144);
    }

    device.synchronize();
    return model;
}

struct SyntheticVisionModel {
    DeviceBuffer patch_w_buf{VisionTowerConfig::hidden * VisionTowerConfig::patch_dim * 2};
    DeviceBuffer patch_b_buf{VisionTowerConfig::hidden * 2};
    DeviceBuffer pos_embed_buf{VisionTowerConfig::position_embeddings * VisionTowerConfig::hidden * 2};

    struct LayerStorage {
        DeviceBuffer norm1_w{VisionTowerConfig::hidden * 2};
        DeviceBuffer norm1_b{VisionTowerConfig::hidden * 2};
        DeviceBuffer qkv_w{3 * VisionTowerConfig::hidden * VisionTowerConfig::hidden * 2};
        DeviceBuffer qkv_b{3 * VisionTowerConfig::hidden * 2};
        DeviceBuffer proj_w{VisionTowerConfig::hidden * VisionTowerConfig::hidden * 2};
        DeviceBuffer proj_b{VisionTowerConfig::hidden * 2};
        DeviceBuffer norm2_w{VisionTowerConfig::hidden * 2};
        DeviceBuffer norm2_b{VisionTowerConfig::hidden * 2};
        DeviceBuffer fc1_w{VisionTowerConfig::intermediate * VisionTowerConfig::hidden * 2};
        DeviceBuffer fc1_b{VisionTowerConfig::intermediate * 2};
        DeviceBuffer fc2_w{VisionTowerConfig::hidden * VisionTowerConfig::intermediate * 2};
        DeviceBuffer fc2_b{VisionTowerConfig::hidden * 2};
    };

    std::vector<LayerStorage> layers{VisionTowerConfig::layers};

    DeviceBuffer merger_norm_w{VisionTowerConfig::hidden * 2};
    DeviceBuffer merger_norm_b{VisionTowerConfig::hidden * 2};
    DeviceBuffer merger_fc1_w{VisionTowerConfig::merger_hidden * VisionTowerConfig::merger_hidden * 2};
    DeviceBuffer merger_fc1_b{VisionTowerConfig::merger_hidden * 2};
    DeviceBuffer merger_fc2_w{2560 * VisionTowerConfig::merger_hidden * 2};
    DeviceBuffer merger_fc2_b{2560 * 2};

    VisionModelView view{};

    explicit SyntheticVisionModel(DeviceContext& device) {
        CUDA_CHECK(cudaMemset(patch_w_buf.p, 0, patch_w_buf.bytes));
        CUDA_CHECK(cudaMemset(patch_b_buf.p, 0, patch_b_buf.bytes));
        CUDA_CHECK(cudaMemset(pos_embed_buf.p, 0, pos_embed_buf.bytes));

        view.patch_embedding = make_bf16_weight(patch_w_buf.p, VisionTowerConfig::hidden, VisionTowerConfig::patch_dim);
        view.patch_embedding_bias = Tensor(patch_b_buf.p, DType::BF16, {VisionTowerConfig::hidden});
        view.position_embedding = Tensor(pos_embed_buf.p, DType::BF16, {VisionTowerConfig::position_embeddings, VisionTowerConfig::hidden});

        for (std::size_t i = 0; i < VisionTowerConfig::layers; ++i) {
            auto& s = layers[i];
            CUDA_CHECK(cudaMemset(s.norm1_w.p, 0, s.norm1_w.bytes));
            CUDA_CHECK(cudaMemset(s.norm1_b.p, 0, s.norm1_b.bytes));
            CUDA_CHECK(cudaMemset(s.qkv_w.p, 0, s.qkv_w.bytes));
            CUDA_CHECK(cudaMemset(s.qkv_b.p, 0, s.qkv_b.bytes));
            CUDA_CHECK(cudaMemset(s.proj_w.p, 0, s.proj_w.bytes));
            CUDA_CHECK(cudaMemset(s.proj_b.p, 0, s.proj_b.bytes));
            CUDA_CHECK(cudaMemset(s.norm2_w.p, 0, s.norm2_w.bytes));
            CUDA_CHECK(cudaMemset(s.norm2_b.p, 0, s.norm2_b.bytes));
            CUDA_CHECK(cudaMemset(s.fc1_w.p, 0, s.fc1_w.bytes));
            CUDA_CHECK(cudaMemset(s.fc1_b.p, 0, s.fc1_b.bytes));
            CUDA_CHECK(cudaMemset(s.fc2_w.p, 0, s.fc2_w.bytes));
            CUDA_CHECK(cudaMemset(s.fc2_b.p, 0, s.fc2_b.bytes));

            auto& lw = view.layers[i];
            lw.norm1_weight = Tensor(s.norm1_w.p, DType::BF16, {VisionTowerConfig::hidden});
            lw.norm1_bias   = Tensor(s.norm1_b.p, DType::BF16, {VisionTowerConfig::hidden});
            lw.qkv          = make_bf16_weight(s.qkv_w.p, 3 * VisionTowerConfig::hidden, VisionTowerConfig::hidden);
            lw.qkv_bias     = Tensor(s.qkv_b.p, DType::BF16, {3 * VisionTowerConfig::hidden});
            lw.output       = make_bf16_weight(s.proj_w.p, VisionTowerConfig::hidden, VisionTowerConfig::hidden);
            lw.output_bias  = Tensor(s.proj_b.p, DType::BF16, {VisionTowerConfig::hidden});
            lw.norm2_weight = Tensor(s.norm2_w.p, DType::BF16, {VisionTowerConfig::hidden});
            lw.norm2_bias   = Tensor(s.norm2_b.p, DType::BF16, {VisionTowerConfig::hidden});
            lw.fc1          = make_bf16_weight(s.fc1_w.p, VisionTowerConfig::intermediate, VisionTowerConfig::hidden);
            lw.fc1_bias     = Tensor(s.fc1_b.p, DType::BF16, {VisionTowerConfig::intermediate});
            lw.fc2          = make_bf16_weight(s.fc2_w.p, VisionTowerConfig::hidden, VisionTowerConfig::intermediate);
            lw.fc2_bias     = Tensor(s.fc2_b.p, DType::BF16, {VisionTowerConfig::hidden});
        }

        CUDA_CHECK(cudaMemset(merger_norm_w.p, 0, merger_norm_w.bytes));
        CUDA_CHECK(cudaMemset(merger_norm_b.p, 0, merger_norm_b.bytes));
        CUDA_CHECK(cudaMemset(merger_fc1_w.p, 0, merger_fc1_w.bytes));
        CUDA_CHECK(cudaMemset(merger_fc1_b.p, 0, merger_fc1_b.bytes));
        CUDA_CHECK(cudaMemset(merger_fc2_w.p, 0, merger_fc2_w.bytes));
        CUDA_CHECK(cudaMemset(merger_fc2_b.p, 0, merger_fc2_b.bytes));

        view.merger_norm_weight = Tensor(merger_norm_w.p, DType::BF16, {VisionTowerConfig::hidden});
        view.merger_norm_bias   = Tensor(merger_norm_b.p, DType::BF16, {VisionTowerConfig::hidden});
        view.merger_fc1         = make_bf16_weight(merger_fc1_w.p, VisionTowerConfig::merger_hidden, VisionTowerConfig::merger_hidden);
        view.merger_fc1_bias    = Tensor(merger_fc1_b.p, DType::BF16, {VisionTowerConfig::merger_hidden});
        view.merger_fc2         = make_bf16_weight(merger_fc2_w.p, 2560, VisionTowerConfig::merger_hidden);
        view.merger_fc2_bias    = Tensor(merger_fc2_b.p, DType::BF16, {2560});
        device.synchronize();
    }
};

int test_vision_adapter() {
    VisionModelView v{};
    std::uint16_t dummy = 0;
    v.patch_embedding = make_bf16_weight(&dummy, 1152, 1536);
    v.patch_embedding_bias = Tensor(&dummy, DType::BF16, {1152});
    v.position_embedding = Tensor(&dummy, DType::BF16, {2304, 1152});

    for (std::size_t i = 0; i < v.layers.size(); ++i) {
        auto& l = v.layers[i];
        l.qkv = make_bf16_weight(&dummy, 3456, 1152);
        l.qkv_bias = Tensor(&dummy, DType::BF16, {3456});
        l.output = make_bf16_weight(&dummy, 1152, 1152);
        l.output_bias = Tensor(&dummy, DType::BF16, {1152});
        l.fc1 = make_bf16_weight(&dummy, 4304, 1152);
        l.fc1_bias = Tensor(&dummy, DType::BF16, {4304});
        l.fc2 = make_bf16_weight(&dummy, 1152, 4304);
        l.fc2_bias = Tensor(&dummy, DType::BF16, {1152});
        l.norm1_weight = Tensor(&dummy, DType::BF16, {1152});
        l.norm1_bias = Tensor(&dummy, DType::BF16, {1152});
        l.norm2_weight = Tensor(&dummy, DType::BF16, {1152});
        l.norm2_bias = Tensor(&dummy, DType::BF16, {1152});
    }

    v.merger_fc1 = make_bf16_weight(&dummy, 4608, 4608);
    v.merger_fc1_bias = Tensor(&dummy, DType::BF16, {4608});
    v.merger_fc2 = make_bf16_weight(&dummy, 2560, 4608);
    v.merger_fc2_bias = Tensor(&dummy, DType::BF16, {2560});
    v.merger_norm_weight = Tensor(&dummy, DType::BF16, {1152});
    v.merger_norm_bias = Tensor(&dummy, DType::BF16, {1152});

    const auto adapted = adapt_vision_weights(v);
    if (adapted.output_hidden != 2560) {
        std::cerr << "adapted.output_hidden != 2560\n";
        return 1;
    }
    if (adapted.patch_embed == nullptr || adapted.position_embed == nullptr ||
        adapted.merger.fc2 == nullptr) {
        std::cerr << "adapted weight pointer is null\n";
        return 1;
    }

    std::cout << "PASS: test_vision_adapter" << std::endl;
    return 0;
}

int test_vision_session_lifecycle(DeviceContext& device) {
    SyntheticVisionModel synth_vision(device);
    FlashNextVisionSession session(synth_vision.view, device, 512);

    if (session.workspace_capacity_bytes() == 0) {
        std::cerr << "workspace_capacity_bytes is 0" << std::endl;
        return 1;
    }

    auto summary_before = session.memory_summary(256);
    if (summary_before.max_item_tokens != 512 || summary_before.handoff_active_bytes != 0) {
        std::cerr << "summary_before invalid" << std::endl;
        return 1;
    }

    // Prepare 4 patches -> 1 merged token
    std::vector<std::uint16_t> patches(4 * VisionTowerConfig::patch_dim, 0x3c00);
    std::vector<std::int32_t> pos_ids(4 * 2, 0);
    std::vector<std::int32_t> table_idx(4 * 4, 0);
    std::vector<float> table_weights(4 * 4, 1.0f);

    ninfer::targets::qwen3_6::VisionItemControl item_control{
        .patch_count            = 4,
        .merged_count           = 1,
        .segment_length         = 4,
        .position_ids           = pos_ids,
        .scatter_indices        = {5},
        .position_table_indices = table_idx,
        .position_table_weights = table_weights,
    };

    Tensor out = session.encode(item_control, patches, device.stream);
    if (out.ne[0] != 2560 || out.ne[1] != 1) {
        std::cerr << "session.encode output shape mismatch: [" << out.ne[0] << ", " << out.ne[1] << "]" << std::endl;
        return 1;
    }
    if (session.vision_tokens() != 1) {
        std::cerr << "session.vision_tokens mismatch" << std::endl;
        return 1;
    }
    if (session.elapsed_seconds() < 0.0) {
        std::cerr << "session.elapsed_seconds invalid" << std::endl;
        return 1;
    }

    auto summary_after = session.memory_summary(256);
    if (summary_after.handoff_active_bytes != 2560 * sizeof(std::uint16_t)) {
        std::cerr << "summary_after handoff_active_bytes mismatch" << std::endl;
        return 1;
    }

    session.retire_handoff();
    auto summary_retired = session.memory_summary(256);
    if (summary_retired.handoff_active_bytes != 0) {
        std::cerr << "summary_retired handoff_active_bytes not 0" << std::endl;
        return 1;
    }

    std::cout << "PASS: test_vision_session_lifecycle" << std::endl;
    return 0;
}

int test_program_vision_request_and_chunk_clipping(DeviceContext& device) {
    std::cout << "Step 1: creating models..." << std::endl;
    auto synth_text = make_synthetic_model(device);
    SyntheticVisionModel synth_vision(device);

    std::cout << "Step 2: creating program..." << std::endl;
    FlashNextRuntimeConfig cfg{
        .max_concurrency     = 1,
        .max_context         = 512,
        .state_slot_capacity = 2,
        .prefill_chunk       = 128,
        .use_cuda_graph      = false,
        .vision_enabled      = true,
        .max_vision_tokens   = 512,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    PleIndexMetadata ple_meta{};
    ple_meta.multipliers.fill(0);
    ple_meta.head_offsets.fill(0);
    ple_meta.head_vocab_sizes.fill(1);

    auto program_impl = std::make_unique<ProgramImpl>(
        nullptr, plan, device, synth_text.view, synth_vision.view, ple_meta);
    Program program(std::move(program_impl));

    std::cout << "Step 3: creating prompt data..." << std::endl;
    // Construct a PreparedPrompt with 1 vision item (4 image tokens from token 10 to 13)
    // Total tokens: 24 (0..9 text, 10..13 image, 14..23 text)
    ninfer::targets::qwen3_6::PreparedPromptData prompt_data;
    prompt_data.token_ids.resize(24);
    prompt_data.token_types.resize(24, 0);
    prompt_data.positions.resize(24 * 3);
    for (std::size_t i = 0; i < 24; ++i) {
        prompt_data.token_ids[i]                  = static_cast<TokenId>(100 + i);
        prompt_data.positions[i]                  = static_cast<std::int32_t>(i);
        prompt_data.positions[24 + i]             = static_cast<std::int32_t>(i);
        prompt_data.positions[48 + i]             = static_cast<std::int32_t>(i);
    }
    // Set tokens 10..13 to type 1 (image)
    for (std::size_t i = 10; i < 14; ++i) {
        prompt_data.token_types[i] = 1;
    }
    prompt_data.prepare.raw_patches   = 16;
    prompt_data.prepare.vision_tokens = 4;

    auto media = std::make_shared<ninfer::targets::qwen3_6::PreparedMediaPayload>();
    media->patch_elements = 16 * VisionTowerConfig::patch_dim;
    media->patches = std::make_unique<std::uint16_t[]>(media->patch_elements);
    std::fill_n(media->patches.get(), media->patch_elements, 0x3c00);
    prompt_data.media_payloads.push_back(std::move(media));

    ninfer::targets::qwen3_6::VisionItem vitem{
        .modality = ninfer::targets::qwen3_6::PromptModality::Image,
        .grid = {.temporal = 1, .height = 4, .width = 4},
        .patch_begin = 0,
        .patch_count = 16,
        .token_spans = {{.begin = 10, .count = 4}},
    };
    prompt_data.vision_items.push_back(vitem);

    auto prompt = ninfer::targets::qwen3_6::PreparedPromptAccess::construct(std::move(prompt_data));

    std::cout << "Step 4: planning request..." << std::endl;
    ninfer::runtime::ResolvedExecutionOptions exec_options{};
    exec_options.requested_output_tokens = 16;
    exec_options.sampling.temperature   = 0.7F;
    exec_options.sampling.top_p         = 0.8F;

    auto base_plan = program.plan_request(prompt, exec_options);
    if (base_plan.summary().prompt_tokens != 24) {
        std::cerr << "prompt_tokens != 24" << std::endl;
        return 1;
    }

    std::cout << "Step 5: admitting..." << std::endl;
    ninfer::runtime::ContextMachineCostModel cost_model{};
    auto candidate = program.inspect_admission(
        prompt, base_plan, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
    if (!candidate.has_value()) {
        std::cerr << "candidate not present" << std::endl;
        return 1;
    }

    std::cout << "Step 6: sealing..." << std::endl;
    auto resource_plan = program.seal_identity(*candidate, prompt);
    if (!resource_plan.has_value()) {
        std::cerr << "resource_plan not present" << std::endl;
        return 1;
    }

    std::cout << "Step 7: start resource transaction..." << std::endl;
    std::atomic<bool> cancel{false};
    ninfer::runtime::CancellationFlagView cancel_view{&cancel};
    auto res_status = program.start_resource_transaction(
        std::move(*resource_plan), std::move(prompt), cancel_view);
    if (res_status != ninfer::runtime::ContextTransactionReserveStatus::Reserved) {
        std::cerr << "start_resource_transaction failed" << std::endl;
        return 1;
    }

    std::cout << "Step 8: progressing context transaction..." << std::endl;
    auto progress = program.progress_context_transaction(cancel_view);
    auto* mat = std::get_if<MaterializationResult>(&progress);
    if (mat == nullptr || !mat->published.has_value()) {
        std::cerr << "progress failed" << std::endl;
        return 1;
    }
    SequenceHandle seq = mat->published->sequence;
    program.finalize_context_transaction();

    std::cout << "Step 9: advance prefill chunk 1..." << std::endl;
    auto p1 = program.advance_prefill(seq);
    if (p1.processed_prompt_tokens != 10 || p1.complete) {
        std::cerr << "Chunk 1 processed_prompt_tokens expected 10, got " << p1.processed_prompt_tokens << std::endl;
        return 1;
    }

    std::cout << "Step 10: advance prefill chunk 2 (image)..." << std::endl;
    auto p2 = program.advance_prefill(seq);
    if (p2.processed_prompt_tokens != 4 || p2.complete) {
        std::cerr << "Chunk 2 processed_prompt_tokens expected 4, got " << p2.processed_prompt_tokens << std::endl;
        return 1;
    }

    std::cout << "Step 11: advance prefill chunk 3..." << std::endl;
    auto p3 = program.advance_prefill(seq);
    if (p3.processed_prompt_tokens != 10 || !p3.complete) {
        std::cerr << "Chunk 3 processed_prompt_tokens expected 10 (complete), got " << p3.processed_prompt_tokens << std::endl;
        return 1;
    }

    std::cout << "Step 12: memory summary..." << std::endl;
    auto mem = program.memory_summary();
    if (!mem.vision_workspace.has_value()) {
        std::cerr << "mem.vision_workspace not present" << std::endl;
        return 1;
    }

    std::cout << "PASS: test_program_vision_request_and_chunk_clipping" << std::endl;
    return 0;
}

} // namespace

int main() {
    std::cout << "Starting Flash-Next Vision Execute Tests..." << std::endl;
    try {
        if (test_vision_adapter() != 0) return 1;

        int device_count = 0;
        const cudaError_t count_error = cudaGetDeviceCount(&device_count);
        if (cuda_unavailable(count_error) || device_count == 0) {
            std::cout << "SKIP: CUDA device tests (no usable device)" << std::endl;
            return 0;
        }
        CUDA_CHECK(count_error);

        ninfer::DeviceContext device(0);
        if (test_vision_session_lifecycle(device) != 0) return 1;
        if (test_program_vision_request_and_chunk_clipping(device) != 0) return 1;
    } catch (const std::exception& ex) {
        std::cerr << "Fatal exception: " << ex.what() << std::endl;
        return 1;
    }
    std::cout << "OK Flash-Next Vision Execute Tests" << std::endl;
    return 0;
}
