#include "core/arena.h"
#include "core/device.h"
#include "runtime/contract/types.h"
#include "runtime/engine/context_cost.h"
#include "targets/qwen3_8_flash_next/impl/lane_ledger.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"
#include "targets/qwen3_8_flash_next/impl/mtp_forward.h"
#include "targets/qwen3_8_flash_next/impl/ple_index.h"
#include "targets/qwen3_8_flash_next/impl/ple_table.h"
#include "targets/qwen3_8_flash_next/impl/program_impl.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"
#include "targets/qwen3_8_flash_next/impl/runtime_state.h"
#include "targets/qwen3_8_flash_next/impl/text_decode.h"
#include "targets/qwen3_8_flash_next/impl/text_executor.h"

#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include <ninfer/targets/qwen3_8_flash_next/runtime.h>
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

float bf16_to_float(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::uint16_t float_to_bf16(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t lsb  = (bits >> 16U) & 1U;
    const std::uint32_t bias = 0x7FFFU + lsb;
    return static_cast<std::uint16_t>((bits + bias) >> 16U);
}

struct SyntheticModel {
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

    ninfer::DeviceBuffer mtp_weights_buf;

    std::vector<std::byte> ple_table_data;
    ninfer::targets::qwen3_8_flash_next::detail::TextModelView view;
};

SyntheticModel make_synthetic_model(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    SyntheticModel model;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_bf16(-0.02f, 0.02f);
    std::uniform_real_distribution<float> dist_norm(0.98f, 1.02f);

    // 1. Generic BF16 weights (token_embedding, output_head, linear projections)
    constexpr std::uint64_t kOutputHeadBytes = 248'320ULL * 2'560 * 2;
    model.big_bf16_buf = ninfer::DeviceBuffer(kOutputHeadBytes);
    model.big_bf16_buf.fill(0x38);

    // 2. RMSNorm weights (~1.0)
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
        buf.fill(0x18);
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

    model.big_nvfp4_gate_codes_buf.fill(0x22);
    model.big_nvfp4_gate_scales_buf.fill(0x38);
    model.big_nvfp4_down_codes_buf.fill(0x22);
    model.big_nvfp4_down_scales_buf.fill(0x38);

    std::vector<float> divisors(512, 1.0f);
    model.big_divisors_buf.copy_from_host(divisors.data(), divisors.size() * sizeof(float));

    // PLE table
    constexpr std::uint64_t ple_rows         = 1;
    constexpr std::uint64_t ple_width        = 160;
    constexpr std::uint64_t scale_offset     = 256;
    model.ple_table_data = std::vector<std::byte>(scale_offset + (ple_width / 16) * 2, std::byte{0});
    for (std::size_t i = 0; i < ple_width / 2; ++i) {
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
        shard = make_ple_shard_view(model.ple_table_data, ple_rows, ple_width);
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

    // Allocate MTP weights
    constexpr std::uint64_t kMtpWeightBytes = 64ULL * 1024 * 1024;
    model.mtp_weights_buf = ninfer::DeviceBuffer(kMtpWeightBytes);
    model.mtp_weights_buf.fill(0x38);

    MtpModelView mtp_view{};
    mtp_view.embedding_projection = make_bf16_weight(2'560, 2'560);
    mtp_view.hidden_projection    = make_bf16_weight(2'560, 2'560);
    mtp_view.embedding_norm       = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {2'560});
    mtp_view.hidden_norm          = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {10'240});

    mtp_view.mixer.norm           = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {10'240});
    mtp_view.mixer.input_mix_down = make_bf16_weight(320, 10'240);
    mtp_view.mixer.input_mix_up   = make_bf16_weight(10'240, 320);

    mtp_view.attention_hyper.block_inject   = make_bf16_weight_from(model.inject_buf, 4, 10'240);
    mtp_view.attention_hyper.norm           = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {10'240});
    mtp_view.attention_hyper.input_mix_down = make_bf16_weight(320, 10'240);
    mtp_view.attention_hyper.input_mix_up   = make_bf16_weight(10'240, 320);

    mtp_view.attention.indexer_query_key    = make_bf16_weight(640, 2'560);
    mtp_view.attention.indexer_key_norm     = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {128});
    mtp_view.attention.indexer_query_norm   = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {128});
    mtp_view.attention.key_norm             = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {256});
    mtp_view.attention.query_norm           = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {256});
    mtp_view.attention.query_gate_key_value = make_bf16_weight(13'312, 2'560);
    mtp_view.attention.output               = make_bf16_weight(2'560, 6'144);

    mtp_view.mlp_hyper.block_inject   = make_bf16_weight_from(model.inject_buf, 4, 10'240);
    mtp_view.mlp_hyper.norm           = ninfer::Tensor(model.norm_bf16_buf.p, ninfer::DType::BF16, {10'240});
    mtp_view.mlp_hyper.input_mix_down = make_bf16_weight(320, 10'240);
    mtp_view.mlp_hyper.input_mix_up   = make_bf16_weight(10'240, 320);

    mtp_view.moe.router             = make_bf16_weight(512, 2'560);
    mtp_view.moe.shared_down        = make_bf16_weight(2'560, 640);
    mtp_view.moe.shared_gate        = make_bf16_weight(640, 2'560);
    mtp_view.moe.shared_up          = make_bf16_weight(640, 2'560);
    mtp_view.moe.shared_gate_weight = make_bf16_weight_from(model.shared_gate_weight_buf, 1, 2'560);
    // Sequence 12: the MTP draft expert bank is NVFP4 (same view type as the
    // text layers), so the synthetic draft head reuses the text-layer banks.
    mtp_view.moe.expert_gate_up     = Nvfp4ExpertBankView{
        .codes                  = static_cast<const std::byte*>(model.big_nvfp4_gate_codes_buf.p),
        .scales                 = static_cast<const std::byte*>(model.big_nvfp4_gate_scales_buf.p),
        .weight_scale_divisors  = static_cast<const float*>(model.big_divisors_buf.p),
        .experts                = 512,
        .rows                   = 1'280,
        .columns                = 2'560,
        .code_bytes_per_expert  = gate_code_bytes_per_expert,
        .scale_bytes_per_expert = gate_scale_bytes_per_expert,
    };
    mtp_view.moe.expert_down        = Nvfp4ExpertBankView{
        .codes                  = static_cast<const std::byte*>(model.big_nvfp4_down_codes_buf.p),
        .scales                 = static_cast<const std::byte*>(model.big_nvfp4_down_scales_buf.p),
        .weight_scale_divisors  = static_cast<const float*>(model.big_divisors_buf.p),
        .experts                = 512,
        .rows                   = 2'560,
        .columns                = 640,
        .code_bytes_per_expert  = down_code_bytes_per_expert,
        .scale_bytes_per_expert = down_scale_bytes_per_expert,
    };

    model.view.mtp = mtp_view;
    device.synchronize();
    return model;
}

int test_speculative_greedy_exact_match(ninfer::DeviceContext& device,
                                        const SyntheticModel& model) {
    std::cout << "[TEST 1/2] test_speculative_greedy_exact_match (K=0 vs K=1..4) ...\n" << std::flush;
    using namespace ninfer::targets::qwen3_8_flash_next;
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    try {
        constexpr std::uint32_t kTargetTokens = 10;
        const std::vector<int32_t> prompt = {100, 200, 300, 400};

        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);

        // 1. Baseline generation with K=0 (no speculation)
        std::vector<int32_t> baseline_tokens;
        {
            FlashNextRuntimeConfig cfg{
                .max_concurrency          = 1,
                .max_context              = 512,
                .prefill_chunk            = 512,
                .speculative_draft_tokens = 0,
                .use_cuda_graph           = false,
            };
            const auto curve = flash_next_capacity_curve(cfg);
            auto plan = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);
            FlashNextRuntimeAllocation alloc(plan);
            alloc.initialize(device.stream);
            FlashNextTextExecutor exec(model.view, ple_meta, device, alloc);
            auto lane = exec.allocate_lane();

            std::vector<std::array<int32_t, 3>> mrope_positions(prompt.size(), {0, 0, 0});
            for (size_t i = 0; i < prompt.size(); ++i) {
                mrope_positions[i] = {static_cast<int32_t>(i), static_cast<int32_t>(i), static_cast<int32_t>(i)};
            }

            std::cout << "    Prefilling baseline ...\n" << std::flush;
            auto prefill = exec.execute_prefill_chunk(lane, prompt, mrope_positions, 0);
            std::array<LaneCommitDecision, 1> decision = {{{.accept = true}}};
            prefill.commit(decision);
            std::cout << "    Prefill baseline done.\n" << std::flush;

            int32_t last_token = prompt.back();
            int32_t last_idx   = static_cast<int32_t>(prompt.size() - 1);
            int32_t last_pos   = last_idx;

            for (uint32_t step = 0; step < kTargetTokens; ++step) {
                LaneStepRequest req{
                    .handle          = lane,
                    .token_id        = last_token,
                    .token_index     = last_idx + 1,
                    .mrope_positions = {last_pos + 1, last_pos + 1, last_pos + 1},
                };
                auto round = exec.execute_round(std::span(&req, 1));
                const auto sampled = round.sampled_tokens()[0];
                round.commit(decision);

                baseline_tokens.push_back(sampled);
                last_token = sampled;
                last_idx += 1;
                last_pos += 1;
            }
            exec.release_lane(lane);
        }

        std::cout << "  Baseline (K=0) generated " << baseline_tokens.size() << " tokens:";
        for (auto t : baseline_tokens) { std::cout << " " << t; }
        std::cout << "\n" << std::flush;

        // 2. Speculative decoding for K in {1, 2, 3, 4}
        for (uint32_t K : {1U, 2U, 3U, 4U}) {
            std::cout << "  Testing speculative decode with K=" << K << " ...\n" << std::flush;
            FlashNextRuntimeConfig cfg{
                .max_concurrency          = 1,
                .max_context              = 512,
                .prefill_chunk            = 512,
                .speculative_draft_tokens = K,
                .use_cuda_graph           = false,
            };
            const auto curve = flash_next_capacity_curve(cfg);
            auto plan = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);
            FlashNextRuntimeAllocation alloc(plan);
            alloc.initialize(device.stream);
            FlashNextTextExecutor exec(model.view, ple_meta, device, alloc);
            auto lane = exec.allocate_lane();

        std::vector<std::array<int32_t, 3>> mrope_positions(prompt.size(), {0, 0, 0});
        for (size_t i = 0; i < prompt.size(); ++i) {
            mrope_positions[i] = {static_cast<int32_t>(i), static_cast<int32_t>(i), static_cast<int32_t>(i)};
        }

        auto prefill = exec.execute_prefill_chunk(lane, prompt, mrope_positions, 0);
        std::array<LaneCommitDecision, 1> decision = {{{.accept = true}}};
        prefill.commit(decision);

        int32_t last_token = prompt.back();
        int32_t last_idx   = static_cast<int32_t>(prompt.size() - 1);
        int32_t last_pos   = last_idx;

        std::vector<int32_t> spec_tokens;
        std::vector<int32_t> current_drafts;
        uint32_t total_rounds = 0;

        while (spec_tokens.size() < kTargetTokens) {
            ++total_rounds;
            std::cout << "    spec round " << total_rounds << " (spec_tokens=" << spec_tokens.size()
                      << ", drafts=" << current_drafts.size() << ") ...\n" << std::flush;
            if (current_drafts.empty()) {
                LaneStepRequest req{
                    .handle          = lane,
                    .token_id        = last_token,
                    .token_index     = last_idx + 1,
                    .mrope_positions = {last_pos + 1, last_pos + 1, last_pos + 1},
                };
                auto round = exec.execute_round(std::span(&req, 1));
                const auto sampled = round.sampled_tokens()[0];
                const auto hyper_hidden = round.hyper_hidden();
                round.commit(decision);

                spec_tokens.push_back(sampled);
                last_token = sampled;
                last_idx += 1;
                last_pos += 1;

                const uint32_t rem = kTargetTokens > spec_tokens.size()
                                         ? static_cast<uint32_t>(kTargetTokens - spec_tokens.size())
                                         : 0;
                const uint32_t k = std::min(K, rem);
                if (k > 0) {
                    current_drafts.resize(k);
                    std::cout << "      drafting " << k << " tokens ...\n" << std::flush;
                    exec.draft_mtp_tokens(lane, last_token, last_idx,
                                          {last_pos, last_pos, last_pos}, hyper_hidden,
                                          k, current_drafts);
                    std::cout << "      drafted " << k << " tokens.\n" << std::flush;
                }
            } else {
                std::cout << "      verifying " << current_drafts.size() << " drafts ...\n" << std::flush;
                auto round = exec.execute_speculative_verify_round(
                    lane, last_token, current_drafts, last_idx + 1,
                    {last_pos + 1, last_pos + 1, last_pos + 1}, {});
                const auto sampled = round.sampled_tokens();
                const auto hyper_hidden = round.hyper_hidden();
                std::cout << "      verify round done.\n" << std::flush;

                size_t A = 0;
                for (size_t j = 0; j < current_drafts.size(); ++j) {
                    if (sampled[j] == current_drafts[j]) {
                        ++A;
                    } else {
                        break;
                    }
                }

                std::vector<int32_t> accepted;
                for (size_t j = 0; j < A; ++j) {
                    accepted.push_back(current_drafts[j]);
                }
                accepted.push_back(sampled[A]); // bonus token

                round.commit_speculative(lane.lane_index(), accepted);

                for (auto tok : accepted) {
                    spec_tokens.push_back(tok);
                    last_token = tok;
                    last_idx += 1;
                    last_pos += 1;
                }

                const uint32_t rem = kTargetTokens > spec_tokens.size()
                                         ? static_cast<uint32_t>(kTargetTokens - spec_tokens.size())
                                         : 0;
                const uint32_t k = std::min(K, rem);
                if (k > 0) {
                    current_drafts.resize(k);
                    ninfer::Tensor last_acc_hyper = hyper_hidden.slice(1, static_cast<std::int32_t>(A), 1);
                    exec.draft_mtp_tokens(lane, last_token, last_idx,
                                          {last_pos, last_pos, last_pos}, last_acc_hyper,
                                          k, current_drafts);
                } else {
                    current_drafts.clear();
                }
            }
        }

        exec.release_lane(lane);

        if (spec_tokens.size() < kTargetTokens) {
            std::cerr << "FAIL: speculative decode under-generated tokens\n";
            return 1;
        }
        for (size_t i = 0; i < kTargetTokens; ++i) {
            if (spec_tokens[i] != baseline_tokens[i]) {
                std::cerr << "FAIL: token mismatch at index " << i << " for K=" << K
                          << " (spec=" << spec_tokens[i] << " != base=" << baseline_tokens[i] << ")\n";
                return 1;
            }
        }
        std::cout << "  PASS: K=" << K << " bit-exact match (" << total_rounds << " rounds)\n";
    }

    std::cout << "PASS: test_speculative_greedy_exact_match\n";
    return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_speculative_greedy_exact_match exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "test_speculative_greedy_exact_match unknown exception\n";
        return 1;
    }
}

int test_speculative_rollback_on_mismatch(ninfer::DeviceContext& device,
                                           const SyntheticModel& model) {
    std::cout << "[TEST 2/2] test_speculative_rollback_on_mismatch ...\n" << std::flush;
    using namespace ninfer::targets::qwen3_8_flash_next;
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    try {
        PleIndexMetadata ple_meta{};
        ple_meta.multipliers = {1, 2, 3};
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);

        FlashNextRuntimeConfig cfg{
            .max_concurrency          = 1,
            .max_context              = 512,
            .prefill_chunk            = 512,
            .speculative_draft_tokens = 3,
            .use_cuda_graph           = false,
        };
        const auto curve = flash_next_capacity_curve(cfg);
        auto plan = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);
        FlashNextRuntimeAllocation alloc(plan);
        alloc.initialize(device.stream);
        FlashNextTextExecutor exec(model.view, ple_meta, device, alloc);
        auto lane = exec.allocate_lane();

        const std::vector<int32_t> prompt = {100, 200, 300, 400};
        std::vector<std::array<int32_t, 3>> mrope_positions(prompt.size(), {0, 0, 0});
        for (size_t i = 0; i < prompt.size(); ++i) {
            mrope_positions[i] = {static_cast<int32_t>(i), static_cast<int32_t>(i), static_cast<int32_t>(i)};
        }

        auto prefill = exec.execute_prefill_chunk(lane, prompt, mrope_positions, 0);
        std::array<LaneCommitDecision, 1> decision = {{{.accept = true}}};
        prefill.commit(decision);

        std::vector<int32_t> bogus_drafts = {99999, 88888, 77777};
        auto round = exec.execute_speculative_verify_round(
            lane, prompt.back(), bogus_drafts, static_cast<int32_t>(prompt.size()),
            {static_cast<int32_t>(prompt.size()), static_cast<int32_t>(prompt.size()),
             static_cast<int32_t>(prompt.size())}, {});

        const auto sampled = round.sampled_tokens();
        std::vector<int32_t> accepted = {sampled[0]};
        round.commit_speculative(lane.lane_index(), accepted);

        if (exec.committed_frontier(lane) != static_cast<int32_t>(prompt.size() + 1)) {
            std::cerr << "FAIL: committed frontier incorrect after rollback\n";
            return 1;
        }

        exec.release_lane(lane);
        std::cout << "PASS: test_speculative_rollback_on_mismatch\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_speculative_rollback_on_mismatch exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "test_speculative_rollback_on_mismatch unknown exception\n";
        return 1;
    }
}

int test_speculative_turn_closure_interaction(ninfer::DeviceContext& device,
                                              const SyntheticModel& model) {
    std::cout << "[TEST 3/3] test_speculative_turn_closure_interaction ...\n" << std::flush;
    using namespace ninfer;
    using namespace ninfer::targets::qwen3_8_flash_next;
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    try {
        auto make_prompt = [](std::span<const TokenId> tokens, bool reusable = true,
                              std::optional<std::uint32_t> rewrite_frontier = std::nullopt) {
            ninfer::targets::qwen3_6::PreparedPromptData data;
            const std::size_t num_tokens = tokens.size();
            data.token_ids.assign(tokens.begin(), tokens.end());
            data.token_types.resize(num_tokens, 0);
            data.positions.resize(num_tokens * 3);
            for (std::size_t i = 0; i < num_tokens; ++i) {
                data.positions[i]                  = static_cast<std::int32_t>(i);
                data.positions[num_tokens + i]     = static_cast<std::int32_t>(i);
                data.positions[2 * num_tokens + i] = static_cast<std::int32_t>(i);
            }
            data.identity.reusable = reusable;
            if (rewrite_frontier.has_value()) {
                data.identity.rewrite_checkpoint = ninfer::targets::qwen3_6::RewriteCheckpointSpec{
                    .kind     = ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure,
                    .frontier = *rewrite_frontier,
                };
            }
            return ninfer::targets::qwen3_6::PreparedPromptAccess::construct(std::move(data));
        };

        PleIndexMetadata ple_meta{};
        ple_meta.multipliers.fill(0);
        ple_meta.head_offsets.fill(0);
        ple_meta.head_vocab_sizes.fill(1);

        ninfer::runtime::ResolvedExecutionOptions exec_options{};
        exec_options.requested_output_tokens = 32;
        ninfer::runtime::ContextMachineCostModel cost_model{};
        std::atomic<bool> flag{false};
        ninfer::runtime::CancellationFlagView cancellation{&flag};

        // 1. Session 1 with speculative decoding (K=3): capture turn closure at F=16
        FlashNextRuntimeConfig cfg1{
            .max_concurrency          = 1,
            .max_context              = 512,
            .continuation_capacity    = 4,
            .prefill_chunk            = 512,
            .speculative_draft_tokens = 3,
            .use_cuda_graph           = false,
        };
        const auto curve1 = flash_next_capacity_curve(cfg1);
        auto plan1 = finalize_flash_next_runtime_plan(cfg1, curve1.maximum_main_page_groups);

        auto pimpl1 = std::make_unique<ProgramImpl>(nullptr, plan1, device, model.view, std::nullopt, ple_meta);
        Program program1(std::move(pimpl1));

        std::vector<TokenId> turn1_tokens(20);
        for (std::size_t i = 0; i < 20; ++i) { turn1_tokens[i] = static_cast<TokenId>(100 + i); }
        const auto prompt1 = make_prompt(turn1_tokens, true, 16);

        auto base1 = program1.plan_request(prompt1, exec_options);
        auto cand1 = program1.inspect_admission(
            prompt1, base1, runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
        if (!cand1.has_value()) {
            std::cerr << "FAIL: Session 1 admission failed\n";
            return 1;
        }

        auto res1 = program1.seal_identity(*cand1, prompt1);
        program1.start_resource_transaction(std::move(*res1), make_prompt(turn1_tokens, true, 16),
                                            cancellation);
        auto prog1 = program1.progress_context_transaction(cancellation);
        auto* mat1 = std::get_if<MaterializationResult>(&prog1);
        SequenceHandle seq1 = mat1->published->sequence;
        program1.finalize_context_transaction();

        // Prefill to F=16 and offer capture
        auto prefill_s1 = program1.advance_prefill(seq1);
        if (prefill_s1.complete || !prefill_s1.capture.has_value()) {
            std::cerr << "FAIL: Session 1 prefill step 1 failed to offer capture\n";
            return 1;
        }

        auto reserve_stat = program1.reserve_active_capture(
            std::move(*prefill_s1.capture), nullptr, nullptr, std::nullopt, cancellation);
        if (reserve_stat != runtime::ContextTransactionReserveStatus::Reserved) {
            std::cerr << "FAIL: Capture reservation failed\n";
            return 1;
        }

        auto capture_prog = program1.progress_context_transaction(cancellation);
        auto* cap_res = std::get_if<ActiveCaptureResult>(&capture_prog);
        if (cap_res == nullptr || cap_res->status != runtime::ContextTransactionStatus::Published) {
            std::cerr << "FAIL: Capture publish failed\n";
            return 1;
        }
        program1.finalize_context_transaction();

        // Complete prefill 16..20 and sample first token
        auto prefill_s2 = program1.advance_prefill(seq1);
        if (!prefill_s2.complete || !prefill_s2.pending.has_value()) {
            std::cerr << "FAIL: Session 1 prefill step 2 failed\n";
            return 1;
        }

        std::array<runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = false}}};
        (void)program1.commit(std::move(*prefill_s2.pending), commit_dec);

        // Run 3 speculative decode rounds
        for (int r = 0; r < 3; ++r) {
            auto pending = program1.decode(std::span(&seq1, 1), {});
            std::array<runtime::CommitDecision, 1> dec = {{{.accepted_tokens = 1, .terminal = false}}};
            (void)program1.commit(std::move(pending), dec);
        }

        FinishResult fin1 = program1.finish(seq1);
        if (fin1.disposition != runtime::FinishDisposition::Catalogued || !fin1.continuation.has_value()) {
            std::cerr << "FAIL: Session 1 finish failed to catalogue continuation\n";
            return 1;
        }

        // 2. Multi-turn Session 2 (resuming from TurnClosure at F=16)
        // We will run Session 2 under speculative mode (K=3) and record generated tokens.
        std::vector<TokenId> turn2_tokens(28);
        for (std::size_t i = 0; i < 16; ++i) { turn2_tokens[i] = turn1_tokens[i]; }
        for (std::size_t i = 16; i < 28; ++i) { turn2_tokens[i] = static_cast<TokenId>(500 + i); }
        const auto prompt2 = make_prompt(turn2_tokens, true);

        auto base2 = program1.plan_request(prompt2, exec_options);
        auto cand2 = program1.inspect_admission(
            prompt2, base2, runtime::LaneId(0), &*fin1.continuation, nullptr, std::nullopt, false, cost_model);
        if (!cand2.has_value() || cand2->summary().reusable_prompt_tokens != 16) {
            std::cerr << "FAIL: Session 2 TurnClosure admission failed\n";
            return 1;
        }

        auto res2 = program1.seal_identity(*cand2, prompt2);
        program1.start_resource_transaction(std::move(*res2), make_prompt(turn2_tokens, true),
                                            cancellation);
        auto prog2 = program1.progress_context_transaction(cancellation);
        auto* mat2 = std::get_if<MaterializationResult>(&prog2);
        SequenceHandle seq2 = mat2->published->sequence;
        program1.finalize_context_transaction();

        auto prefill2 = program1.advance_prefill(seq2);
        if (!prefill2.complete || !prefill2.pending.has_value()) {
            std::cerr << "FAIL: Session 2 prefill failed\n";
            return 1;
        }

        std::vector<TokenId> spec_resumed_tokens;
        spec_resumed_tokens.push_back(prefill2.pending->tokens()[0]);
        (void)program1.commit(std::move(*prefill2.pending), commit_dec);

        for (int r = 0; r < 5; ++r) {
            auto pending = program1.decode(std::span(&seq2, 1), {});
            for (size_t t = 0; t < pending.row_counts()[0]; ++t) {
                spec_resumed_tokens.push_back(pending.tokens()[t]);
            }
            std::array<runtime::CommitDecision, 1> dec = {{{.accepted_tokens = 1, .terminal = false}}};
            (void)program1.commit(std::move(pending), dec);
        }
        (void)program1.finish(seq2);

        // 3. Independent unspeculative baseline (K=0) starting fresh from prompt2
        FlashNextRuntimeConfig cfg_base{
            .max_concurrency          = 1,
            .max_context              = 512,
            .continuation_capacity    = 4,
            .prefill_chunk            = 512,
            .speculative_draft_tokens = 0,
            .use_cuda_graph           = false,
        };
        const auto curve_base = flash_next_capacity_curve(cfg_base);
        auto plan_base = finalize_flash_next_runtime_plan(cfg_base, curve_base.maximum_main_page_groups);

        auto pimpl_base = std::make_unique<ProgramImpl>(nullptr, plan_base, device, model.view, std::nullopt, ple_meta);
        Program program_base(std::move(pimpl_base));

        auto base_plan = program_base.plan_request(prompt2, exec_options);
        auto cand_base = program_base.inspect_admission(
            prompt2, base_plan, runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
        auto res_base = program_base.seal_identity(*cand_base, prompt2);
        program_base.start_resource_transaction(std::move(*res_base), make_prompt(turn2_tokens, true),
                                                cancellation);
        auto prog_base = program_base.progress_context_transaction(cancellation);
        auto* mat_base = std::get_if<MaterializationResult>(&prog_base);
        SequenceHandle seq_base = mat_base->published->sequence;
        program_base.finalize_context_transaction();

        auto prefill_base = program_base.advance_prefill(seq_base);
        std::vector<TokenId> base_tokens;
        base_tokens.push_back(prefill_base.pending->tokens()[0]);
        (void)program_base.commit(std::move(*prefill_base.pending), commit_dec);

        for (size_t r = 1; r < spec_resumed_tokens.size(); ++r) {
            auto pending = program_base.decode(std::span(&seq_base, 1), {});
            base_tokens.push_back(pending.tokens()[0]);
            (void)program_base.commit(std::move(pending), commit_dec);
        }
        (void)program_base.finish(seq_base);

        // Verify exact token identity
        for (size_t i = 0; i < spec_resumed_tokens.size() && i < base_tokens.size(); ++i) {
            if (spec_resumed_tokens[i] != base_tokens[i]) {
                std::cerr << "FAIL: resumed token mismatch at " << i
                          << " (spec=" << spec_resumed_tokens[i] << " != base=" << base_tokens[i] << ")\n";
                return 1;
            }
        }

        std::cout << "  PASS: TurnClosure speculative resume bit-exact match ("
                  << spec_resumed_tokens.size() << " tokens verified)\n";
        std::cout << "PASS: test_speculative_turn_closure_interaction\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_speculative_turn_closure_interaction exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "test_speculative_turn_closure_interaction unknown exception\n";
        return 1;
    }
}

} // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    int device_count = 0;
    const auto err   = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(err) || device_count == 0) {
        std::cerr << "skip: CUDA device unavailable\n";
        return 77;
    }
    if (err != cudaSuccess) {
        std::cerr << "CUDA initialization error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    ninfer::DeviceContext device(0);
    auto model = make_synthetic_model(device);

    if (test_speculative_greedy_exact_match(device, model) != 0) return 1;
    if (test_speculative_rollback_on_mismatch(device, model) != 0) return 1;
    if (test_speculative_turn_closure_interaction(device, model) != 0) return 1;

    std::cout << "ALL MTP SPECULATIVE DECODING TESTS PASSED\n";
    return 0;
}
