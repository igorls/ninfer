#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include <ninfer/targets/qwen3_8_flash_next/runtime.h>
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "core/arena.h"
#include "core/device.h"
#include "runtime/contract/types.h"
#include "runtime/engine/context_cost.h"
#include "targets/qwen3_8_flash_next/impl/expert_bank.h"
#include "targets/qwen3_8_flash_next/impl/load/materialized.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"
#include "targets/qwen3_8_flash_next/impl/ple_index.h"
#include "targets/qwen3_8_flash_next/impl/ple_table.h"
#include "targets/qwen3_8_flash_next/impl/program_impl.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::targets::qwen3_8_flash_next;
using namespace ninfer::targets::qwen3_8_flash_next::detail;

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::fflush(stderr);
    return 1;
}

inline std::uint16_t float_to_bf16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(float));
    return static_cast<std::uint16_t>(x >> 16);
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



ninfer::targets::qwen3_6::PreparedPrompt make_prompt(std::span<const ninfer::TokenId> tokens,
                                                     bool reusable = true,
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
}

ninfer::targets::qwen3_6::PreparedPrompt make_prompt_with_cache_opportunities(
    std::span<const ninfer::TokenId> tokens,
    std::vector<ninfer::targets::qwen3_6::PreparedCacheOpportunity> opportunities) {
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
    data.identity.reusable = true;
    data.context_cache.opportunities = std::move(opportunities);
    return ninfer::targets::qwen3_6::PreparedPromptAccess::construct(std::move(data));
}

PleIndexMetadata make_synthetic_ple_meta() {
    PleIndexMetadata ple_meta{};
    ple_meta.multipliers.fill(0);
    ple_meta.head_offsets.fill(0);
    ple_meta.head_vocab_sizes.fill(1);
    return ple_meta;
}

int test_prefix_shortlist_key(ninfer::DeviceContext& device) {
    std::printf("[RUN] test_prefix_shortlist_key\n");
    std::fflush(stdout);
    int failures = 0;
    auto model   = make_synthetic_model(device);
    auto ple_meta = make_synthetic_ple_meta();

    FlashNextRuntimeConfig cfg{
        .max_concurrency       = 2,
        .max_context           = 512,
        .state_slot_capacity   = 0,
        .continuation_capacity = 4,
        .prefill_chunk         = 256,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    auto program_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, model.view, std::nullopt, ple_meta);
    Program program(std::move(program_impl));

    std::vector<ninfer::TokenId> tokens = {100, 101, 102, 103, 104, 105, 106, 107};
    const auto prompt                   = make_prompt(tokens);

    ninfer::runtime::ResolvedExecutionOptions exec_options{};
    exec_options.requested_output_tokens = 16;
    auto base_plan                       = program.plan_request(prompt, exec_options);

    auto key0 = base_plan.prefix_shortlist_key(0);
    failures += check(!key0.has_value(), "Frontier 0 must have no shortlist key");

    auto key4 = base_plan.prefix_shortlist_key(4);
    failures += check(key4.has_value(), "Frontier 4 must have a shortlist key");
    if (key4.has_value()) {
        failures += check(key4->frontier == 4, "Key frontier must be 4");
        failures += check(key4->digests != std::array<std::uint64_t, 2>{}, "Key digests must be non-zero");
    }

    auto key8 = base_plan.prefix_shortlist_key(8);
    failures += check(key8.has_value(), "Frontier 8 must have a shortlist key");
    if (key4.has_value() && key8.has_value()) {
        failures += check(key4->digests != key8->digests, "Digests for different frontiers must differ");
    }

    auto key9 = base_plan.prefix_shortlist_key(9);
    failures += check(!key9.has_value(), "Frontier 9 (exceeding prompt) must have no shortlist key");

    std::printf("[DONE] test_prefix_shortlist_key, failures: %d\n", failures);
    std::fflush(stdout);
    return failures;
}

int test_continuation_lifecycle_and_reuse(ninfer::DeviceContext& device) {
    std::printf("[RUN] test_continuation_lifecycle_and_reuse\n");
    std::fflush(stdout);
    int failures = 0;
    auto model   = make_synthetic_model(device);
    auto ple_meta = make_synthetic_ple_meta();

    FlashNextRuntimeConfig cfg{
        .max_concurrency       = 2,
        .max_context           = 512,
        .state_slot_capacity   = 0,
        .continuation_capacity = 4,
        .prefill_chunk         = 256,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    auto program_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, model.view, std::nullopt, ple_meta);
    Program program(std::move(program_impl));

    std::printf("  [Turn 1] Planning request...\n");
    std::fflush(stdout);

    // Turn 1: 16 tokens
    std::vector<ninfer::TokenId> turn1_tokens(16);
    for (std::size_t i = 0; i < 16; ++i) { turn1_tokens[i] = static_cast<ninfer::TokenId>(200 + i); }
    const auto prompt1 = make_prompt(turn1_tokens, true);

    ninfer::runtime::ResolvedExecutionOptions exec_options1{};
    exec_options1.requested_output_tokens = 16;
    auto base_plan1                       = program.plan_request(prompt1, exec_options1);
    failures += check(base_plan1.summary().publish_continuation,
                      "Turn 1 base plan must have publish_continuation=true");

    std::printf("  [Turn 1] Inspecting admission...\n");
    std::fflush(stdout);

    ninfer::runtime::ContextMachineCostModel cost_model{};
    auto candidate1 = program.inspect_admission(
        prompt1, base_plan1, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
    failures += check(candidate1.has_value(), "Turn 1 admission must succeed");
    failures += check(candidate1->summary().prefix_reuse_path == ninfer::PrefixReusePath::Root,
                      "Turn 1 reuse path must be Root");

    auto resource_plan1 = program.seal_identity(*candidate1, prompt1);
    failures += check(resource_plan1.has_value(), "Turn 1 seal identity must succeed");

    std::printf("  [Turn 1] Starting resource transaction...\n");
    std::fflush(stdout);

    std::atomic<bool> flag1{false};
    ninfer::runtime::CancellationFlagView cancellation1{&flag1};
    auto reserve_status1 = program.start_resource_transaction(
        std::move(*resource_plan1), make_prompt(turn1_tokens, true), cancellation1);
    failures += check(
        reserve_status1 == ninfer::runtime::ContextTransactionReserveStatus::Reserved,
        "Turn 1 transaction must reserve");

    std::printf("  [Turn 1] Progressing transaction...\n");
    std::fflush(stdout);

    auto progress1 = program.progress_context_transaction(cancellation1);
    auto* mat1     = std::get_if<MaterializationResult>(&progress1);
    failures += check(mat1 != nullptr && mat1->published.has_value(),
                      "Turn 1 transaction progress must publish sequence");
    SequenceHandle seq1 = mat1->published->sequence;
    program.finalize_context_transaction();

    auto advance_prefill_full = [&](Program& prog, SequenceHandle seq) -> PrefillProgress {
        auto p = prog.advance_prefill(seq);
        if (p.capture.has_value()) {
            prog.skip_capture(std::move(*p.capture));
            p = prog.advance_prefill(seq);
        }
        return p;
    };

    std::printf("  [Turn 1] Advancing prefill...\n");
    std::fflush(stdout);

    auto prefill_prog1 = advance_prefill_full(program, seq1);
    failures += check(prefill_prog1.complete, "Turn 1 prefill must be complete");
    failures += check(prefill_prog1.processed_prompt_tokens == 16,
                      "Turn 1 processed prompt tokens on completion step must be 16");

    std::printf("  [Turn 1] Finishing sequence...\n");
    std::fflush(stdout);

    FinishResult finish1 = program.finish(seq1);
    failures += check(finish1.disposition == ninfer::runtime::FinishDisposition::Catalogued,
                      "Turn 1 finish disposition must be Catalogued");
    failures += check(finish1.continuation.has_value(),
                      "Turn 1 finish result must contain ContinuationHandle");
    failures += check(finish1.summary.endpoint.has_value(),
                      "Turn 1 finish summary must have endpoint checkpoint");
    if (finish1.summary.endpoint.has_value()) {
        failures += check(finish1.summary.endpoint->ref.frontier == 16,
                          "Endpoint frontier must be 16");
    }

    std::printf("  [Turn 1] Checking physical usage...\n");
    std::fflush(stdout);

    failures += check(program.physical_usage().device_state_slots == 1,
                      "Physical usage must report 1 device state slot for catalogued continuation");

    ContinuationHandle cont_handle = std::move(*finish1.continuation);

    std::printf("  [Turn 2] Planning request...\n");
    std::fflush(stdout);

    // Turn 2: Turn 1 (16 tokens) + Turn 2 delta (8 tokens) = 24 tokens
    std::vector<ninfer::TokenId> turn2_tokens = turn1_tokens;
    for (std::size_t i = 0; i < 8; ++i) {
        turn2_tokens.push_back(static_cast<ninfer::TokenId>(300 + i));
    }
    const auto prompt2 = make_prompt(turn2_tokens, true);

    ninfer::runtime::ResolvedExecutionOptions exec_options2{};
    exec_options2.requested_output_tokens = 16;
    auto base_plan2                       = program.plan_request(prompt2, exec_options2);

    std::printf("  [Turn 2] Inspecting admission with continuation...\n");
    std::fflush(stdout);

    auto candidate2 = program.inspect_admission(
        prompt2, base_plan2, ninfer::runtime::LaneId(0), &cont_handle, nullptr, std::nullopt, false, cost_model);
    failures += check(candidate2.has_value(), "Turn 2 admission must succeed");
    failures += check(candidate2->summary().reusable_prompt_tokens == 16,
                      "Turn 2 reusable prompt tokens must be 16");
    failures += check(candidate2->summary().prefix_reuse_path == ninfer::PrefixReusePath::PrivateEndpoint,
                      "Turn 2 prefix reuse path must be PrivateEndpoint");

    auto resource_plan2 = program.seal_identity(*candidate2, prompt2);
    failures += check(resource_plan2.has_value(), "Turn 2 seal identity must succeed");

    std::printf("  [Turn 2] Starting resource transaction (Copy-on-Resume)...\n");
    std::fflush(stdout);

    std::atomic<bool> flag2{false};
    ninfer::runtime::CancellationFlagView cancellation2{&flag2};
    auto reserve_status2 = program.start_resource_transaction(
        std::move(*resource_plan2), make_prompt(turn2_tokens, true), cancellation2);
    failures += check(
        reserve_status2 == ninfer::runtime::ContextTransactionReserveStatus::Reserved,
        "Turn 2 transaction must reserve");

    std::printf("  [Turn 2] Progressing transaction...\n");
    std::fflush(stdout);

    auto progress2 = program.progress_context_transaction(cancellation2);
    auto* mat2     = std::get_if<MaterializationResult>(&progress2);
    failures += check(mat2 != nullptr && mat2->published.has_value(),
                      "Turn 2 transaction progress must publish sequence");
    SequenceHandle seq2 = mat2->published->sequence;
    program.finalize_context_transaction();

    std::printf("  [Turn 2] Advancing prefill (delta tokens)...\n");
    std::fflush(stdout);

    // Advance prefill Turn 2: only 8 delta tokens!
    auto prefill_prog2 = advance_prefill_full(program, seq2);
    failures += check(prefill_prog2.complete, "Turn 2 prefill must be complete");
    failures += check(prefill_prog2.summary.reused_prompt_tokens == 16,
                      "Turn 2 summary reused prompt tokens must be 16");

    std::printf("  [Turn 2] Finishing sequence...\n");
    std::fflush(stdout);

    FinishResult finish2 = program.finish(seq2);
    failures += check(finish2.disposition == ninfer::runtime::FinishDisposition::Catalogued,
                      "Turn 2 finish disposition must be Catalogued");

    std::printf("[DONE] test_continuation_lifecycle_and_reuse, failures: %d\n", failures);
    std::fflush(stdout);
    return failures;
}

int test_continuation_mismatch_fallback(ninfer::DeviceContext& device) {
    std::printf("[RUN] test_continuation_mismatch_fallback\n");
    std::fflush(stdout);
    int failures = 0;
    auto model   = make_synthetic_model(device);
    auto ple_meta = make_synthetic_ple_meta();

    FlashNextRuntimeConfig cfg{
        .max_concurrency       = 2,
        .max_context           = 512,
        .state_slot_capacity   = 0,
        .continuation_capacity = 4,
        .prefill_chunk         = 256,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    auto program_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, model.view, std::nullopt, ple_meta);
    Program program(std::move(program_impl));

    auto advance_prefill_full = [&](Program& prog, SequenceHandle s) -> PrefillProgress {
        auto p = prog.advance_prefill(s);
        if (p.capture.has_value()) {
            prog.skip_capture(std::move(*p.capture));
            p = prog.advance_prefill(s);
        }
        return p;
    };

    std::vector<ninfer::TokenId> turn1_tokens = {10, 11, 12, 13, 14, 15, 16, 17};
    const auto prompt1                       = make_prompt(turn1_tokens, true);

    ninfer::runtime::ResolvedExecutionOptions exec_options{};
    exec_options.requested_output_tokens = 8;
    auto base_plan1                       = program.plan_request(prompt1, exec_options);

    ninfer::runtime::ContextMachineCostModel cost_model{};
    auto candidate1 = program.inspect_admission(
        prompt1, base_plan1, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
    auto resource_plan1 = program.seal_identity(*candidate1, prompt1);

    std::atomic<bool> flag{false};
    ninfer::runtime::CancellationFlagView cancellation{&flag};
    (void)program.start_resource_transaction(std::move(*resource_plan1), make_prompt(turn1_tokens, true),
                                             cancellation);
    auto progress1     = program.progress_context_transaction(cancellation);
    auto* mat1         = std::get_if<MaterializationResult>(&progress1);
    SequenceHandle seq = mat1->published->sequence;
    program.finalize_context_transaction();

    (void)advance_prefill_full(program, seq);
    FinishResult finish1 = program.finish(seq);
    ContinuationHandle cont = std::move(*finish1.continuation);

    // Prompt 2: diverges at token index 2 (10, 11, 999, ...)
    std::vector<ninfer::TokenId> divergent_tokens = {10, 11, 999, 13, 14, 15, 16, 17, 18, 19};
    const auto prompt2                            = make_prompt(divergent_tokens, true);
    auto base_plan2                               = program.plan_request(prompt2, exec_options);

    auto candidate2 = program.inspect_admission(
        prompt2, base_plan2, ninfer::runtime::LaneId(0), &cont, nullptr, std::nullopt, false, cost_model);
    failures += check(!candidate2.has_value(), "Admission on divergent prompt with mismatching source must return nullopt");

    auto candidate_root = program.inspect_admission(
        prompt2, base_plan2, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
    failures += check(candidate_root.has_value(), "Admission on root fallback must succeed");
    if (candidate_root.has_value()) {
        failures += check(candidate_root->summary().reusable_prompt_tokens == 0,
                          "Root prompt must have 0 reusable prompt tokens");
        failures += check(candidate_root->summary().prefix_reuse_path == ninfer::PrefixReusePath::Root,
                          "Root prompt must fallback to Root reuse path");
    }

    // Release continuation cleanly
    (void)program.release_continuation(std::move(cont));
    failures += check(program.physical_usage().device_state_slots == 0,
                      "Releasing continuation must return device state slots to 0");

    std::printf("[DONE] test_continuation_mismatch_fallback, failures: %d\n", failures);
    std::fflush(stdout);
    return failures;
}

int test_continuation_lru_eviction(ninfer::DeviceContext& device) {
    std::printf("[RUN] test_continuation_lru_eviction\n");
    std::fflush(stdout);
    int failures = 0;
    auto model   = make_synthetic_model(device);
    auto ple_meta = make_synthetic_ple_meta();

    // Capacity of only 2 continuation slots
    FlashNextRuntimeConfig cfg{
        .max_concurrency       = 2,
        .max_context           = 512,
        .state_slot_capacity   = 0,
        .continuation_capacity = 2,
        .prefill_chunk         = 256,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    auto program_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, model.view, std::nullopt, ple_meta);
    Program program(std::move(program_impl));

    auto advance_prefill_full = [&](Program& prog, SequenceHandle s) -> PrefillProgress {
        auto p = prog.advance_prefill(s);
        if (p.capture.has_value()) {
            prog.skip_capture(std::move(*p.capture));
            p = prog.advance_prefill(s);
        }
        return p;
    };

    ninfer::runtime::ResolvedExecutionOptions exec_options{};
    exec_options.requested_output_tokens = 8;
    ninfer::runtime::ContextMachineCostModel cost_model{};

    std::vector<ContinuationHandle> continuations;

    // Run 3 requests that finish as continuations
    for (int req = 0; req < 3; ++req) {
        std::vector<ninfer::TokenId> tokens(8);
        for (std::size_t i = 0; i < 8; ++i) {
            tokens[i] = static_cast<ninfer::TokenId>(req * 100 + i);
        }
        const auto prompt = make_prompt(tokens, true);
        auto base_plan    = program.plan_request(prompt, exec_options);
        auto candidate    = program.inspect_admission(
            prompt, base_plan, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
        failures += check(candidate.has_value(), "Admission must succeed");

        std::optional<ResourcePlan> res_plan;
        if (req < 2) {
            res_plan = program.seal_identity(*candidate, prompt);
        } else {
            // Request 2 plans eviction of LRU continuation 0 via pressure planning
            std::vector<const ContinuationHandle*> owners = {&continuations[0], &continuations[1]};
            std::vector<std::uint32_t> owner_ordinals = {0, 1};
            const AdmissionCandidate* cand_ptr = &*candidate;
            auto session = program.begin_pressure_planning(cost_model, std::span(&cand_ptr, 1), owners, owner_ordinals, {}, {});
            auto prep = session.prepare_expansion(session.identity_target(*candidate));
            auto view = session.commit_expansion(std::move(prep));
            failures += check(!view.children.empty(), "Expansion must provide children");
            res_plan = session.seal(view.children[0], prompt);
        }
        failures += check(res_plan.has_value(), "Resource plan must be created");

        std::atomic<bool> flag{false};
        ninfer::runtime::CancellationFlagView cancellation{&flag};
        (void)program.start_resource_transaction(std::move(*res_plan), make_prompt(tokens, true),
                                                 cancellation);
        auto progress      = program.progress_context_transaction(cancellation);
        auto* mat          = std::get_if<MaterializationResult>(&progress);
        SequenceHandle seq = mat->published->sequence;
        if (req == 2) {
            failures += check(mat->victims.size() == 1, "Request 2 must have 1 eviction victim");
            if (mat->victims.size() == 1) {
                failures += check(mat->victims[0].pressure_committed, "Victim must have pressure_committed = true");
            }
        }
        program.finalize_context_transaction();

        (void)advance_prefill_full(program, seq);
        FinishResult fin = program.finish(seq);
        failures += check(fin.disposition == ninfer::runtime::FinishDisposition::Catalogued,
                          "Request finish must be Catalogued");
        if (fin.continuation.has_value()) {
            continuations.push_back(std::move(*fin.continuation));
        }
    }

    // Since continuation_capacity is 2, slot count should be 2
    failures += check(program.physical_usage().device_state_slots == 2,
                      "Catalogued continuation slots must be capped at capacity 2");

    // The first continuation (index 0) was evicted by the 3rd request!
    // If we try to admit with continuation 0, it should fail verification (stale generation)
    std::vector<ninfer::TokenId> req0_tokens(8);
    for (std::size_t i = 0; i < 8; ++i) { req0_tokens[i] = static_cast<ninfer::TokenId>(i); }
    const auto prompt0 = make_prompt(req0_tokens, true);
    auto base0         = program.plan_request(prompt0, exec_options);
    auto cand0         = program.inspect_admission(
        prompt0, base0, ninfer::runtime::LaneId(0), &continuations[0], nullptr, std::nullopt, false, cost_model);
    failures += check(!cand0.has_value(),
                      "Evicted continuation must return nullopt on inspection");

    // Release remaining continuations
    for (auto& c : continuations) {
        (void)program.release_continuation(std::move(c));
    }
    failures += check(program.physical_usage().device_state_slots == 0,
                      "All continuation slots must be reclaimed");

    std::printf("[DONE] test_continuation_lru_eviction, failures: %d\n", failures);
    std::fflush(stdout);
    return failures;
}

int test_turn_closure_checkpoint_and_multi_turn_reuse(ninfer::DeviceContext& device) {
    std::printf("[RUN] test_turn_closure_checkpoint_and_multi_turn_reuse\n");
    std::fflush(stdout);
    int failures = 0;
    auto model   = make_synthetic_model(device);
    auto ple_meta = make_synthetic_ple_meta();

    FlashNextRuntimeConfig cfg{
        .max_concurrency       = 2,
        .max_context           = 512,
        .state_slot_capacity   = 0,
        .continuation_capacity = 4,
        .prefill_chunk         = 256,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    auto program_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, model.view, std::nullopt, ple_meta);
    Program program(std::move(program_impl));

    auto advance_prefill_full = [&](Program& prog, SequenceHandle s) -> PrefillProgress {
        auto p = prog.advance_prefill(s);
        if (p.capture.has_value()) {
            prog.skip_capture(std::move(*p.capture));
            p = prog.advance_prefill(s);
        }
        return p;
    };

    ninfer::runtime::ResolvedExecutionOptions exec_options{};
    exec_options.requested_output_tokens = 8;
    ninfer::runtime::ContextMachineCostModel cost_model{};
    std::atomic<bool> flag{false};
    ninfer::runtime::CancellationFlagView cancellation{&flag};

    // 0. Frontier Derivation tests from PreparedPromptData
    {
        std::vector<ninfer::TokenId> test_tokens(20);
        for (std::size_t i = 0; i < 20; ++i) { test_tokens[i] = static_cast<ninfer::TokenId>(100 + i); }

        // (a) rewrite_checkpoint provided (e.g. F = 16)
        const auto prompt_rewrite = make_prompt(test_tokens, true, 16);
        const auto& d_rewrite = ninfer::targets::qwen3_6::PreparedPromptAccess::view(prompt_rewrite);
        failures += check(d_rewrite.identity.rewrite_checkpoint.has_value() &&
                          d_rewrite.identity.rewrite_checkpoint->frontier == 16,
                          "Rewrite checkpoint frontier must be 16 in PreparedPromptData");

        // (b) context_cache opportunities provided (e.g. F = 8 and F = 16)
        std::vector<ninfer::targets::qwen3_6::PreparedCacheOpportunity> opps = {
            {.kind = ninfer::PromptCacheMarkerKind::SharedStablePrefix, .frontier = 8, .input_order = 0},
            {.kind = ninfer::PromptCacheMarkerKind::PrivateLongAnchor, .frontier = 16, .input_order = 1},
        };
        const auto prompt_opps = make_prompt_with_cache_opportunities(test_tokens, opps);
        const auto& d_opps = ninfer::targets::qwen3_6::PreparedPromptAccess::view(prompt_opps);
        failures += check(d_opps.context_cache.opportunities.size() == 2,
                          "Context cache opportunities must have 2 entries");

        // (c) No boundary
        const auto prompt_none = make_prompt(test_tokens, true, std::nullopt);
        const auto& d_none = ninfer::targets::qwen3_6::PreparedPromptAccess::view(prompt_none);
        failures += check(!d_none.identity.rewrite_checkpoint.has_value() &&
                          d_none.context_cache.opportunities.empty(),
                          "Prompt without boundary must have nullopt rewrite and empty opportunities");
    }

    // 1. Turn 1: 20 prompt tokens with F = 16 (16 prompt tokens before prologue, 4 prologue tokens)
    //    Step 1: prefill 0..16 -> complete=false, capture=offer, pending=nullopt, processed=16
    //    Step 2: prefill 16..20 -> complete=true, pending=PendingBatch, capture=nullopt, processed=4
    std::vector<ninfer::TokenId> turn1_tokens(20);
    for (std::size_t i = 0; i < 20; ++i) { turn1_tokens[i] = static_cast<ninfer::TokenId>(500 + i); }
    const auto prompt1 = make_prompt(turn1_tokens, true, 16);

    auto base1 = program.plan_request(prompt1, exec_options);
    auto cand1 = program.inspect_admission(
        prompt1, base1, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
    failures += check(cand1.has_value(), "Turn 1 admission must succeed");
    auto res1 = program.seal_identity(*cand1, prompt1);

    std::printf("  [Turn 1] Starting resource transaction...\n");
    (void)program.start_resource_transaction(std::move(*res1), make_prompt(turn1_tokens, true, 16), cancellation);
    std::printf("  [Turn 1] Progressing transaction...\n");
    auto prog1 = program.progress_context_transaction(cancellation);
    auto* mat1 = std::get_if<MaterializationResult>(&prog1);
    SequenceHandle seq1 = mat1->published->sequence;
    program.finalize_context_transaction();

    // Step 1: Run prompt chunk to F=16 -> complete=false, capture offered, pending empty, processed=16
    std::printf("  [Turn 1] Advancing prefill (Step 1: capture offer at F=16)...\n");
    auto prefill_step1 = program.advance_prefill(seq1);
    failures += check(!prefill_step1.complete, "Turn 1 prefill step 1 must NOT be complete");
    failures += check(prefill_step1.capture.has_value(), "Turn 1 step 1 must offer capture");
    failures += check(!prefill_step1.pending.has_value(), "Turn 1 step 1 pending must be empty");
    failures += check(prefill_step1.processed_prompt_tokens == 16,
                      "Turn 1 step 1 processed prompt tokens must be 16");

    // Inspect capture offer
    std::printf("  [Turn 1] Inspecting capture...\n");
    auto assess = program.inspect_capture(*prefill_step1.capture, nullptr, nullptr, std::nullopt);
    failures += check(assess.publishes_private, "Capture assessment must publish private turn closure");
    failures += check(assess.frontier == 16, "Capture assessment frontier must be 16");

    // Reserve capture transaction (captures TurnClosure checkpoint at F=16)
    std::printf("  [Turn 1] Reserving active capture...\n");
    auto reserve_stat = program.reserve_active_capture(
        std::move(*prefill_step1.capture), nullptr, nullptr, std::nullopt, cancellation);
    failures += check(reserve_stat == ninfer::runtime::ContextTransactionReserveStatus::Reserved,
                      "Capture reservation must return Reserved");

    // Progress capture transaction
    std::printf("  [Turn 1] Progressing capture transaction...\n");
    auto capture_prog = program.progress_context_transaction(cancellation);
    auto* cap_res = std::get_if<ActiveCaptureResult>(&capture_prog);
    failures += check(cap_res != nullptr, "Progress must return ActiveCaptureResult");
    failures += check(cap_res->status == ninfer::runtime::ContextTransactionStatus::Published,
                      "Capture transaction status must be Published");
    failures += check(cap_res->active_summary.rewrite.has_value(), "Capture result must populate rewrite summary");
    if (cap_res->active_summary.rewrite.has_value()) {
        failures += check(cap_res->active_summary.rewrite->ref.kind == ninfer::runtime::CheckpointKind::TurnClosure,
                          "Rewrite checkpoint kind must be TurnClosure");
        failures += check(cap_res->active_summary.rewrite->ref.frontier == 16,
                          "Rewrite checkpoint frontier must be 16");
    }
    program.finalize_context_transaction();

    // Step 2: Next advance_prefill completes remaining prologue tokens (16..20) and samples first output token
    std::printf("  [Turn 1] Advancing prefill (Step 2: prologue completion 16..20)...\n");
    auto prefill_step2 = program.advance_prefill(seq1);
    failures += check(prefill_step2.complete, "Turn 1 prefill step 2 must be complete");
    failures += check(prefill_step2.pending.has_value(), "Turn 1 step 2 must have pending batch");
    failures += check(!prefill_step2.capture.has_value(), "Turn 1 step 2 capture must be empty");
    failures += check(prefill_step2.processed_prompt_tokens == 4,
                      "Turn 1 step 2 processed prompt tokens must be 4");

    // Commit generated token for Turn 1
    std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = true}}};
    if (prefill_step2.pending.has_value()) {
        (void)program.commit(std::move(*prefill_step2.pending), commit_dec);
    }

    std::printf("  [Turn 1] Finishing sequence...\n");
    FinishResult fin1 = program.finish(seq1);
    failures += check(fin1.disposition == ninfer::runtime::FinishDisposition::Catalogued,
                      "Turn 1 finish must be Catalogued");
    failures += check(fin1.continuation.has_value(), "Turn 1 finish must return continuation");
    failures += check(fin1.summary.rewrite.has_value(), "Turn 1 finish summary must contain rewrite TurnClosure");
    if (fin1.summary.rewrite.has_value()) {
        failures += check(fin1.summary.rewrite->ref.frontier == 16, "Turn 1 finish rewrite frontier must be 16");
    }

    // 2. Turn 2: 28 prompt tokens starting with first 16 tokens of Turn 1, plus 12 new tokens (F2 = 24)
    std::vector<ninfer::TokenId> turn2_tokens(28);
    for (std::size_t i = 0; i < 16; ++i) { turn2_tokens[i] = turn1_tokens[i]; }
    for (std::size_t i = 16; i < 28; ++i) { turn2_tokens[i] = static_cast<ninfer::TokenId>(900 + i); }
    const auto prompt2 = make_prompt(turn2_tokens, true, 24);

    std::printf("  [Turn 2] Planning request...\n");
    auto base2 = program.plan_request(prompt2, exec_options);
    std::printf("  [Turn 2] Inspecting admission...\n");
    auto cand2 = program.inspect_admission(
        prompt2, base2, ninfer::runtime::LaneId(0), &*fin1.continuation, nullptr, std::nullopt, false, cost_model);
    failures += check(cand2.has_value(), "Turn 2 admission with TurnClosure must succeed");
    if (cand2.has_value()) {
        failures += check(cand2->summary().reusable_prompt_tokens == 16,
                          "Turn 2 reusable prompt tokens must be 16");
        failures += check(cand2->summary().prefix_reuse_path == PrefixReusePath::PrivateTurnClosure,
                          "Turn 2 prefix reuse path must be PrivateTurnClosure");
    }

    std::printf("  [Turn 2] Sealing identity and starting transaction...\n");
    auto res2 = program.seal_identity(*cand2, prompt2);
    (void)program.start_resource_transaction(std::move(*res2), make_prompt(turn2_tokens, true, 24), cancellation);
    std::printf("  [Turn 2] Progressing transaction...\n");
    auto prog2 = program.progress_context_transaction(cancellation);
    auto* mat2 = std::get_if<MaterializationResult>(&prog2);
    SequenceHandle seq2 = mat2->published->sequence;
    program.finalize_context_transaction();

    // Advance prefill for Turn 2: prefills from 16 to F2=24 (8 tokens)
    std::printf("  [Turn 2] Advancing prefill (delta tokens 16..24)...\n");
    auto prefill_prog2_step1 = program.advance_prefill(seq2);
    failures += check(!prefill_prog2_step1.complete, "Turn 2 step 1 must offer capture at F2=24");
    failures += check(prefill_prog2_step1.processed_prompt_tokens == 8,
                      "Turn 2 processed prompt tokens must be 8");
    failures += check(prefill_prog2_step1.summary.reused_prompt_tokens == 16,
                      "Turn 2 summary reused prompt tokens must be 16");
    failures += check(prefill_prog2_step1.summary.prefix_reuse_path == PrefixReusePath::PrivateTurnClosure,
                      "Turn 2 summary prefix reuse path must be PrivateTurnClosure");

    // Skip capture for Turn 2 and complete step 2 (prologue 24..28, 4 tokens)
    if (prefill_prog2_step1.capture.has_value()) {
        program.skip_capture(std::move(*prefill_prog2_step1.capture));
    }
    auto prefill_prog2 = program.advance_prefill(seq2);
    failures += check(prefill_prog2.complete, "Turn 2 step 2 must complete");
    failures += check(prefill_prog2.processed_prompt_tokens == 4,
                      "Turn 2 step 2 processed prompt tokens must be 4");
    failures += check(prefill_prog2.pending.has_value(), "Turn 2 prefill must yield pending round");

    std::vector<std::uint16_t> t2_logits;
    if (prefill_prog2.pending.has_value()) {
        const auto& logits_tensor = program.impl_->allocation_.round_tensors().logits;
        t2_logits.resize(logits_tensor.numel());
        CUDA_CHECK(cudaMemcpy(t2_logits.data(), logits_tensor.data,
                               t2_logits.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
        (void)program.commit(std::move(*prefill_prog2.pending), commit_dec);
    }

    std::printf("  [Turn 2] Finishing sequence...\n");
    FinishResult fin2 = program.finish(seq2);

    // Assert that Turn 1's TurnClosure groups remain allocated after Turn 2 finishes
    failures += check(program.physical_usage().device_main_kv_pages > 0,
                      "TurnClosure page groups must remain allocated (not in free list) after Turn 2 release");

    // 3. Unrelated request: Allocate and write free physical page groups (filling the pool)
    std::printf("  [Pool Fill] Running unrelated request across free pool...\n");
    std::vector<ninfer::TokenId> fill_tokens(256);
    for (std::size_t i = 0; i < 256; ++i) { fill_tokens[i] = static_cast<ninfer::TokenId>(3000 + i); }
    const auto prompt_fill = make_prompt(fill_tokens, true);
    auto base_fill = program.plan_request(prompt_fill, exec_options);
    auto cand_fill = program.inspect_admission(
        prompt_fill, base_fill, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
    failures += check(cand_fill.has_value(), "Pool fill admission must succeed");
    auto res_fill = program.seal_identity(*cand_fill, prompt_fill);
    (void)program.start_resource_transaction(std::move(*res_fill), make_prompt(fill_tokens, true), cancellation);
    auto prog_fill = program.progress_context_transaction(cancellation);
    auto* mat_fill = std::get_if<MaterializationResult>(&prog_fill);
    SequenceHandle seq_fill = mat_fill->published->sequence;
    program.finalize_context_transaction();

    auto prefill_fill = advance_prefill_full(program, seq_fill);
    failures += check(prefill_fill.complete, "Pool fill prefill must be complete");
    if (prefill_fill.pending.has_value()) {
        (void)program.commit(std::move(*prefill_fill.pending), commit_dec);
    }
    FinishResult fin_fill = program.finish(seq_fill);

    // 4. Turn 3: Resumes from the same Turn 1 TurnClosure checkpoint
    std::printf("  [Turn 3] Resuming from same TurnClosure after pool fill...\n");
    auto base3 = program.plan_request(prompt2, exec_options);
    auto cand3 = program.inspect_admission(
        prompt2, base3, ninfer::runtime::LaneId(0), &*fin1.continuation, nullptr, std::nullopt, false, cost_model);
    failures += check(cand3.has_value(), "Turn 3 admission with TurnClosure must succeed");
    if (cand3.has_value()) {
        failures += check(cand3->summary().reusable_prompt_tokens == 16,
                          "Turn 3 reusable prompt tokens must be 16");
        failures += check(cand3->summary().prefix_reuse_path == PrefixReusePath::PrivateTurnClosure,
                          "Turn 3 prefix reuse path must be PrivateTurnClosure");
    }

    auto res3 = program.seal_identity(*cand3, prompt2);
    (void)program.start_resource_transaction(std::move(*res3), make_prompt(turn2_tokens, true, 24), cancellation);
    auto prog3 = program.progress_context_transaction(cancellation);
    auto* mat3 = std::get_if<MaterializationResult>(&prog3);
    SequenceHandle seq3 = mat3->published->sequence;
    program.finalize_context_transaction();

    auto prefill_prog3_step1 = program.advance_prefill(seq3);
    failures += check(!prefill_prog3_step1.complete, "Turn 3 step 1 must offer capture");
    failures += check(prefill_prog3_step1.processed_prompt_tokens == 8,
                      "Turn 3 processed prompt tokens must be 8");
    failures += check(prefill_prog3_step1.summary.reused_prompt_tokens == 16,
                      "Turn 3 reused prompt tokens must be 16");

    if (prefill_prog3_step1.capture.has_value()) {
        program.skip_capture(std::move(*prefill_prog3_step1.capture));
    }
    auto prefill_prog3 = program.advance_prefill(seq3);
    failures += check(prefill_prog3.complete, "Turn 3 prefill must be complete");
    failures += check(prefill_prog3.pending.has_value(), "Turn 3 prefill must yield pending round");

    std::vector<std::uint16_t> t3_logits;
    if (prefill_prog3.pending.has_value()) {
        const auto& logits_tensor = program.impl_->allocation_.round_tensors().logits;
        t3_logits.resize(logits_tensor.numel());
        CUDA_CHECK(cudaMemcpy(t3_logits.data(), logits_tensor.data,
                               t3_logits.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
        (void)program.commit(std::move(*prefill_prog3.pending), commit_dec);
    }

    FinishResult fin3 = program.finish(seq3);

    // Bit-identical logits check between Turn 2 and Turn 3
    failures += check(!t2_logits.empty() && t2_logits.size() == t3_logits.size(),
                      "T2 and T3 logits must be non-empty and of matching size");
    bool bit_identical = (t2_logits == t3_logits);
    failures += check(bit_identical, "Turn 3 logits must be bit-identical to Turn 2 logits after pool fill");

    // 5. Test offer skipped with F < N
    std::printf("  [Skip Test] Testing skip_capture with F=12 < N=16...\n");
    {
        std::vector<ninfer::TokenId> skip_tokens(16);
        for (std::size_t i = 0; i < 16; ++i) { skip_tokens[i] = static_cast<ninfer::TokenId>(700 + i); }
        const auto prompt_skip = make_prompt(skip_tokens, true, 12);
        auto base_skip = program.plan_request(prompt_skip, exec_options);
        auto cand_skip = program.inspect_admission(
            prompt_skip, base_skip, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
        failures += check(cand_skip.has_value(), "Skip test admission must succeed");
        auto res_skip = program.seal_identity(*cand_skip, prompt_skip);
        (void)program.start_resource_transaction(std::move(*res_skip), make_prompt(skip_tokens, true, 12), cancellation);
        auto prog_skip = program.progress_context_transaction(cancellation);
        auto* mat_skip = std::get_if<MaterializationResult>(&prog_skip);
        SequenceHandle seq_skip = mat_skip->published->sequence;
        program.finalize_context_transaction();

        // Step 1: runs 0..12 -> returns capture offer
        auto p1 = program.advance_prefill(seq_skip);
        failures += check(!p1.complete, "Skip test step 1 must not be complete");
        failures += check(p1.capture.has_value(), "Skip test step 1 must offer capture");
        failures += check(p1.processed_prompt_tokens == 12, "Skip test step 1 processed must be 12");

        // Skip capture
        program.skip_capture(std::move(*p1.capture));

        // Step 2: runs 12..16 -> completes prompt
        auto p2 = program.advance_prefill(seq_skip);
        failures += check(p2.complete, "Skip test step 2 must complete");
        failures += check(p2.processed_prompt_tokens == 4, "Skip test step 2 processed must be 4");
        failures += check(p2.pending.has_value(), "Skip test step 2 must have pending batch");
        (void)program.abort(seq_skip);
    }

    // 6. Test prompt with no boundary (no capture offered)
    std::printf("  [No Boundary Test] Testing prompt with no boundary (no capture)...\n");
    {
        std::vector<ninfer::TokenId> nobound_tokens(16);
        for (std::size_t i = 0; i < 16; ++i) { nobound_tokens[i] = static_cast<ninfer::TokenId>(800 + i); }
        const auto prompt_nobound = make_prompt(nobound_tokens, true, std::nullopt);
        auto base_nobound = program.plan_request(prompt_nobound, exec_options);
        auto cand_nobound = program.inspect_admission(
            prompt_nobound, base_nobound, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
        failures += check(cand_nobound.has_value(), "No-boundary admission must succeed");
        auto res_nobound = program.seal_identity(*cand_nobound, prompt_nobound);
        (void)program.start_resource_transaction(std::move(*res_nobound), make_prompt(nobound_tokens, true, std::nullopt), cancellation);
        auto prog_nobound = program.progress_context_transaction(cancellation);
        auto* mat_nobound = std::get_if<MaterializationResult>(&prog_nobound);
        SequenceHandle seq_nobound = mat_nobound->published->sequence;
        program.finalize_context_transaction();

        // Advance prefill: completes in single step without capture
        auto p_nobound = program.advance_prefill(seq_nobound);
        failures += check(p_nobound.complete, "No-boundary prefill must complete in single step");
        failures += check(!p_nobound.capture.has_value(), "No-boundary prefill must NOT offer capture");
        failures += check(p_nobound.pending.has_value(), "No-boundary prefill must have pending batch");
        failures += check(p_nobound.processed_prompt_tokens == 16, "No-boundary processed tokens must be 16");
        (void)program.abort(seq_nobound);
    }

    // 7. Release continuations and verify complete reclamation to free list
    std::printf("  [Reclamation] Releasing continuations and verifying free list...\n");
    for (std::size_t c = 0; c < program.impl_->continuation_slots_.size(); ++c) {
        auto& c_slot = program.impl_->continuation_slots_[c];
        if (c_slot.role == detail::ContinuationSlotRole::Catalogued) {
            (void)program.release_continuation(
                ContinuationHandle(&program, static_cast<std::uint32_t>(c), c_slot.generation));
        }
    }

    const auto usage_after = program.physical_usage();
    failures += check(usage_after.device_state_slots == 0,
                      "All device state slots must be reclaimed (0)");
    failures += check(usage_after.device_main_kv_pages == 0,
                      "All physical page groups must be returned to free list (0 used pages)");

    std::printf("[DONE] test_turn_closure_checkpoint_and_multi_turn_reuse, failures: %d\n", failures);
    std::fflush(stdout);
    return failures;
}

int test_turn_closure_chain_retention(ninfer::DeviceContext& device) {
    std::printf("[TEST] test_turn_closure_chain_retention\n");
    int failures = 0;

    auto model    = make_synthetic_model(device);
    auto ple_meta = make_synthetic_ple_meta();

    FlashNextRuntimeConfig cfg{
        .max_concurrency       = 2,
        .max_context           = 512,
        .state_slot_capacity   = 0,
        .continuation_capacity = 16,
        .prefill_chunk         = 256,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    auto program_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, model.view, std::nullopt, ple_meta);
    Program program(std::move(program_impl));

    ninfer::runtime::ResolvedExecutionOptions exec_options{};
    exec_options.requested_output_tokens = 8;
    ninfer::runtime::ContextMachineCostModel cost_model{};
    std::atomic<bool> flag{false};
    ninfer::runtime::CancellationFlagView cancellation{&flag};

    std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = false}}};

    // 1. T1: 20 tokens, F1 = 12
    std::vector<ninfer::TokenId> t1_tokens(20);
    for (std::size_t i = 0; i < 20; ++i) { t1_tokens[i] = static_cast<ninfer::TokenId>(1000 + i); }
    const auto prompt1 = make_prompt(t1_tokens, true, 12);
    auto base1 = program.plan_request(prompt1, exec_options);
    auto cand1 = program.inspect_admission(prompt1, base1, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
    failures += check(cand1.has_value(), "T1 admission must succeed");
    auto res1 = program.seal_identity(*cand1, prompt1);
    (void)program.start_resource_transaction(std::move(*res1), make_prompt(t1_tokens, true, 12), cancellation);
    auto prog1 = program.progress_context_transaction(cancellation);
    SequenceHandle seq1 = std::get_if<MaterializationResult>(&prog1)->published->sequence;
    program.finalize_context_transaction();

    auto p1_step1 = program.advance_prefill(seq1);
    failures += check(p1_step1.capture.has_value(), "T1 step 1 must offer capture at 12");
    auto cap1 = program.reserve_active_capture(std::move(*p1_step1.capture), nullptr, nullptr, std::nullopt, cancellation);
    failures += check(cap1 == ninfer::runtime::ContextTransactionReserveStatus::Reserved, "T1 capture must reserve");
    (void)program.progress_context_transaction(cancellation);
    program.finalize_context_transaction();

    auto p1_step2 = program.advance_prefill(seq1);
    failures += check(p1_step2.complete, "T1 step 2 must complete");
    if (p1_step2.pending.has_value()) {
        (void)program.commit(std::move(*p1_step2.pending), commit_dec);
    }
    FinishResult fin1 = program.finish(seq1);
    failures += check(fin1.continuation.has_value(), "T1 finish must yield continuation");

    // 2. T2: 40 tokens (first 20 same as T1), F2 = 30
    std::vector<ninfer::TokenId> t2_tokens(40);
    for (std::size_t i = 0; i < 20; ++i) { t2_tokens[i] = t1_tokens[i]; }
    for (std::size_t i = 20; i < 40; ++i) { t2_tokens[i] = static_cast<ninfer::TokenId>(2000 + i); }
    const auto prompt2 = make_prompt(t2_tokens, true, 30);
    auto base2 = program.plan_request(prompt2, exec_options);
    auto cand2 = program.inspect_admission(prompt2, base2, ninfer::runtime::LaneId(0), &*fin1.continuation, nullptr, std::nullopt, false, cost_model);
    failures += check(cand2.has_value(), "T2 admission must succeed against T1 continuation");
    if (cand2.has_value()) {
        failures += check(cand2->summary().reusable_prompt_tokens == 12, "T2 reusable prompt tokens must be 12");
        failures += check(cand2->summary().prefix_reuse_path == PrefixReusePath::PrivateTurnClosure, "T2 reuse path must be PrivateTurnClosure");
        failures += check(cand2->identity_assessment().source_disposition == runtime::ClaimDisposition::Retained,
                          "T2 source disposition must be Retained for TurnClosure");
    }
    auto res2 = program.seal_identity(*cand2, prompt2);
    (void)program.start_resource_transaction(std::move(*res2), make_prompt(t2_tokens, true, 30), cancellation);
    auto prog2 = program.progress_context_transaction(cancellation);
    SequenceHandle seq2 = std::get_if<MaterializationResult>(&prog2)->published->sequence;
    program.finalize_context_transaction();

    auto p2_step1 = program.advance_prefill(seq2);
    failures += check(p2_step1.capture.has_value(), "T2 step 1 must offer capture at 30");
    auto cap2 = program.reserve_active_capture(std::move(*p2_step1.capture), nullptr, nullptr, std::nullopt, cancellation);
    failures += check(cap2 == ninfer::runtime::ContextTransactionReserveStatus::Reserved, "T2 capture must reserve");
    (void)program.progress_context_transaction(cancellation);
    program.finalize_context_transaction();

    auto p2_step2 = program.advance_prefill(seq2);
    failures += check(p2_step2.complete, "T2 step 2 must complete");
    if (p2_step2.pending.has_value()) {
        (void)program.commit(std::move(*p2_step2.pending), commit_dec);
    }
    FinishResult fin2 = program.finish(seq2);
    failures += check(fin2.continuation.has_value(), "T2 finish must yield continuation");

    // 3. T3: 60 tokens (first 40 same as T2), F3 = 50
    std::vector<ninfer::TokenId> t3_tokens(60);
    for (std::size_t i = 0; i < 40; ++i) { t3_tokens[i] = t2_tokens[i]; }
    for (std::size_t i = 40; i < 60; ++i) { t3_tokens[i] = static_cast<ninfer::TokenId>(3000 + i); }
    const auto prompt3 = make_prompt(t3_tokens, true, 50);
    auto base3 = program.plan_request(prompt3, exec_options);
    auto cand3 = program.inspect_admission(prompt3, base3, ninfer::runtime::LaneId(0), &*fin2.continuation, nullptr, std::nullopt, false, cost_model);
    failures += check(cand3.has_value(), "T3 admission must succeed against T2 continuation");
    if (cand3.has_value()) {
        failures += check(cand3->summary().reusable_prompt_tokens == 30, "T3 reusable prompt tokens must be 30");
        failures += check(cand3->summary().prefix_reuse_path == PrefixReusePath::PrivateTurnClosure, "T3 reuse path must be PrivateTurnClosure");
        failures += check(cand3->identity_assessment().source_disposition == runtime::ClaimDisposition::Retained,
                          "T3 source disposition must be Retained for TurnClosure");
    }
    auto res3 = program.seal_identity(*cand3, prompt3);
    (void)program.start_resource_transaction(std::move(*res3), make_prompt(t3_tokens, true, 50), cancellation);
    auto prog3 = program.progress_context_transaction(cancellation);
    SequenceHandle seq3 = std::get_if<MaterializationResult>(&prog3)->published->sequence;
    program.finalize_context_transaction();

    auto p3_step1 = program.advance_prefill(seq3);
    failures += check(p3_step1.capture.has_value(), "T3 step 1 must offer capture at 50");
    auto cap3 = program.reserve_active_capture(std::move(*p3_step1.capture), nullptr, nullptr, std::nullopt, cancellation);
    failures += check(cap3 == ninfer::runtime::ContextTransactionReserveStatus::Reserved, "T3 capture must reserve");
    (void)program.progress_context_transaction(cancellation);
    program.finalize_context_transaction();

    auto p3_step2 = program.advance_prefill(seq3);
    failures += check(p3_step2.complete, "T3 step 2 must complete");
    if (p3_step2.pending.has_value()) {
        (void)program.commit(std::move(*p3_step2.pending), commit_dec);
    }
    FinishResult fin3 = program.finish(seq3);
    failures += check(fin3.continuation.has_value(), "T3 finish must yield continuation");

    // 4. Repeating T2 request (or branching from T1): Prompt begins with T1's 20 tokens, but has different 2nd turn
    std::vector<ninfer::TokenId> t2_repeat_tokens(36);
    for (std::size_t i = 0; i < 20; ++i) { t2_repeat_tokens[i] = t1_tokens[i]; }
    for (std::size_t i = 20; i < 36; ++i) { t2_repeat_tokens[i] = static_cast<ninfer::TokenId>(4000 + i); }
    const auto prompt2_rep = make_prompt(t2_repeat_tokens, true, 28);
    auto base2_rep = program.plan_request(prompt2_rep, exec_options);

    // Test admission with source = T3 continuation (which is descendant of T2, which is descendant of T1)
    auto cand2_rep_from_t3 = program.inspect_admission(prompt2_rep, base2_rep, ninfer::runtime::LaneId(0), &*fin3.continuation, nullptr, std::nullopt, false, cost_model);
    failures += check(cand2_rep_from_t3.has_value(), "T2 repeated admission against T3 continuation must succeed");
    if (cand2_rep_from_t3.has_value()) {
        failures += check(cand2_rep_from_t3->summary().reusable_prompt_tokens == 12,
                          "T2 repeated reusable tokens must be 12 (from T1's TurnClosure)");
        failures += check(cand2_rep_from_t3->summary().prefix_reuse_path == PrefixReusePath::PrivateTurnClosure,
                          "T2 repeated reuse path must be PrivateTurnClosure");
    }

    // Test admission with source = T1 continuation
    auto cand2_rep_from_t1 = program.inspect_admission(prompt2_rep, base2_rep, ninfer::runtime::LaneId(0), &*fin1.continuation, nullptr, std::nullopt, false, cost_model);
    failures += check(cand2_rep_from_t1.has_value(), "T2 repeated admission against T1 continuation must succeed");
    if (cand2_rep_from_t1.has_value()) {
        failures += check(cand2_rep_from_t1->summary().reusable_prompt_tokens == 12,
                          "T2 repeated reusable tokens must be 12 (from T1's TurnClosure)");
        failures += check(cand2_rep_from_t1->summary().prefix_reuse_path == PrefixReusePath::PrivateTurnClosure,
                          "T2 repeated reuse path must be PrivateTurnClosure");
    }

    // Run T2 repeated to completion
    auto res2_rep = program.seal_identity(*cand2_rep_from_t1, prompt2_rep);
    (void)program.start_resource_transaction(std::move(*res2_rep), make_prompt(t2_repeat_tokens, true, 28), cancellation);
    auto prog2_rep = program.progress_context_transaction(cancellation);
    SequenceHandle seq2_rep = std::get_if<MaterializationResult>(&prog2_rep)->published->sequence;
    program.finalize_context_transaction();

    auto p2_rep_step1 = program.advance_prefill(seq2_rep);
    failures += check(p2_rep_step1.summary.reused_prompt_tokens == 12, "T2 repeated step 1 reused tokens must be 12");
    failures += check(p2_rep_step1.summary.prefix_reuse_path == PrefixReusePath::PrivateTurnClosure,
                      "T2 repeated step 1 prefix reuse path must be PrivateTurnClosure");
    if (p2_rep_step1.capture.has_value()) {
        program.skip_capture(std::move(*p2_rep_step1.capture));
    }
    auto p2_rep_step2 = program.advance_prefill(seq2_rep);
    failures += check(p2_rep_step2.complete, "T2 repeated step 2 must complete");
    if (p2_rep_step2.pending.has_value()) {
        (void)program.commit(std::move(*p2_rep_step2.pending), commit_dec);
    }
    FinishResult fin2_rep = program.finish(seq2_rep);

    // Release all continuations
    for (std::size_t c = 0; c < program.impl_->continuation_slots_.size(); ++c) {
        auto& c_slot = program.impl_->continuation_slots_[c];
        if (c_slot.role == detail::ContinuationSlotRole::Catalogued) {
            (void)program.release_continuation(
                ContinuationHandle(&program, static_cast<std::uint32_t>(c), c_slot.generation));
        }
    }

    const auto usage_after = program.physical_usage();
    failures += check(usage_after.device_state_slots == 0,
                      "All device state slots must be reclaimed (0)");
    failures += check(usage_after.device_main_kv_pages == 0,
                      "All physical page groups must be returned to free list (0 used pages)");

    std::printf("[DONE] test_turn_closure_chain_retention, failures: %d\n", failures);
    std::fflush(stdout);
    return failures;
}

int test_turn2_resumed_vs_scratch_divergence(ninfer::DeviceContext& device) {
    std::printf("[TEST] test_turn2_resumed_vs_scratch_divergence\n");
    int failures = 0;

    auto model    = make_synthetic_model(device);
    auto ple_meta = make_synthetic_ple_meta();

    FlashNextRuntimeConfig cfg{
        .max_concurrency       = 2,
        .max_context           = 512,
        .state_slot_capacity   = 0,
        .continuation_capacity = 16,
        .prefill_chunk         = 256,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    auto program_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, model.view, std::nullopt, ple_meta);
    Program program(std::move(program_impl));

    ninfer::runtime::ResolvedExecutionOptions exec_options{};
    exec_options.requested_output_tokens = 8;
    ninfer::runtime::ContextMachineCostModel cost_model{};
    std::atomic<bool> flag{false};
    ninfer::runtime::CancellationFlagView cancellation{&flag};

    std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = false}}};

    // 1. Turn 1: 24 tokens, F = 16
    std::vector<ninfer::TokenId> t1_tokens(24);
    for (std::size_t i = 0; i < 24; ++i) { t1_tokens[i] = static_cast<ninfer::TokenId>(500 + i); }
    const auto prompt1 = make_prompt(t1_tokens, true, 16);
    auto base1 = program.plan_request(prompt1, exec_options);
    auto cand1 = program.inspect_admission(prompt1, base1, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
    failures += check(cand1.has_value(), "T1 admission must succeed");
    auto res1 = program.seal_identity(*cand1, prompt1);
    (void)program.start_resource_transaction(std::move(*res1), make_prompt(t1_tokens, true, 16), cancellation);
    auto prog1 = program.progress_context_transaction(cancellation);
    SequenceHandle seq1 = std::get_if<MaterializationResult>(&prog1)->published->sequence;
    program.finalize_context_transaction();

    auto p1_step1 = program.advance_prefill(seq1);
    failures += check(p1_step1.capture.has_value(), "T1 step 1 must offer capture at 16");
    auto cap1 = program.reserve_active_capture(std::move(*p1_step1.capture), nullptr, nullptr, std::nullopt, cancellation);
    failures += check(cap1 == ninfer::runtime::ContextTransactionReserveStatus::Reserved, "T1 capture must reserve");
    (void)program.progress_context_transaction(cancellation);
    program.finalize_context_transaction();

    auto p1_step2 = program.advance_prefill(seq1);
    failures += check(p1_step2.complete, "T1 step 2 must complete");
    if (p1_step2.pending.has_value()) {
        (void)program.commit(std::move(*p1_step2.pending), commit_dec);
    }
    FinishResult fin1 = program.finish(seq1);

    // 2. Turn 2 Resumed: 48 tokens (first 24 same as T1), resumes with cache = 16
    std::vector<ninfer::TokenId> t2_tokens(48);
    for (std::size_t i = 0; i < 24; ++i) { t2_tokens[i] = t1_tokens[i]; }
    for (std::size_t i = 24; i < 48; ++i) { t2_tokens[i] = static_cast<ninfer::TokenId>(600 + i); }
    const auto prompt2_res = make_prompt(t2_tokens, true, 36);
    auto base2_res = program.plan_request(prompt2_res, exec_options);
    auto cand2_res = program.inspect_admission(prompt2_res, base2_res, ninfer::runtime::LaneId(0), &*fin1.continuation, nullptr, std::nullopt, false, cost_model);
    failures += check(cand2_res.has_value(), "T2 resumed admission must succeed");
    failures += check(cand2_res->summary().reusable_prompt_tokens == 16, "T2 resumed reusable tokens must be 16");
    auto res2_res = program.seal_identity(*cand2_res, prompt2_res);
    (void)program.start_resource_transaction(std::move(*res2_res), make_prompt(t2_tokens, true, 36), cancellation);
    auto prog2_res = program.progress_context_transaction(cancellation);
    SequenceHandle seq2_res = std::get_if<MaterializationResult>(&prog2_res)->published->sequence;
    program.finalize_context_transaction();

    auto p2_res_step1 = program.advance_prefill(seq2_res);
    if (p2_res_step1.capture.has_value()) {
        program.skip_capture(std::move(*p2_res_step1.capture));
    }
    auto p2_res_step2 = program.advance_prefill(seq2_res);
    failures += check(p2_res_step2.complete, "T2 resumed prefill must complete");
    failures += check(p2_res_step2.pending.has_value(), "T2 resumed must have pending batch with logits");

    std::vector<float> logits_resumed;
    if (p2_res_step2.pending.has_value()) {
        const auto& logits_tensor = program.impl_->allocation_.round_tensors().logits;
        std::vector<std::uint16_t> bf16_h(logits_tensor.numel());
        CUDA_CHECK(cudaMemcpy(bf16_h.data(), logits_tensor.data,
                              bf16_h.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
        logits_resumed.resize(bf16_h.size());
        for (std::size_t i = 0; i < bf16_h.size(); ++i) {
            std::uint32_t u = static_cast<std::uint32_t>(bf16_h[i]) << 16;
            float f;
            std::memcpy(&f, &u, sizeof(float));
            logits_resumed[i] = f;
        }
        (void)program.commit(std::move(*p2_res_step2.pending), commit_dec);
    }
    (void)program.finish(seq2_res);

    // 3. Turn 2 From Scratch: exact same 48 tokens
    const auto prompt2_scr = make_prompt(t2_tokens, false, std::nullopt);
    auto base2_scr = program.plan_request(prompt2_scr, exec_options);
    auto cand2_scr = program.inspect_admission(prompt2_scr, base2_scr, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
    failures += check(cand2_scr.has_value(), "T2 scratch admission must succeed");
    auto res2_scr = program.seal_identity(*cand2_scr, prompt2_scr);
    (void)program.start_resource_transaction(std::move(*res2_scr), make_prompt(t2_tokens, false, std::nullopt), cancellation);
    auto prog2_scr = program.progress_context_transaction(cancellation);
    SequenceHandle seq2_scr = std::get_if<MaterializationResult>(&prog2_scr)->published->sequence;
    program.finalize_context_transaction();

    auto p2_scr = program.advance_prefill(seq2_scr);
    failures += check(p2_scr.complete, "T2 scratch prefill must complete");
    failures += check(p2_scr.pending.has_value(), "T2 scratch must have pending batch with logits");

    std::vector<float> logits_scratch;
    if (p2_scr.pending.has_value()) {
        const auto& logits_tensor = program.impl_->allocation_.round_tensors().logits;
        std::vector<std::uint16_t> bf16_h(logits_tensor.numel());
        CUDA_CHECK(cudaMemcpy(bf16_h.data(), logits_tensor.data,
                              bf16_h.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
        logits_scratch.resize(bf16_h.size());
        for (std::size_t i = 0; i < bf16_h.size(); ++i) {
            std::uint32_t u = static_cast<std::uint32_t>(bf16_h[i]) << 16;
            float f;
            std::memcpy(&f, &u, sizeof(float));
            logits_scratch[i] = f;
        }
        (void)program.commit(std::move(*p2_scr.pending), commit_dec);
    }
    (void)program.finish(seq2_scr);

    // 4. Numerical Divergence Check (rel-L2 <= 1e-2, argmax match)
    failures += check(!logits_resumed.empty() && logits_resumed.size() == logits_scratch.size(),
                      "Resumed and scratch logits must be non-empty and matching size");

    double diff_sq = 0.0;
    double ref_sq  = 0.0;
    for (std::size_t i = 0; i < logits_resumed.size(); ++i) {
        double d = static_cast<double>(logits_resumed[i]) - static_cast<double>(logits_scratch[i]);
        double r = static_cast<double>(logits_scratch[i]);
        diff_sq += d * d;
        ref_sq  += r * r;
    }
    const double rel_l2 = ref_sq > 0.0 ? std::sqrt(diff_sq / ref_sq) : 0.0;

    const auto max_it_res = std::max_element(logits_resumed.begin(), logits_resumed.end());
    const std::int32_t argmax_res = static_cast<std::int32_t>(std::distance(logits_resumed.begin(), max_it_res));

    const auto max_it_scr = std::max_element(logits_scratch.begin(), logits_scratch.end());
    const std::int32_t argmax_scr = static_cast<std::int32_t>(std::distance(logits_scratch.begin(), max_it_scr));

    std::printf("  [Divergence] rel-L2: %.6e (bound: 1.000000e-02), argmax_res: %d, argmax_scr: %d\n",
                rel_l2, argmax_res, argmax_scr);

    failures += check(rel_l2 <= 1e-2, "Turn 2 resumed vs from-scratch rel-L2 must be <= 1e-2 on synthetic model");
    failures += check(argmax_res == argmax_scr, "Turn 2 resumed vs from-scratch argmax must match");

    // Clean up
    for (std::size_t c = 0; c < program.impl_->continuation_slots_.size(); ++c) {
        auto& c_slot = program.impl_->continuation_slots_[c];
        if (c_slot.role == detail::ContinuationSlotRole::Catalogued) {
            (void)program.release_continuation(
                ContinuationHandle(&program, static_cast<std::uint32_t>(c), c_slot.generation));
        }
    }

    std::printf("[DONE] test_turn2_resumed_vs_scratch_divergence, failures: %d\n", failures);
    std::fflush(stdout);
    return failures;
}

int test_state_slot_capacity_saturation_and_eviction(ninfer::DeviceContext& device) {
    std::printf("[TEST] test_state_slot_capacity_saturation_and_eviction\n");
    int failures = 0;

    auto model    = make_synthetic_model(device);
    auto ple_meta = make_synthetic_ple_meta();

    // 2 lanes (active+standby = 4 slots: 0..3) + 4 continuation slots (cache slots 4..7) = 8 total state slots
    FlashNextRuntimeConfig cfg{
        .max_concurrency       = 2,
        .max_context           = 512,
        .state_slot_capacity   = 8,
        .continuation_capacity = 4,
        .prefill_chunk         = 256,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    failures += check(plan.state_slots == 8, "Plan state_slots must be exactly 8");

    auto program_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, model.view, std::nullopt, ple_meta);
    Program program(std::move(program_impl));

    ninfer::runtime::ResolvedExecutionOptions exec_options{};
    exec_options.requested_output_tokens = 8;
    ninfer::runtime::ContextMachineCostModel cost_model{};
    std::atomic<bool> flag{false};
    ninfer::runtime::CancellationFlagView cancellation{&flag};

    std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = false}}};

    // 1. Request 1 on Lane 0: 20 tokens, F1 = 12 -> creates TurnClosure (slot 0, cache_slot 4) & SessionEndpoint (slot 1, cache_slot 5)
    std::vector<ninfer::TokenId> r1_tokens(20);
    for (std::size_t i = 0; i < 20; ++i) { r1_tokens[i] = static_cast<ninfer::TokenId>(100 + i); }
    const auto prompt1 = make_prompt(r1_tokens, true, 12);
    auto base1 = program.plan_request(prompt1, exec_options);
    auto cand1 = program.inspect_admission(prompt1, base1, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
    failures += check(cand1.has_value(), "R1 admission must succeed");
    auto res1 = program.seal_identity(*cand1, prompt1);
    (void)program.start_resource_transaction(std::move(*res1), make_prompt(r1_tokens, true, 12), cancellation);
    auto prog1 = program.progress_context_transaction(cancellation);
    SequenceHandle seq1 = std::get_if<MaterializationResult>(&prog1)->published->sequence;
    program.finalize_context_transaction();

    auto p1_step1 = program.advance_prefill(seq1);
    failures += check(p1_step1.capture.has_value(), "R1 step 1 must offer capture at 12");
    auto cap1 = program.reserve_active_capture(std::move(*p1_step1.capture), nullptr, nullptr, std::nullopt, cancellation);
    failures += check(cap1 == ninfer::runtime::ContextTransactionReserveStatus::Reserved, "R1 capture must reserve");
    (void)program.progress_context_transaction(cancellation);
    program.finalize_context_transaction();

    auto p1_step2 = program.advance_prefill(seq1);
    failures += check(p1_step2.complete, "R1 step 2 must complete");
    if (p1_step2.pending.has_value()) {
        (void)program.commit(std::move(*p1_step2.pending), commit_dec);
    }
    FinishResult fin1 = program.finish(seq1);
    failures += check(fin1.continuation.has_value(), "R1 finish must yield continuation");

    // 2. Request 2 on Lane 1: 24 tokens, F2 = 16 -> creates TurnClosure (slot 2, cache_slot 6) & SessionEndpoint (slot 3, cache_slot 7)
    std::vector<ninfer::TokenId> r2_tokens(24);
    for (std::size_t i = 0; i < 24; ++i) { r2_tokens[i] = static_cast<ninfer::TokenId>(200 + i); }
    const auto prompt2 = make_prompt(r2_tokens, true, 16);
    auto base2 = program.plan_request(prompt2, exec_options);
    auto cand2 = program.inspect_admission(prompt2, base2, ninfer::runtime::LaneId(1), nullptr, nullptr, std::nullopt, false, cost_model);
    failures += check(cand2.has_value(), "R2 admission must succeed on Lane 1");
    auto res2 = program.seal_identity(*cand2, prompt2);
    (void)program.start_resource_transaction(std::move(*res2), make_prompt(r2_tokens, true, 16), cancellation);
    auto prog2 = program.progress_context_transaction(cancellation);
    SequenceHandle seq2 = std::get_if<MaterializationResult>(&prog2)->published->sequence;
    program.finalize_context_transaction();

    auto p2_step1 = program.advance_prefill(seq2);
    failures += check(p2_step1.capture.has_value(), "R2 step 1 must offer capture at 16");
    auto cap2 = program.reserve_active_capture(std::move(*p2_step1.capture), nullptr, nullptr, std::nullopt, cancellation);
    failures += check(cap2 == ninfer::runtime::ContextTransactionReserveStatus::Reserved, "R2 capture must reserve");
    (void)program.progress_context_transaction(cancellation);
    program.finalize_context_transaction();

    auto p2_step2 = program.advance_prefill(seq2);
    failures += check(p2_step2.complete, "R2 step 2 must complete");
    if (p2_step2.pending.has_value()) {
        (void)program.commit(std::move(*p2_step2.pending), commit_dec);
    }
    FinishResult fin2 = program.finish(seq2);
    failures += check(fin2.continuation.has_value(), "R2 finish must yield continuation");

    // Verify all 4 continuation slots are catalogued with cache_slots in [4, 7]
    for (std::size_t c = 0; c < 4; ++c) {
        const auto& c_slot = program.impl_->continuation_slots_[c];
        failures += check(c_slot.role == detail::ContinuationSlotRole::Catalogued,
                          "Continuation slot must be catalogued");
        failures += check(c_slot.cache_slot >= 4 && c_slot.cache_slot < 8,
                          "Cache slot index must be within [4, 7]");
    }

    // 3. Request 3 on Lane 0: 30 tokens, F3 = 20 -> drives pressure planning to evict LRU slots 0 and 1
    std::vector<ninfer::TokenId> r3_tokens(30);
    for (std::size_t i = 0; i < 30; ++i) { r3_tokens[i] = static_cast<ninfer::TokenId>(300 + i); }
    const auto prompt3 = make_prompt(r3_tokens, true, 20);
    auto base3 = program.plan_request(prompt3, exec_options);
    auto cand3 = program.inspect_admission(prompt3, base3, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
    failures += check(cand3.has_value(), "R3 admission must succeed");

    std::vector<ContinuationHandle> current_handles_r3;
    current_handles_r3.reserve(4);
    std::vector<std::uint32_t> ordinals_r3;
    for (std::size_t c = 0; c < program.impl_->continuation_slots_.size(); ++c) {
        const auto& sl = program.impl_->continuation_slots_[c];
        if (sl.role == detail::ContinuationSlotRole::Catalogued && c < 2) {
            current_handles_r3.push_back(ContinuationHandle(&program, static_cast<std::uint32_t>(c), sl.generation));
            ordinals_r3.push_back(static_cast<std::uint32_t>(c));
        }
    }
    std::vector<const ContinuationHandle*> owners_r3;
    for (const auto& h : current_handles_r3) { owners_r3.push_back(&h); }

    std::optional<ResourcePlan> res3;
    {
        const AdmissionCandidate* cand3_ptr = &*cand3;
        auto session_r3 = program.begin_pressure_planning(cost_model, std::span(&cand3_ptr, 1), owners_r3, ordinals_r3, {}, {});
        auto root_max3 = session_r3.root_maximal_target(*cand3);
        res3 = session_r3.seal(root_max3, prompt3);
    }
    failures += check(res3.has_value(), "R3 pressure seal must succeed");
    (void)program.start_resource_transaction(std::move(*res3), make_prompt(r3_tokens, true, 20), cancellation);
    auto prog3 = program.progress_context_transaction(cancellation);
    auto* mat3 = std::get_if<MaterializationResult>(&prog3);
    failures += check(mat3 != nullptr && mat3->victims.size() == 2, "R3 must have 2 pressure eviction victims");
    SequenceHandle seq3 = mat3->published->sequence;
    program.finalize_context_transaction();

    auto p3_step1 = program.advance_prefill(seq3);
    failures += check(p3_step1.capture.has_value(), "R3 step 1 must offer capture at 20");
    auto cap3 = program.reserve_active_capture(std::move(*p3_step1.capture), nullptr, nullptr, std::nullopt, cancellation);
    failures += check(cap3 == ninfer::runtime::ContextTransactionReserveStatus::Reserved, "R3 capture must reserve");
    (void)program.progress_context_transaction(cancellation);
    program.finalize_context_transaction();

    auto p3_step2 = program.advance_prefill(seq3);
    failures += check(p3_step2.complete, "R3 step 2 must complete");
    if (p3_step2.pending.has_value()) {
        (void)program.commit(std::move(*p3_step2.pending), commit_dec);
    }
    FinishResult fin3 = program.finish(seq3);
    failures += check(fin3.continuation.has_value(), "R3 finish must yield continuation");

    // 4. Request 4 on Lane 1: 40 tokens (resumed from R3's TurnClosure) -> evicts remaining old slots
    std::vector<ninfer::TokenId> r4_tokens(40);
    for (std::size_t i = 0; i < 20; ++i) { r4_tokens[i] = r3_tokens[i]; }
    for (std::size_t i = 20; i < 40; ++i) { r4_tokens[i] = static_cast<ninfer::TokenId>(400 + i); }
    const auto prompt4 = make_prompt(r4_tokens, true, 30);
    auto base4 = program.plan_request(prompt4, exec_options);
    auto cand4 = program.inspect_admission(prompt4, base4, ninfer::runtime::LaneId(1), &*fin3.continuation, nullptr, std::nullopt, false, cost_model);
    failures += check(cand4.has_value(), "R4 admission must succeed");

    std::vector<ContinuationHandle> current_handles_r4;
    current_handles_r4.reserve(4);
    std::vector<std::uint32_t> ordinals_r4;
    for (std::size_t c = 0; c < program.impl_->continuation_slots_.size(); ++c) {
        const auto& sl = program.impl_->continuation_slots_[c];
        if (sl.role == detail::ContinuationSlotRole::Catalogued && c >= 2) {
            current_handles_r4.push_back(ContinuationHandle(&program, static_cast<std::uint32_t>(c), sl.generation));
            ordinals_r4.push_back(static_cast<std::uint32_t>(c));
        }
    }
    std::vector<const ContinuationHandle*> owners_r4;
    for (const auto& h : current_handles_r4) { owners_r4.push_back(&h); }

    std::optional<ResourcePlan> res4;
    if (!owners_r4.empty()) {
        const AdmissionCandidate* cand4_ptr = &*cand4;
        auto session_r4 = program.begin_pressure_planning(cost_model, std::span(&cand4_ptr, 1), owners_r4, ordinals_r4, {}, {});
        auto root_max4 = session_r4.root_maximal_target(*cand4);
        res4 = session_r4.seal(root_max4, prompt4);
    } else {
        res4 = program.seal_identity(*cand4, prompt4);
    }
    failures += check(res4.has_value(), "R4 seal must succeed");
    (void)program.start_resource_transaction(std::move(*res4), make_prompt(r4_tokens, true, 30), cancellation);
    auto prog4 = program.progress_context_transaction(cancellation);
    SequenceHandle seq4 = std::get_if<MaterializationResult>(&prog4)->published->sequence;
    program.finalize_context_transaction();

    auto p4_step1 = program.advance_prefill(seq4);
    failures += check(p4_step1.capture.has_value(), "R4 step 1 must offer capture at 30");
    auto cap4 = program.reserve_active_capture(std::move(*p4_step1.capture), nullptr, nullptr, std::nullopt, cancellation);
    failures += check(cap4 == ninfer::runtime::ContextTransactionReserveStatus::Reserved, "R4 capture must reserve");
    (void)program.progress_context_transaction(cancellation);
    program.finalize_context_transaction();

    auto p4_step2 = program.advance_prefill(seq4);
    failures += check(p4_step2.complete, "R4 step 2 must complete");
    if (p4_step2.pending.has_value()) {
        (void)program.commit(std::move(*p4_step2.pending), commit_dec);
    }
    FinishResult fin4 = program.finish(seq4);
    failures += check(fin4.continuation.has_value(), "R4 finish must yield continuation");

    // Release all continuations
    for (std::size_t c = 0; c < program.impl_->continuation_slots_.size(); ++c) {
        auto& c_slot = program.impl_->continuation_slots_[c];
        if (c_slot.role == detail::ContinuationSlotRole::Catalogued) {
            (void)program.release_continuation(
                ContinuationHandle(&program, static_cast<std::uint32_t>(c), c_slot.generation));
        }
    }

    const auto usage_after = program.physical_usage();
    failures += check(usage_after.device_state_slots == 0,
                      "All device state slots must be reclaimed (0)");
    failures += check(usage_after.device_main_kv_pages == 0,
                      "All physical page groups must be returned to free list (0 used pages)");

    std::printf("[DONE] test_state_slot_capacity_saturation_and_eviction, failures: %d\n", failures);
    std::fflush(stdout);
    return failures;
}

int test_small_pool_page_pressure_eviction(ninfer::DeviceContext& device) {
    std::printf("[RUN] test_small_pool_page_pressure_eviction\n");
    std::fflush(stdout);
    int failures = 0;
    auto model   = make_synthetic_model(device);
    auto ple_meta = make_synthetic_ple_meta();

    // Pool has 4 page groups total. Max concurrency 2, max context 512 (2 groups per lane).
    FlashNextRuntimeConfig cfg{
        .max_concurrency       = 2,
        .max_context           = 512,
        .state_slot_capacity   = 0,
        .continuation_capacity = 4,
        .prefill_chunk         = 256,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    auto program_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, model.view, std::nullopt, ple_meta);
    Program program(std::move(program_impl));

    ninfer::runtime::ResolvedExecutionOptions exec_options{};
    exec_options.allow_prefix_reuse     = true;
    exec_options.requested_output_tokens = 8;
    ninfer::runtime::ContextMachineCostModel cost_model{};
    std::atomic<bool> flag{false};
    ninfer::runtime::CancellationFlagView cancellation{&flag};
    std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = true}}};

    // 1. Fill catalog with checkpoints owning all 4 page groups
    // Checkpoint A: 300 tokens -> 2 page groups
    std::vector<ninfer::TokenId> tokens_a(300);
    for (std::size_t i = 0; i < 300; ++i) { tokens_a[i] = static_cast<ninfer::TokenId>(1000 + i); }
    const auto prompt_a = make_prompt(tokens_a, true);
    auto base_a = program.plan_request(prompt_a, exec_options);
    auto cand_a = program.inspect_admission(prompt_a, base_a, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
    failures += check(cand_a.has_value(), "Admission A must succeed");
    auto res_a = program.seal_identity(*cand_a, prompt_a);
    (void)program.start_resource_transaction(std::move(*res_a), make_prompt(tokens_a, true), cancellation);
    auto prog_a = program.progress_context_transaction(cancellation);
    SequenceHandle seq_a = std::get_if<MaterializationResult>(&prog_a)->published->sequence;
    program.finalize_context_transaction();
    auto pa_step1 = program.advance_prefill(seq_a); // 0..256
    auto pa_step2 = program.advance_prefill(seq_a); // 256..300 (samples 1st token)
    if (pa_step2.pending.has_value()) { (void)program.commit(std::move(*pa_step2.pending), commit_dec); }
    FinishResult fin_a = program.finish(seq_a);
    failures += check(fin_a.disposition == ninfer::runtime::FinishDisposition::Catalogued, "Finish A must catalogue");

    // Checkpoint B: 300 tokens -> 2 page groups
    std::vector<ninfer::TokenId> tokens_b(300);
    for (std::size_t i = 0; i < 300; ++i) { tokens_b[i] = static_cast<ninfer::TokenId>(2000 + i); }
    const auto prompt_b = make_prompt(tokens_b, true);
    auto base_b = program.plan_request(prompt_b, exec_options);
    auto cand_b = program.inspect_admission(prompt_b, base_b, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
    failures += check(cand_b.has_value(), "Admission B must succeed");
    auto res_b = program.seal_identity(*cand_b, prompt_b);
    (void)program.start_resource_transaction(std::move(*res_b), make_prompt(tokens_b, true), cancellation);
    auto prog_b = program.progress_context_transaction(cancellation);
    SequenceHandle seq_b = std::get_if<MaterializationResult>(&prog_b)->published->sequence;
    program.finalize_context_transaction();
    auto pb_step1 = program.advance_prefill(seq_b); // 0..256
    auto pb_step2 = program.advance_prefill(seq_b); // 256..300
    if (pb_step2.pending.has_value()) { (void)program.commit(std::move(*pb_step2.pending), commit_dec); }
    FinishResult fin_b = program.finish(seq_b);
    failures += check(fin_b.disposition == ninfer::runtime::FinishDisposition::Catalogued, "Finish B must catalogue");

    // Check that available_physical_groups() is 0 (all 4 groups held in catalog)
    failures += check(program.impl_->executor_.available_physical_groups() == 0,
                      "Available physical groups must be 0 before pressure test");

    // 2. Concurrently admit, prefill, and decode 2 lanes to 300 tokens each (requires all 4 groups)
    std::vector<ninfer::TokenId> tokens_1(300);
    for (std::size_t i = 0; i < 300; ++i) { tokens_1[i] = static_cast<ninfer::TokenId>(3000 + i); }
    const auto prompt_1 = make_prompt(tokens_1, true);
    auto base_1 = program.plan_request(prompt_1, exec_options);
    auto cand_1 = program.inspect_admission(prompt_1, base_1, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
    failures += check(cand_1.has_value(), "Admission 1 must succeed under page pressure");
    if (cand_1.has_value()) {
        // Identity semantics: no evictions. With every group held by the catalog the identity
        // target is Infeasible; the pressure planner (root-maximal below) makes it feasible.
        failures += check(cand_1->identity_assessment().physical_status == ninfer::runtime::MaterializationPhysicalStatus::Infeasible,
                          "Admission 1 identity must be Infeasible under page pressure");
    }

    std::vector<ninfer::TokenId> tokens_2(300);
    for (std::size_t i = 0; i < 300; ++i) { tokens_2[i] = static_cast<ninfer::TokenId>(4000 + i); }
    const auto prompt_2 = make_prompt(tokens_2, true);
    auto base_2 = program.plan_request(prompt_2, exec_options);

    std::vector<ContinuationHandle> current_handles;
    current_handles.reserve(2);
    if (fin_a.continuation) current_handles.push_back(std::move(*fin_a.continuation));
    if (fin_b.continuation) current_handles.push_back(std::move(*fin_b.continuation));
    std::vector<const ContinuationHandle*> owners;
    std::vector<std::uint32_t> ordinals;
    for (std::size_t i = 0; i < current_handles.size(); ++i) {
        owners.push_back(&current_handles[i]);
        ordinals.push_back(static_cast<std::uint32_t>(i));
    }

    std::optional<ResourcePlan> res_1;
    {
        const AdmissionCandidate* cand_1_ptr = &*cand_1;
        auto session_1 = program.begin_pressure_planning(cost_model, std::span(&cand_1_ptr, 1), owners, ordinals, {}, {});
        auto root_max1 = session_1.root_maximal_target(*cand_1);
        res_1 = session_1.seal(root_max1, prompt_1);
    }
    failures += check(res_1.has_value(), "Admission 1 pressure seal must succeed");
    (void)program.start_resource_transaction(std::move(*res_1), make_prompt(tokens_1, true), cancellation);
    auto prog_1 = program.progress_context_transaction(cancellation);
    auto* mat_1 = std::get_if<MaterializationResult>(&prog_1);
    failures += check(mat_1 != nullptr && mat_1->victims.size() == 2, "Admission 1 must evict Checkpoint A and B");
    SequenceHandle seq_1 = mat_1->published->sequence;
    program.finalize_context_transaction();

    // Now with Checkpoints A and B evicted and 4 page groups free, admit Request 2
    auto cand_2 = program.inspect_admission(prompt_2, base_2, ninfer::runtime::LaneId(1), nullptr, nullptr, std::nullopt, false, cost_model);
    failures += check(cand_2.has_value(), "Admission 2 must succeed under page pressure");
    if (cand_2.has_value()) {
        failures += check(cand_2->identity_assessment().physical_status == ninfer::runtime::MaterializationPhysicalStatus::Feasible,
                          "Admission 2 must be Feasible");
    }
    auto res_2 = program.seal_identity(*cand_2, prompt_2);
    failures += check(res_2.has_value(), "Admission 2 identity seal must succeed");
    (void)program.start_resource_transaction(std::move(*res_2), make_prompt(tokens_2, true), cancellation);
    auto prog_2 = program.progress_context_transaction(cancellation);
    SequenceHandle seq_2 = std::get_if<MaterializationResult>(&prog_2)->published->sequence;
    program.finalize_context_transaction();

    // Prefill both sequences
    std::array<ninfer::runtime::CommitDecision, 1> non_term = {{{.accepted_tokens = 1, .terminal = false}}};
    auto p1_chunk1 = program.advance_prefill(seq_1); // uses freed group 1
    auto p1_chunk2 = program.advance_prefill(seq_1); // gets freed group 2 (samples 1st token)
    failures += check(p1_chunk2.complete, "Prefill 1 must complete");
    if (p1_chunk2.pending.has_value()) { (void)program.commit(std::move(*p1_chunk2.pending), commit_dec); }

    auto p2_chunk1 = program.advance_prefill(seq_2); // uses freed group 3
    auto p2_chunk2 = program.advance_prefill(seq_2); // gets freed group 4 (samples 1st token)
    failures += check(p2_chunk2.complete, "Prefill 2 must complete");
    if (p2_chunk2.pending.has_value()) { (void)program.commit(std::move(*p2_chunk2.pending), commit_dec); }

    // Decode 2 concurrent rounds with batch size 2
    std::array<SequenceHandle, 2> seqs = {seq_1, seq_2};
    std::array<ninfer::runtime::RoundBudget, 2> budgets{};
    std::array<ninfer::runtime::CommitDecision, 2> commit_term = {
        {{.accepted_tokens = 1, .terminal = true}, {.accepted_tokens = 1, .terminal = true}}};

    auto dec_batch = program.decode(seqs, budgets);
    failures += check(dec_batch.row_count() == 2, "Decode batch row count must be 2");
    (void)program.commit(std::move(dec_batch), commit_term);

    // Finish both
    FinishResult fin_1 = program.finish(seq_1);
    FinishResult fin_2 = program.finish(seq_2);

    // Release any continuations
    for (std::size_t c = 0; c < program.impl_->continuation_slots_.size(); ++c) {
        auto& c_slot = program.impl_->continuation_slots_[c];
        if (c_slot.role == detail::ContinuationSlotRole::Catalogued) {
            (void)program.release_continuation(
                ContinuationHandle(&program, static_cast<std::uint32_t>(c), c_slot.generation));
        }
    }

    const auto usage_after = program.physical_usage();
    failures += check(usage_after.device_state_slots == 0,
                      "All device state slots must be reclaimed (0)");
    failures += check(usage_after.device_main_kv_pages == 0,
                      "All physical page groups must be returned to free list (0 used pages)");

    std::printf("[DONE] test_small_pool_page_pressure_eviction, failures: %d\n", failures);
    std::fflush(stdout);
    return failures;
}

int test_repeated_8way_batches_over_full_catalog(ninfer::DeviceContext& device) {
    std::printf("[RUN] test_repeated_8way_batches_over_full_catalog\n");
    std::fflush(stdout);
    int failures = 0;
    auto model   = make_synthetic_model(device);
    auto ple_meta = make_synthetic_ple_meta();

    // Concurrency 8, max_context 256, continuation_capacity 8, prefill_chunk 128
    FlashNextRuntimeConfig cfg{
        .max_concurrency       = 8,
        .max_context           = 256,
        .state_slot_capacity   = 0,
        .continuation_capacity = 8,
        .prefill_chunk         = 128,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    auto program_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, model.view, std::nullopt, ple_meta);
    Program program(std::move(program_impl));

    ninfer::runtime::ResolvedExecutionOptions exec_options{};
    exec_options.allow_prefix_reuse     = true;
    exec_options.requested_output_tokens = 8;
    ninfer::runtime::ContextMachineCostModel cost_model{};
    std::atomic<bool> flag{false};
    ninfer::runtime::CancellationFlagView cancellation{&flag};

    std::vector<ContinuationHandle> catalog_handles(8);

    // Run 3 consecutive 8-way batches
    for (int batch_idx = 0; batch_idx < 3; ++batch_idx) {
        std::vector<SequenceHandle> sequences(8);

        for (std::uint32_t lane = 0; lane < 8; ++lane) {
            std::vector<ninfer::TokenId> tokens(128);
            for (std::size_t i = 0; i < 128; ++i) {
                tokens[i] = static_cast<ninfer::TokenId>(batch_idx * 1000 + lane * 100 + i);
            }
            const auto prompt = make_prompt(tokens, true);
            auto base = program.plan_request(prompt, exec_options);
            auto cand = program.inspect_admission(prompt, base, ninfer::runtime::LaneId(lane), nullptr, nullptr, std::nullopt, false, cost_model);
            failures += check(cand.has_value(), "Admission in 8-way batch must succeed");

            std::optional<ResourcePlan> res;
            if (batch_idx == 0) {
                res = program.seal_identity(*cand, prompt);
            } else {
                // Catalog is full: drive pressure planning to plan evicting a catalogued slot
                std::vector<ContinuationHandle> current_catalog_handles;
                current_catalog_handles.reserve(8);
                std::vector<std::uint32_t> owner_ordinals;
                for (std::size_t c = 0; c < program.impl_->continuation_slots_.size(); ++c) {
                    const auto& sl = program.impl_->continuation_slots_[c];
                    if (sl.role == detail::ContinuationSlotRole::Catalogued) {
                        current_catalog_handles.push_back(ContinuationHandle(&program, static_cast<std::uint32_t>(c), sl.generation));
                        owner_ordinals.push_back(static_cast<std::uint32_t>(c));
                    }
                }
                std::vector<const ContinuationHandle*> owners;
                for (const auto& h : current_catalog_handles) { owners.push_back(&h); }

                const AdmissionCandidate* cand_ptr = &*cand;
                auto session = program.begin_pressure_planning(cost_model, std::span(&cand_ptr, 1), owners, owner_ordinals, {}, {});
                auto prep = session.prepare_expansion(session.identity_target(*cand));
                auto view = session.commit_expansion(std::move(prep));
                failures += check(!view.children.empty(), "Expansion must provide children");
                res = session.seal(view.children[0], prompt);
            }
            failures += check(res.has_value(), "Resource plan must be created");
            (void)program.start_resource_transaction(std::move(*res), make_prompt(tokens, true), cancellation);
            auto prog = program.progress_context_transaction(cancellation);
            auto* mat = std::get_if<MaterializationResult>(&prog);
            failures += check(mat != nullptr, "MaterializationResult must not be null");
            if (batch_idx > 0 && mat != nullptr) {
                failures += check(mat->victims.size() == 1, "Must have 1 pressure eviction victim");
                if (mat->victims.size() == 1) {
                    failures += check(mat->victims[0].pressure_committed, "Victim must have pressure_committed = true");
                }
            }
            sequences[lane] = mat->published->sequence;
            program.finalize_context_transaction();
        }

        // Prefill all 8 lanes
        for (std::uint32_t lane = 0; lane < 8; ++lane) {
            auto pref = program.advance_prefill(sequences[lane]);
            failures += check(pref.complete, "Prefill must complete");
            std::array<ninfer::runtime::CommitDecision, 1> non_term = {{{.accepted_tokens = 1, .terminal = false}}};
            if (pref.pending.has_value()) {
                (void)program.commit(std::move(*pref.pending), non_term);
            }
        }

        // Decode 2 rounds for all 8 lanes concurrently
        std::array<ninfer::runtime::RoundBudget, 8> budgets{};
        std::array<ninfer::runtime::CommitDecision, 8> dec_non_term;
        for (auto& d : dec_non_term) { d = {.accepted_tokens = 1, .terminal = false}; }
        std::array<ninfer::runtime::CommitDecision, 8> dec_term;
        for (auto& d : dec_term) { d = {.accepted_tokens = 1, .terminal = true}; }

        auto dec1 = program.decode(sequences, budgets);
        failures += check(dec1.row_count() == 8, "Decode round 1 row count must be 8");
        (void)program.commit(std::move(dec1), dec_non_term);

        auto dec2 = program.decode(sequences, budgets);
        failures += check(dec2.row_count() == 8, "Decode round 2 row count must be 8");
        (void)program.commit(std::move(dec2), dec_term);

        // Finish all 8 sequences
        for (std::uint32_t lane = 0; lane < 8; ++lane) {
            FinishResult fin = program.finish(sequences[lane]);
            failures += check(fin.disposition == ninfer::runtime::FinishDisposition::Catalogued,
                              "8-way batch finish must catalogue");
            if (fin.continuation.has_value()) {
                catalog_handles[lane] = std::move(*fin.continuation);
            }
        }
    }

    // Release all continuations in catalog
    for (std::size_t c = 0; c < program.impl_->continuation_slots_.size(); ++c) {
        auto& c_slot = program.impl_->continuation_slots_[c];
        if (c_slot.role == detail::ContinuationSlotRole::Catalogued) {
            (void)program.release_continuation(
                ContinuationHandle(&program, static_cast<std::uint32_t>(c), c_slot.generation));
        }
    }

    const auto usage_after = program.physical_usage();
    failures += check(usage_after.device_state_slots == 0,
                      "All device state slots must be reclaimed (0)");
    failures += check(usage_after.device_main_kv_pages == 0,
                      "All physical page groups must be returned to free list (0 used pages)");

    std::printf("[DONE] test_repeated_8way_batches_over_full_catalog, failures: %d\n", failures);
    std::fflush(stdout);
    return failures;
}

int test_resumed_checkpoint_not_evicted_while_lane_runs(ninfer::DeviceContext& device) {
    std::printf("[RUN] test_resumed_checkpoint_not_evicted_while_lane_runs\n");
    std::fflush(stdout);
    int failures = 0;
    auto model   = make_synthetic_model(device);
    auto ple_meta = make_synthetic_ple_meta();

    FlashNextRuntimeConfig cfg{
        .max_concurrency       = 2,
        .max_context           = 256,
        .state_slot_capacity   = 0,
        .continuation_capacity = 4,
        .prefill_chunk         = 128,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    auto program_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, model.view, std::nullopt, ple_meta);
    Program program(std::move(program_impl));

    ninfer::runtime::ResolvedExecutionOptions exec_options{};
    exec_options.allow_prefix_reuse     = true;
    exec_options.requested_output_tokens = 8;
    ninfer::runtime::ContextMachineCostModel cost_model{};
    std::atomic<bool> flag{false};
    ninfer::runtime::CancellationFlagView cancellation{&flag};
    std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = true}}};

    // 1. Run Sequence 1 on Lane 0, capture TurnClosure checkpoint at 32 tokens
    std::vector<ninfer::TokenId> t1_tokens(48);
    for (std::size_t i = 0; i < 48; ++i) { t1_tokens[i] = static_cast<ninfer::TokenId>(500 + i); }
    const auto prompt1 = make_prompt(t1_tokens, true, 32);
    auto base1 = program.plan_request(prompt1, exec_options);
    auto cand1 = program.inspect_admission(prompt1, base1, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
    auto res1 = program.seal_identity(*cand1, prompt1);
    (void)program.start_resource_transaction(std::move(*res1), make_prompt(t1_tokens, true, 32), cancellation);
    auto prog1 = program.progress_context_transaction(cancellation);
    SequenceHandle seq1 = std::get_if<MaterializationResult>(&prog1)->published->sequence;
    program.finalize_context_transaction();

    auto p1 = program.advance_prefill(seq1);
    failures += check(p1.capture.has_value(), "Turn 1 must offer capture");
    (void)program.reserve_active_capture(std::move(*p1.capture), nullptr, nullptr, std::nullopt, cancellation);
    (void)program.progress_context_transaction(cancellation);
    program.finalize_context_transaction();

    auto p1_step2 = program.advance_prefill(seq1);
    failures += check(p1_step2.complete, "Turn 1 step 2 must complete");
    if (p1_step2.pending.has_value()) { (void)program.commit(std::move(*p1_step2.pending), commit_dec); }
    FinishResult fin1 = program.finish(seq1);

    // Identify which slot has the TurnClosure checkpoint
    std::int32_t tc_slot_idx = -1;
    std::uint64_t tc_gen = 0;
    for (std::size_t c = 0; c < program.impl_->continuation_slots_.size(); ++c) {
        const auto& sl = program.impl_->continuation_slots_[c];
        if (sl.role == detail::ContinuationSlotRole::Catalogued && sl.kind == runtime::CheckpointKind::TurnClosure) {
            tc_slot_idx = static_cast<std::int32_t>(c);
            tc_gen = sl.generation;
            break;
        }
    }
    failures += check(tc_slot_idx >= 0, "TurnClosure slot must exist");

    // 2. Start Turn 2 on Lane 0, resumed from TurnClosure
    std::vector<ninfer::TokenId> t2_tokens(64);
    for (std::size_t i = 0; i < 32; ++i) { t2_tokens[i] = t1_tokens[i]; }
    for (std::size_t i = 32; i < 64; ++i) { t2_tokens[i] = static_cast<ninfer::TokenId>(600 + i); }
    const auto prompt2 = make_prompt(t2_tokens, true);
    auto base2 = program.plan_request(prompt2, exec_options);
    auto cand2 = program.inspect_admission(prompt2, base2, ninfer::runtime::LaneId(0), &*fin1.continuation, nullptr, std::nullopt, false, cost_model);
    failures += check(cand2.has_value(), "Turn 2 admission against TurnClosure must succeed");
    auto res2 = program.seal_identity(*cand2, prompt2);
    (void)program.start_resource_transaction(std::move(*res2), make_prompt(t2_tokens, true), cancellation);
    auto prog2 = program.progress_context_transaction(cancellation);
    SequenceHandle seq2 = std::get_if<MaterializationResult>(&prog2)->published->sequence;
    program.finalize_context_transaction();

    // Verify Lane 0 is actively referencing tc_slot_idx
    failures += check(program.impl_->lane_states_[0].reused_from_continuation_index.has_value() &&
                      *program.impl_->lane_states_[0].reused_from_continuation_index == static_cast<std::uint32_t>(tc_slot_idx),
                      "Lane 0 must track reused continuation index");
    failures += check(program.impl_->is_slot_protected(tc_slot_idx),
                      "TurnClosure slot must be protected while Lane 0 is active");

    // 3. Trigger pressure on Lane 1 and fill remaining continuation slots
    std::vector<ninfer::TokenId> t3_tokens(64);
    for (std::size_t i = 0; i < 64; ++i) { t3_tokens[i] = static_cast<ninfer::TokenId>(700 + i); }
    const auto prompt3 = make_prompt(t3_tokens, true);
    auto base3 = program.plan_request(prompt3, exec_options);
    auto cand3 = program.inspect_admission(prompt3, base3, ninfer::runtime::LaneId(1), nullptr, nullptr, std::nullopt, false, cost_model);
    auto res3 = program.seal_identity(*cand3, prompt3);
    (void)program.start_resource_transaction(std::move(*res3), make_prompt(t3_tokens, true), cancellation);
    auto prog3 = program.progress_context_transaction(cancellation);
    SequenceHandle seq3 = std::get_if<MaterializationResult>(&prog3)->published->sequence;
    program.finalize_context_transaction();

    // Advance prefill on Lane 1 (which triggers page group and continuation eviction)
    auto p3 = program.advance_prefill(seq3);

    // Verify tc_slot_idx was NEVER evicted
    failures += check(program.impl_->continuation_slots_[tc_slot_idx].role == detail::ContinuationSlotRole::Catalogued,
                      "Resumed TurnClosure checkpoint must NOT be evicted while lane is running");
    failures += check(program.impl_->continuation_slots_[tc_slot_idx].generation == tc_gen,
                      "TurnClosure generation must be intact");

    // Clean up sequences
    if (p3.pending.has_value()) { (void)program.commit(std::move(*p3.pending), commit_dec); }
    (void)program.finish(seq3);

    auto p2 = program.advance_prefill(seq2);
    if (p2.pending.has_value()) { (void)program.commit(std::move(*p2.pending), commit_dec); }
    (void)program.finish(seq2);

    // Release all
    for (std::size_t c = 0; c < program.impl_->continuation_slots_.size(); ++c) {
        auto& c_slot = program.impl_->continuation_slots_[c];
        if (c_slot.role == detail::ContinuationSlotRole::Catalogued) {
            (void)program.release_continuation(
                ContinuationHandle(&program, static_cast<std::uint32_t>(c), c_slot.generation));
        }
    }

    const auto usage_after = program.physical_usage();
    failures += check(usage_after.device_state_slots == 0,
                      "All device state slots must be reclaimed (0)");
    failures += check(usage_after.device_main_kv_pages == 0,
                      "All physical page groups must be returned to free list (0 used pages)");

    std::printf("[DONE] test_resumed_checkpoint_not_evicted_while_lane_runs, failures: %d\n", failures);
    std::fflush(stdout);
    return failures;
}

int test_materialization_planner_call_sequence(ninfer::DeviceContext& device) {
    std::printf("[RUN] test_materialization_planner_call_sequence\n");
    std::fflush(stdout);
    int failures = 0;
    auto model   = make_synthetic_model(device);
    auto ple_meta = make_synthetic_ple_meta();

    FlashNextRuntimeConfig cfg{
        .max_concurrency       = 2,
        .max_context           = 1024,
        .state_slot_capacity   = 0,
        .continuation_capacity = 4,
        .prefill_chunk         = 128,
    };
    const auto curve = flash_next_capacity_curve(cfg);
    auto plan        = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);

    auto program_impl = std::make_unique<ProgramImpl>(nullptr, plan, device, model.view, std::nullopt, ple_meta);
    Program program(std::move(program_impl));

    ninfer::runtime::ResolvedExecutionOptions exec_options{};
    exec_options.allow_prefix_reuse     = true;
    exec_options.requested_output_tokens = 8;
    ninfer::runtime::ContextMachineCostModel cost_model{};
    std::atomic<bool> flag{false};
    ninfer::runtime::CancellationFlagView cancellation{&flag};
    std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = true}}};

    // 1. Fill catalog with 4 continuations (slots 0..3)
    std::vector<ContinuationHandle> catalog_handles;
    for (int req = 0; req < 4; ++req) {
        std::vector<ninfer::TokenId> tokens(16);
        for (std::size_t i = 0; i < 16; ++i) { tokens[i] = static_cast<ninfer::TokenId>(req * 100 + i); }
        const auto prompt = make_prompt(tokens, true);
        auto base = program.plan_request(prompt, exec_options);
        auto cand = program.inspect_admission(prompt, base, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
        failures += check(cand.has_value(), "Fill admission must succeed");
        auto res = program.seal_identity(*cand, prompt);
        (void)program.start_resource_transaction(std::move(*res), make_prompt(tokens, true), cancellation);
        auto prog = program.progress_context_transaction(cancellation);
        SequenceHandle seq = std::get_if<MaterializationResult>(&prog)->published->sequence;
        program.finalize_context_transaction();

        auto pref = program.advance_prefill(seq);
        if (pref.pending.has_value()) { (void)program.commit(std::move(*pref.pending), commit_dec); }
        FinishResult fin = program.finish(seq);
        failures += check(fin.disposition == ninfer::runtime::FinishDisposition::Catalogued, "Fill finish must catalogue");
        if (fin.continuation.has_value()) {
            catalog_handles.push_back(std::move(*fin.continuation));
        }
    }
    failures += check(catalog_handles.size() == 4, "Must have 4 catalogued handles");

    // 2. Drive exact MaterializationPlanner call sequence for a new root request:
    // Plan request 5
    std::vector<ninfer::TokenId> r5_tokens(16);
    for (std::size_t i = 0; i < 16; ++i) { r5_tokens[i] = static_cast<ninfer::TokenId>(500 + i); }
    const auto prompt5 = make_prompt(r5_tokens, true);
    auto base5 = program.plan_request(prompt5, exec_options);
    auto cand5 = program.inspect_admission(prompt5, base5, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
    failures += check(cand5.has_value(), "Admission 5 must succeed");

    std::vector<const ContinuationHandle*> owners;
    std::vector<std::uint32_t> owner_ordinals;
    for (std::size_t i = 0; i < 4; ++i) {
        owners.push_back(&catalog_handles[i]);
        owner_ordinals.push_back(static_cast<std::uint32_t>(i));
    }

    std::optional<ResourcePlan> sealed_plan;
    {
        // begin_pressure_planning
        const AdmissionCandidate* cand5_ptr = &*cand5;
        auto session = program.begin_pressure_planning(cost_model, std::span(&cand5_ptr, 1), owners, owner_ordinals, {}, {});

        // root_maximal_target -> assess
        auto root_max = session.root_maximal_target(*cand5);
        auto assess_max = session.assess(root_max);
        failures += check(assess_max.physical_status == ninfer::runtime::MaterializationPhysicalStatus::Feasible,
                          "Root maximal physical status must be Feasible");
        failures += check(assess_max.owner_outcomes.size() == 4,
                          "Root maximal must evict all 4 non-protected owners");
        for (std::size_t i = 0; i < 4; ++i) {
            failures += check(assess_max.owner_outcomes[i].disposition == ninfer::runtime::ClaimDisposition::Evicted,
                              "Owner outcome must be Evicted");
        }

        // identity_target -> prepare_expansion -> commit_expansion
        auto ident = session.identity_target(*cand5);
        auto prep = session.prepare_expansion(ident);
        failures += check(prep.new_canonical_count() == 4, "Prepared expansion must have 4 new canonical targets");
        auto view = session.commit_expansion(std::move(prep));
        failures += check(view.children.size() == 4, "Committed expansion view must have 4 children");

        // Assess child 0 (evicts only owner 0)
        auto assess_child0 = session.assess(view.children[0]);
        failures += check(assess_child0.owner_outcomes.size() == 1,
                          "Child 0 assessment must have 1 owner outcome");
        failures += check(assess_child0.owner_outcomes[0].owner_ordinal == 0,
                          "Child 0 must evict owner ordinal 0");
        failures += check(assess_child0.owner_outcomes[0].disposition == ninfer::runtime::ClaimDisposition::Evicted,
                          "Child 0 outcome must be Evicted");

        // seal
        sealed_plan = session.seal(view.children[0], prompt5);
    }
    failures += check(sealed_plan.has_value(), "Sealed plan must be valid");

    auto reserve_status = program.start_resource_transaction(std::move(*sealed_plan), make_prompt(r5_tokens, true), cancellation);
    failures += check(reserve_status == ninfer::runtime::ContextTransactionReserveStatus::Reserved,
                      "Resource transaction with pressure evictions must be Reserved");

    auto progress = program.progress_context_transaction(cancellation);
    auto* mat = std::get_if<MaterializationResult>(&progress);
    failures += check(mat != nullptr, "MaterializationResult must not be null");
    if (mat != nullptr) {
        failures += check(mat->victims.size() == 1, "Must have exactly 1 victim in MaterializationResult");
        if (mat->victims.size() == 1) {
            failures += check(mat->victims[0].disposition == ninfer::runtime::ClaimDisposition::Evicted,
                              "Victim disposition must be Evicted");
            failures += check(mat->victims[0].pressure_committed == true,
                              "Victim must have pressure_committed == true");
            failures += check(!mat->victims[0].final_summary.has_value(),
                              "Victim final_summary must be nullopt");
        }
        SequenceHandle seq5 = mat->published->sequence;
        program.finalize_context_transaction();

        auto pref5 = program.advance_prefill(seq5);
        if (pref5.pending.has_value()) { (void)program.commit(std::move(*pref5.pending), commit_dec); }
        FinishResult fin5 = program.finish(seq5);
        failures += check(fin5.disposition == ninfer::runtime::FinishDisposition::Catalogued,
                          "Finish 5 must catalogue into newly vacated slot 0");
        if (fin5.continuation.has_value()) {
            catalog_handles[0] = std::move(*fin5.continuation);
        }
    }

    // Clean up remaining continuations
    for (auto& c : catalog_handles) {
        (void)program.release_continuation(std::move(c));
    }
    failures += check(program.physical_usage().device_state_slots == 0,
                      "All state slots must be reclaimed after test");

    std::printf("[DONE] test_materialization_planner_call_sequence, failures: %d\n", failures);
    std::fflush(stdout);
    return failures;
}

} // namespace

int main() {
    std::printf("Starting continuation lifecycle tests...\n");
    std::fflush(stdout);
    try {
        int device_count = 0;
        const cudaError_t err = cudaGetDeviceCount(&device_count);
        if (cuda_unavailable(err) || device_count == 0) {
            std::printf("CUDA device unavailable; skipped GPU Program lifecycle execution.\n");
            std::fflush(stdout);
            return 0;
        }

        int failures = 0;
        ninfer::DeviceContext device(0);
        failures += test_prefix_shortlist_key(device);
        failures += test_continuation_lifecycle_and_reuse(device);
        failures += test_continuation_mismatch_fallback(device);
        failures += test_continuation_lru_eviction(device);
        failures += test_turn_closure_checkpoint_and_multi_turn_reuse(device);
        failures += test_turn_closure_chain_retention(device);
        failures += test_turn2_resumed_vs_scratch_divergence(device);
        failures += test_state_slot_capacity_saturation_and_eviction(device);
        failures += test_small_pool_page_pressure_eviction(device);
        failures += test_repeated_8way_batches_over_full_catalog(device);
        failures += test_resumed_checkpoint_not_evicted_while_lane_runs(device);
        failures += test_materialization_planner_call_sequence(device);

        if (failures == 0) {
            std::printf("ALL FLASH-NEXT CONTINUATION LIFECYCLE TESTS PASSED\n");
            std::fflush(stdout);
            return 0;
        }
        std::fprintf(stderr, "%d TEST FAILURES OCCURRED\n", failures);
        std::fflush(stderr);
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXCEPTION CAUGHT: %s\n", e.what());
        std::fflush(stderr);
        return 1;
    }
}
