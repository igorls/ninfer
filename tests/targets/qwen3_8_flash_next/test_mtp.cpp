#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/expert_bank.h"
#include "targets/qwen3_8_flash_next/impl/hyper_connection.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"
#include "targets/qwen3_8_flash_next/impl/moe.h"
#include "targets/qwen3_8_flash_next/impl/mtp_forward.h"
#include "targets/qwen3_8_flash_next/impl/mtp_forward_kernels.h"
#include "targets/qwen3_8_flash_next/impl/qsa_attention.h"
#include "targets/qwen3_8_flash_next/impl/state_dumper.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

std::uint16_t float_to_bf16(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t lsb  = (bits >> 16U) & 1U;
    const std::uint32_t bias = 0x7FFFU + lsb;
    return static_cast<std::uint16_t>((bits + bias) >> 16U);
}

float bf16_to_float(std::uint16_t val) {
    const std::uint32_t bits = static_cast<std::uint32_t>(val) << 16U;
    float f                  = 0.0f;
    std::memcpy(&f, &bits, sizeof(float));
    return f;
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

int test_stem_combine_and_repeat(ninfer::DeviceContext& device) {
    std::cout << "[TEST 1/3] test_stem_combine_and_repeat ...\n" << std::flush;
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t batch = 2;
    constexpr std::int32_t dim   = 2'560;

    ninfer::DeviceArena arena(64 * 1024 * 1024);
    ninfer::Tensor emb_proj     = arena.alloc(ninfer::DType::BF16, {dim, batch}, 256);
    ninfer::Tensor hid_proj     = arena.alloc(ninfer::DType::BF16, {dim, batch}, 256);
    ninfer::Tensor trunk_sum    = arena.alloc(ninfer::DType::BF16, {dim, batch}, 256);
    ninfer::Tensor hyper_hidden = arena.alloc(ninfer::DType::BF16, {10'240, batch}, 256);

    std::vector<std::uint16_t> host_emb(dim * batch);
    std::vector<std::uint16_t> host_hid(dim * batch);
    for (std::size_t i = 0; i < dim * batch; ++i) {
        host_emb[i] = float_to_bf16(static_cast<float>(i % 100) * 0.1f);
        host_hid[i] = float_to_bf16(static_cast<float>(i % 50) * 0.2f);
    }
    cudaMemcpy(emb_proj.data, host_emb.data(), host_emb.size() * sizeof(std::uint16_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(hid_proj.data, host_hid.data(), host_hid.size() * sizeof(std::uint16_t),
               cudaMemcpyHostToDevice);

    flash_next_mtp_stem_combine_and_repeat_launch(emb_proj, hid_proj, &trunk_sum, hyper_hidden,
                                                  device.stream);
    device.synchronize();

    std::vector<std::uint16_t> res_trunk(dim * batch);
    std::vector<std::uint16_t> res_hyper(10'240 * batch);
    cudaMemcpy(res_trunk.data(), trunk_sum.data, res_trunk.size() * sizeof(std::uint16_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(res_hyper.data(), hyper_hidden.data, res_hyper.size() * sizeof(std::uint16_t),
               cudaMemcpyDeviceToHost);

    double base_sq = 0.0;
    for (std::size_t i = 0; i < dim * batch; ++i) {
        float e = bf16_to_float(host_emb[i]);
        float h = bf16_to_float(host_hid[i]);
        float expected = bf16_to_float(float_to_bf16(e + h));
        float actual = bf16_to_float(res_trunk[i]);
        if (std::abs(expected - actual) > 1e-4f) {
            std::cerr << "FAIL: stem combine mismatch at " << i << " expected " << expected
                      << " got " << actual << "\n";
            return 1;
        }
        base_sq += actual * actual;
    }
    if (base_sq <= 0.0) {
        std::cerr << "FAIL: trunk_sum is vacuous zero\n";
        return 1;
    }

    // Verify hyper broadcast across 4 streams
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t s = 0; s < 4; ++s) {
            for (std::int32_t c = 0; c < dim; ++c) {
                std::uint16_t trunk_val = res_trunk[b * dim + c];
                std::uint16_t hyper_val = res_hyper[b * 10'240 + s * dim + c];
                if (trunk_val != hyper_val) {
                    std::cerr << "FAIL: hyper broadcast stream " << s << " mismatch at col " << c
                              << "\n";
                    return 1;
                }
            }
        }
    }

    std::cout << "PASS: test_stem_combine_and_repeat (base_sq=" << base_sq << ")\n";
    return 0;
}

int test_mtp_moe_bf16_synthetic(ninfer::DeviceContext& device) {
    std::cout << "[TEST 2/3] test_mtp_moe_bf16_synthetic ...\n" << std::flush;
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t batch = 2;
    constexpr std::int32_t dim   = 2'560;

    ninfer::DeviceArena arena(256 * 1024 * 1024);
    ninfer::Tensor input              = arena.alloc(ninfer::DType::BF16, {dim, batch}, 256);
    ninfer::Tensor output             = arena.alloc(ninfer::DType::BF16, {dim, batch}, 256);
    ninfer::Tensor router_t           = arena.alloc(ninfer::DType::BF16, {512, dim}, 256);
    ninfer::Tensor shared_down_t      = arena.alloc(ninfer::DType::BF16, {dim, 640}, 256);
    ninfer::Tensor shared_gate_t      = arena.alloc(ninfer::DType::BF16, {640, dim}, 256);
    ninfer::Tensor shared_up_t        = arena.alloc(ninfer::DType::BF16, {640, dim}, 256);
    ninfer::Tensor shared_gate_weight_t = arena.alloc(ninfer::DType::BF16, {1, dim}, 256);

    constexpr std::uint64_t gate_bytes_per_exp = 1'280ULL * dim * 2;
    constexpr std::uint64_t down_bytes_per_exp = dim * 640ULL * 2;
    void* exp_gate_up_ptr = arena.alloc_bytes(10 * gate_bytes_per_exp, 256).data;
    void* exp_down_ptr    = arena.alloc_bytes(10 * down_bytes_per_exp, 256).data;

    std::vector<std::uint16_t> host_input(dim * batch);
    for (std::size_t i = 0; i < host_input.size(); ++i) {
        host_input[i] = float_to_bf16(0.1f * ((i % 17) + 1));
    }
    cudaMemcpy(input.data, host_input.data(), host_input.size() * 2, cudaMemcpyHostToDevice);
    cudaMemset(output.data, 0, output.bytes());
    cudaMemset(router_t.data, 0, router_t.bytes());
    cudaMemset(shared_down_t.data, 0, shared_down_t.bytes());
    cudaMemset(shared_gate_t.data, 0, shared_gate_t.bytes());
    cudaMemset(shared_up_t.data, 0, shared_up_t.bytes());
    cudaMemset(shared_gate_weight_t.data, 0, shared_gate_weight_t.bytes());

    std::vector<std::uint16_t> host_gate_up(10 * 1'280 * dim, 0x3F80U); // 1.0 BF16
    std::vector<std::uint16_t> host_down(10 * dim * 640, 0x3800U);    // 0.25 BF16
    cudaMemcpy(exp_gate_up_ptr, host_gate_up.data(), host_gate_up.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(exp_down_ptr, host_down.data(), host_down.size() * 2, cudaMemcpyHostToDevice);

    Bf16ExpertBankView gate_up_bank{
        .data             = static_cast<const std::byte*>(exp_gate_up_ptr),
        .experts          = 512,
        .rows             = 1'280,
        .columns          = dim,
        .bytes_per_expert = gate_bytes_per_exp,
    };
    Bf16ExpertBankView down_bank{
        .data             = static_cast<const std::byte*>(exp_down_ptr),
        .experts          = 512,
        .rows             = dim,
        .columns          = 640,
        .bytes_per_expert = down_bytes_per_exp,
    };

    MoeBf16Weights weights{
        .router             = bf16_weight(router_t.data, 512, dim),
        .shared_down        = bf16_weight(shared_down_t.data, dim, 640),
        .shared_gate        = bf16_weight(shared_gate_t.data, 640, dim),
        .shared_up          = bf16_weight(shared_up_t.data, 640, dim),
        .shared_gate_weight = bf16_weight(shared_gate_weight_t.data, 1, dim),
        .expert_gate_up     = gate_up_bank,
        .expert_down        = down_bank,
    };

    flash_next_moe_bf16(input, weights, output, arena, device.stream);
    device.synchronize();

    std::vector<std::uint16_t> host_out1(dim * batch);
    cudaMemcpy(host_out1.data(), output.data, host_out1.size() * 2, cudaMemcpyDeviceToHost);

    double base_sq = 0.0;
    for (std::size_t i = 0; i < host_out1.size(); ++i) {
        float val = bf16_to_float(host_out1[i]);
        if (!std::isfinite(val)) {
            std::cerr << "FAIL: non-finite output at index " << i << "\n";
            return 1;
        }
        base_sq += val * val;
    }
    if (base_sq <= 0.0) {
        std::cerr << "FAIL: MoE output is vacuous zero\n";
        return 1;
    }

    // Determinism check
    cudaMemset(output.data, 0, output.bytes());
    flash_next_moe_bf16(input, weights, output, arena, device.stream);
    device.synchronize();

    std::vector<std::uint16_t> host_out2(dim * batch);
    cudaMemcpy(host_out2.data(), output.data, host_out2.size() * 2, cudaMemcpyDeviceToHost);

    if (std::memcmp(host_out1.data(), host_out2.data(), host_out1.size() * 2) != 0) {
        std::cerr << "FAIL: MoE execution is non-deterministic between consecutive runs\n";
        return 1;
    }

    std::cout << "PASS: test_mtp_moe_bf16_synthetic (base_sq=" << base_sq << ", bit-exact determinism verified)\n";
    return 0;
}

int test_mtp_full_step_synthetic(ninfer::DeviceContext& device, const std::string& dump_dir = "") {
    std::cout << "[TEST 3/3] test_mtp_full_step_synthetic ...\n" << std::flush;
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    constexpr std::int32_t batch = 1;
    constexpr std::int32_t dim   = 2'560;
    constexpr std::int32_t vocab = 248'320;

    // Single 1.8 GB arena allocation for ALL test weights, activations, and workspaces
    ninfer::DeviceArena arena(1'800ULL * 1024 * 1024);

    // Weights
    ninfer::Tensor emb_proj_t      = arena.alloc(ninfer::DType::BF16, {dim, dim}, 256);
    ninfer::Tensor hid_proj_t      = arena.alloc(ninfer::DType::BF16, {dim, dim}, 256);
    ninfer::Tensor emb_norm_t      = arena.alloc(ninfer::DType::BF16, {dim}, 256);
    ninfer::Tensor hid_norm_t      = arena.alloc(ninfer::DType::BF16, {10'240}, 256);
    ninfer::Tensor mix_norm_t      = arena.alloc(ninfer::DType::BF16, {10'240}, 256);
    ninfer::Tensor mix_down_t      = arena.alloc(ninfer::DType::BF16, {320, 10'240}, 256);
    ninfer::Tensor mix_up_t        = arena.alloc(ninfer::DType::BF16, {10'240, 320}, 256);
    ninfer::Tensor attn_hc_inj_t   = arena.alloc(ninfer::DType::BF16, {4, 10'240}, 256);
    ninfer::Tensor attn_hc_norm_t  = arena.alloc(ninfer::DType::BF16, {10'240}, 256);
    ninfer::Tensor attn_hc_down_t  = arena.alloc(ninfer::DType::BF16, {320, 10'240}, 256);
    ninfer::Tensor attn_hc_up_t    = arena.alloc(ninfer::DType::BF16, {10'240, 320}, 256);
    ninfer::Tensor mlp_hc_inj_t    = arena.alloc(ninfer::DType::BF16, {4, 10'240}, 256);
    ninfer::Tensor mlp_hc_norm_t   = arena.alloc(ninfer::DType::BF16, {10'240}, 256);
    ninfer::Tensor mlp_hc_down_t   = arena.alloc(ninfer::DType::BF16, {320, 10'240}, 256);
    ninfer::Tensor mlp_hc_up_t     = arena.alloc(ninfer::DType::BF16, {10'240, 320}, 256);
    ninfer::Tensor qk_idx_t        = arena.alloc(ninfer::DType::BF16, {640, dim}, 256);
    ninfer::Tensor qk_knorm_t      = arena.alloc(ninfer::DType::BF16, {128}, 256);
    ninfer::Tensor qk_qnorm_t      = arena.alloc(ninfer::DType::BF16, {128}, 256);
    ninfer::Tensor q_norm_t        = arena.alloc(ninfer::DType::BF16, {256}, 256);
    ninfer::Tensor k_norm_t        = arena.alloc(ninfer::DType::BF16, {256}, 256);
    ninfer::Tensor qgkv_t          = arena.alloc(ninfer::DType::BF16, {13'312, dim}, 256);
    ninfer::Tensor o_proj_t        = arena.alloc(ninfer::DType::BF16, {dim, 6'144}, 256);
    ninfer::Tensor router_t        = arena.alloc(ninfer::DType::BF16, {512, dim}, 256);
    ninfer::Tensor shared_down_t   = arena.alloc(ninfer::DType::BF16, {dim, 640}, 256);
    ninfer::Tensor shared_gate_t   = arena.alloc(ninfer::DType::BF16, {640, dim}, 256);
    ninfer::Tensor shared_up_t     = arena.alloc(ninfer::DType::BF16, {640, dim}, 256);
    ninfer::Tensor shared_gw_t     = arena.alloc(ninfer::DType::BF16, {1, dim}, 256);

    constexpr std::uint64_t gate_bytes_per_exp = 1'280ULL * dim * 2;
    constexpr std::uint64_t down_bytes_per_exp = dim * 640ULL * 2;
    void* exp_gate_up_ptr = arena.alloc_bytes(10 * gate_bytes_per_exp, 256).data;
    void* exp_down_ptr    = arena.alloc_bytes(10 * down_bytes_per_exp, 256).data;

    ninfer::Tensor head_t = arena.alloc(ninfer::DType::BF16, {vocab, dim}, 256);

    // Activations
    ninfer::Tensor input_emb       = arena.alloc(ninfer::DType::BF16, {dim, batch}, 256);
    ninfer::Tensor backbone_hidden = arena.alloc(ninfer::DType::BF16, {10'240, batch}, 256);
    ninfer::Tensor token_indices   = arena.alloc(ninfer::DType::I32, {batch}, 16);
    ninfer::Tensor mrope_positions = arena.alloc(ninfer::DType::I32, {batch, 3}, 16);
    ninfer::Tensor table_rows      = arena.alloc(ninfer::DType::I32, {batch}, 16);
    ninfer::Tensor selected_blocks = arena.alloc(ninfer::DType::I32, {512, batch}, 16);
    ninfer::Tensor selected_counts = arena.alloc(ninfer::DType::I32, {batch}, 16);
    ninfer::Tensor draft_logits    = arena.alloc(ninfer::DType::BF16, {vocab, batch}, 256);
    ninfer::Tensor draft_tokens    = arena.alloc(ninfer::DType::I32, {batch}, 16);

    // MTP KV Cache
    ninfer::Tensor key_pages    = arena.alloc(ninfer::DType::BF16, {256, 64, 2, 64}, 256);
    ninfer::Tensor value_pages  = arena.alloc(ninfer::DType::BF16, {256, 64, 2, 64}, 256);
    ninfer::Tensor block_tables = arena.alloc(ninfer::DType::I32, {64, 1}, 16);

    // Clear and initialize
    cudaMemset(arena.base(), 0, arena.capacity());

    // Populate diag identity on projections
    std::vector<std::uint16_t> diag(dim * dim, 0);
    for (int i = 0; i < dim; ++i) {
        diag[i * dim + i] = 0x3F80U; // 1.0 BF16
    }
    cudaMemcpy(emb_proj_t.data, diag.data(), diag.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(hid_proj_t.data, diag.data(), diag.size() * 2, cudaMemcpyHostToDevice);

    std::vector<std::uint16_t> head_prefix(100 * dim, 0);
    for (int i = 0; i < 100; ++i) {
        head_prefix[i * dim] = float_to_bf16(static_cast<float>(i + 1));
    }
    cudaMemcpy(head_t.data, head_prefix.data(), head_prefix.size() * 2, cudaMemcpyHostToDevice);

    std::vector<std::uint16_t> host_input_emb(dim * batch, 0x3F80U);
    std::vector<std::uint16_t> host_backbone(10'240 * batch, 0x3F80U);
    cudaMemcpy(input_emb.data, host_input_emb.data(), host_input_emb.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(backbone_hidden.data, host_backbone.data(), host_backbone.size() * 2, cudaMemcpyHostToDevice);

    std::vector<std::uint16_t> host_gate_up(10 * 1'280 * dim, 0x3F80U); // 1.0 BF16
    std::vector<std::uint16_t> host_down(10 * dim * 640, 0x3800U);    // 0.25 BF16
    cudaMemcpy(exp_gate_up_ptr, host_gate_up.data(), host_gate_up.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(exp_down_ptr, host_down.data(), host_down.size() * 2, cudaMemcpyHostToDevice);

    std::vector<std::uint16_t> host_qgkv(13'312 * dim, 0x3800U); // 0.25 BF16
    std::vector<std::uint16_t> host_oproj(dim * 6'144, 0x3800U); // 0.25 BF16
    cudaMemcpy(qgkv_t.data, host_qgkv.data(), host_qgkv.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(o_proj_t.data, host_oproj.data(), host_oproj.size() * 2, cudaMemcpyHostToDevice);

    Bf16ExpertBankView gate_up_bank{
        .data             = static_cast<const std::byte*>(exp_gate_up_ptr),
        .experts          = 512,
        .rows             = 1'280,
        .columns          = dim,
        .bytes_per_expert = gate_bytes_per_exp,
    };
    Bf16ExpertBankView down_bank{
        .data             = static_cast<const std::byte*>(exp_down_ptr),
        .experts          = 512,
        .rows             = dim,
        .columns          = 640,
        .bytes_per_expert = down_bytes_per_exp,
    };

    MtpModelView mtp{
        .embedding_projection = bf16_weight(emb_proj_t.data, dim, dim),
        .hidden_projection    = bf16_weight(hid_proj_t.data, dim, dim),
        .mixer = {
            .norm           = mix_norm_t,
            .input_mix_down = bf16_weight(mix_down_t.data, 320, 10'240),
            .input_mix_up   = bf16_weight(mix_up_t.data, 10'240, 320),
        },
        .attention_hyper = {
            .block_inject   = bf16_weight(attn_hc_inj_t.data, 4, 10'240),
            .norm           = attn_hc_norm_t,
            .input_mix_down = bf16_weight(attn_hc_down_t.data, 320, 10'240),
            .input_mix_up   = bf16_weight(attn_hc_up_t.data, 10'240, 320),
        },
        .attention = {
            .indexer_query_key    = bf16_weight(qk_idx_t.data, 640, dim),
            .indexer_key_norm     = qk_knorm_t,
            .indexer_query_norm   = qk_qnorm_t,
            .key_norm             = k_norm_t,
            .query_norm           = q_norm_t,
            .query_gate_key_value = bf16_weight(qgkv_t.data, 13'312, dim),
            .output               = bf16_weight(o_proj_t.data, dim, 6'144),
        },
        .mlp_hyper = {
            .block_inject   = bf16_weight(mlp_hc_inj_t.data, 4, 10'240),
            .norm           = mlp_hc_norm_t,
            .input_mix_down = bf16_weight(mlp_hc_down_t.data, 320, 10'240),
            .input_mix_up   = bf16_weight(mlp_hc_up_t.data, 10'240, 320),
        },
        .moe = {
            .router             = bf16_weight(router_t.data, 512, dim),
            .shared_down        = bf16_weight(shared_down_t.data, dim, 640),
            .shared_gate        = bf16_weight(shared_gate_t.data, 640, dim),
            .shared_up          = bf16_weight(shared_up_t.data, 640, dim),
            .shared_gate_weight = bf16_weight(shared_gw_t.data, 1, dim),
            .expert_gate_up     = gate_up_bank,
            .expert_down        = down_bank,
        },
        .embedding_norm = emb_norm_t,
        .hidden_norm    = hid_norm_t,
    };

    TextModelView model{};
    model.output_head = bf16_weight(head_t.data, vocab, dim);
    model.mtp         = mtp;

    QsaAttentionCacheView mtp_cache{
        .key_pages    = key_pages,
        .value_pages  = value_pages,
        .block_tables = block_tables,
    };
 
    std::unique_ptr<StateDumper> dumper;
    std::optional<FlashNextDecodeStateSink> sink;
    if (!dump_dir.empty()) {
        dumper = std::make_unique<StateDumper>(dump_dir);
        sink   = dumper->make_sink(0, 0, {0, 0, 0});
    }

    flash_next_mtp_step(model, input_emb, backbone_hidden, token_indices, mrope_positions,
                        table_rows, selected_blocks, selected_counts, mtp_cache, arena,
                        draft_logits, draft_tokens, device.stream, sink ? &*sink : nullptr);
    device.synchronize();

    if (dumper) {
        dumper->write_manifest();
    }

    std::vector<std::uint16_t> host_logits(vocab * batch);
    std::vector<std::int32_t> host_drafts(batch);
    cudaMemcpy(host_logits.data(), draft_logits.data, host_logits.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_drafts.data(), draft_tokens.data, host_drafts.size() * sizeof(std::int32_t),
               cudaMemcpyDeviceToHost);

    double base_sq = 0.0;
    for (std::size_t i = 0; i < host_logits.size(); ++i) {
        float val = bf16_to_float(host_logits[i]);
        if (!std::isfinite(val)) {
            std::cerr << "FAIL: non-finite draft logit at " << i << "\n";
            return 1;
        }
        base_sq += val * val;
    }
    if (base_sq <= 0.0) {
        std::cerr << "FAIL: draft logits are vacuous zero\n";
        return 1;
    }

    if (host_drafts[0] < 0 || host_drafts[0] >= vocab) {
        std::cerr << "FAIL: draft token out of valid range: " << host_drafts[0] << "\n";
        return 1;
    }

    std::cout << "PASS: test_mtp_full_step_synthetic (draft_token=" << host_drafts[0]
              << ", base_sq=" << base_sq << ")\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
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

    std::string dump_dir;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--dump-states" && i + 1 < argc) {
            dump_dir = argv[++i];
        }
    }

    ninfer::DeviceContext device(0);

    if (test_stem_combine_and_repeat(device) != 0) return 1;
    if (test_mtp_moe_bf16_synthetic(device) != 0) return 1;
    if (test_mtp_full_step_synthetic(device, dump_dir) != 0) return 1;

    std::cout << "ALL MTP TESTS PASSED\n";
    return 0;
}
