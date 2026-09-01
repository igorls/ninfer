#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/hyper_connection.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
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

struct RelL2Stats {
    double rel         = 0.0;
    double base_sq     = 0.0;
    double act_sq      = 0.0;
    double dot         = 0.0;
    double max_abs_act = 0.0;
    double max_abs_ref = 0.0;
    int n              = 0;
    int nfinite        = 0;
    int nnan           = 0;
    int nzero          = 0;
};

RelL2Stats rel_l2_stats(const float* act, const float* ref, std::size_t n) {
    RelL2Stats s;
    s.n = static_cast<int>(n);
    double diff_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double a = static_cast<double>(act[i]);
        const double e = static_cast<double>(ref[i]);
        if (!std::isfinite(a) || !std::isfinite(e)) {
            ++s.nnan;
            continue;
        }
        ++s.nfinite;
        if (a == 0.0) {
            ++s.nzero;
        }
        s.act_sq += a * a;
        s.base_sq += e * e;
        s.dot += a * e;
        const double d = a - e;
        diff_sq += d * d;
        s.max_abs_act = std::max(s.max_abs_act, std::abs(a));
        s.max_abs_ref = std::max(s.max_abs_ref, std::abs(e));
    }
    if (s.base_sq > 0.0) {
        s.rel = std::sqrt(diff_sq / s.base_sq);
    } else if (diff_sq > 0.0) {
        s.rel = std::sqrt(diff_sq);
    }
    return s;
}

void print_rel_l2_stats(const char* label, const RelL2Stats& s) {
    const double corr =
        (s.act_sq > 0.0 && s.base_sq > 0.0) ? (s.dot / std::sqrt(s.act_sq * s.base_sq)) : 0.0;
    std::cout << "    " << label << ": rel-L2=" << s.rel << " base_sq=" << s.base_sq
              << " act_sq=" << s.act_sq << " corr=" << corr << " nfinite=" << s.nfinite << "/" << s.n
              << " nnan=" << s.nnan << " nzero=" << s.nzero << " max_abs_act=" << s.max_abs_act
              << " max_abs_ref=" << s.max_abs_ref << "\n";
}

bool accept_stage(const char* stage, int tokens, const RelL2Stats& s, double* max_rel,
                  double tolerance) {
    if (s.nnan > 0 || s.nfinite != s.n || s.base_sq <= 0.0 || !std::isfinite(s.rel) ||
        !std::isfinite(s.base_sq)) {
        std::cerr << "VACUOUS/NaN: T=" << tokens << " " << stage << " nnan=" << s.nnan
                  << " nfinite=" << s.nfinite << "/" << s.n << " base_sq=" << s.base_sq
                  << " rel=" << s.rel << "\n";
        return false;
    }
    *max_rel = std::max(*max_rel, s.rel);
    if (s.rel > tolerance) {
        std::cerr << "FAIL: T=" << tokens << " " << stage << " rel-L2=" << s.rel << " exceeds "
                  << tolerance << " (base_sq=" << s.base_sq << ")\n";
        return false;
    }
    return true;
}

struct HyperReferenceOutput {
    std::vector<float> normalized;
    std::vector<float> low_rank;
    std::vector<float> injection;
    std::vector<float> block_input;
    std::vector<float> hidden_injected;
};

HyperReferenceOutput evaluate_reference(
    int tokens,
    const std::vector<float>& host_hidden,
    const std::vector<float>& host_norm,
    const std::vector<float>& host_down,
    const std::vector<float>& host_up,
    const std::vector<float>& host_inject,
    const std::vector<float>& host_block_output) {
    HyperReferenceOutput out;
    out.normalized.resize(static_cast<std::size_t>(tokens) * 10'240);
    out.low_rank.resize(static_cast<std::size_t>(tokens) * 320);
    out.injection.resize(static_cast<std::size_t>(tokens) * 4);
    out.block_input.resize(static_cast<std::size_t>(tokens) * 2'560);
    out.hidden_injected.resize(static_cast<std::size_t>(tokens) * 10'240);

    for (int t = 0; t < tokens; ++t) {
        // 1. Group Norm
        for (int s = 0; s < 4; ++s) {
            float sum_sq = 0.0F;
            for (int c = 0; c < 2'560; ++c) {
                float v = host_hidden[static_cast<std::size_t>(t) * 10'240 + s * 2'560 + c];
                sum_sq += v * v;
            }
            float inv_rms = 1.0F / std::sqrt(sum_sq / 2'560.0F + 1.0e-6F);
            for (int c = 0; c < 2'560; ++c) {
                int flat = s * 2'560 + c;
                float v  = host_hidden[static_cast<std::size_t>(t) * 10'240 + flat];
                float n  = host_norm[flat];
                float normed = v * inv_rms * (1.0F + n);
                out.normalized[static_cast<std::size_t>(t) * 10'240 + flat] =
                    bf16_to_float(float_to_bf16(normed));
            }
        }

        // 2. Low Rank Down Projection
        for (int r = 0; r < 320; ++r) {
            float sum = 0.0F;
            for (int c = 0; c < 10'240; ++c) {
                sum += host_down[static_cast<std::size_t>(r) * 10'240 + c] *
                       out.normalized[static_cast<std::size_t>(t) * 10'240 + c];
            }
            float act = (sum * 0.25F) / (1.0F + std::exp(-(sum * 0.25F)));
            out.low_rank[static_cast<std::size_t>(t) * 320 + r] =
                bf16_to_float(float_to_bf16(act));
        }

        // 3. Injection Projection
        for (int s = 0; s < 4; ++s) {
            float sum = 0.0F;
            for (int c = 0; c < 10'240; ++c) {
                sum += host_inject[static_cast<std::size_t>(s) * 10'240 + c] *
                       out.normalized[static_cast<std::size_t>(t) * 10'240 + c];
            }
            float act = 2.0F / (1.0F + std::exp(-(sum * 0.25F)));
            out.injection[static_cast<std::size_t>(t) * 4 + s] = act;
        }

        // 4. Mix Up Projection
        for (int h = 0; h < 2'560; ++h) {
            float mean_contrib = 0.0F;
            for (int s = 0; s < 4; ++s) {
                int row   = s * 2'560 + h;
                float sum = 0.0F;
                for (int c = 0; c < 320; ++c) {
                    sum += host_up[static_cast<std::size_t>(row) * 320 + c] *
                           out.low_rank[static_cast<std::size_t>(t) * 320 + c];
                }
                float mix_gate = 1.0F / (1.0F + std::exp(-sum));
                mean_contrib  += mix_gate * out.normalized[static_cast<std::size_t>(t) * 10'240 + row];
            }
            out.block_input[static_cast<std::size_t>(t) * 2'560 + h] =
                bf16_to_float(float_to_bf16(mean_contrib * 0.25F));
        }

        // 5. Inject
        for (int s = 0; s < 4; ++s) {
            float inj = out.injection[static_cast<std::size_t>(t) * 4 + s];
            for (int h = 0; h < 2'560; ++h) {
                int flat   = s * 2'560 + h;
                float orig = host_hidden[static_cast<std::size_t>(t) * 10'240 + flat];
                float upd  = host_block_output[static_cast<std::size_t>(t) * 2'560 + h] * inj;
                out.hidden_injected[static_cast<std::size_t>(t) * 10'240 + flat] =
                    bf16_to_float(float_to_bf16(orig + upd));
            }
        }
    }
    return out;
}

int test_basic_unit_injection(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    ninfer::DeviceBuffer hidden(10'240 * 2);
    ninfer::DeviceBuffer norm(10'240 * 2);
    ninfer::DeviceBuffer down(320ULL * 10'240 * 2);
    ninfer::DeviceBuffer up(10'240ULL * 320 * 2);
    ninfer::DeviceBuffer inject(4ULL * 10'240 * 2);
    ninfer::DeviceBuffer block_input(2'560 * 2);
    ninfer::DeviceBuffer block_output(2'560 * 2);
    std::array<std::uint16_t, 10'240> hidden_values{};
    hidden_values.fill(0x3F80U); // 1.0
    hidden.copy_from_host(hidden_values.data(), sizeof(hidden_values));
    norm.fill(0);
    down.fill(0);
    up.fill(0);
    inject.fill(0);
    std::array<std::uint16_t, 2'560> output_values{};
    output_values.fill(0x4000U); // 2.0
    block_output.copy_from_host(output_values.data(), sizeof(output_values));

    HyperConnectionWeights weights{
        .block_inject   = bf16_weight(inject.p, 4, 10'240),
        .norm           = ninfer::Tensor(norm.p, ninfer::DType::BF16, {10'240}),
        .input_mix_down = bf16_weight(down.p, 320, 10'240),
        .input_mix_up   = bf16_weight(up.p, 10'240, 320),
    };
    ninfer::WorkspaceArena workspace(flash_next_hyper_workspace_capacity_bytes(1, 1));
    auto scope                      = workspace.scope();
    FlashNextHyperWorkspace scratch = allocate_flash_next_hyper_workspace(workspace, 1);
    ninfer::Tensor hidden_view(hidden.p, ninfer::DType::BF16, {10'240, 1});
    ninfer::Tensor input_view(block_input.p, ninfer::DType::BF16, {2'560, 1});
    ninfer::Tensor output_view(block_output.p, ninfer::DType::BF16, {2'560, 1});
    flash_next_hyper_prepare(hidden_view, weights, scratch, input_view, device.stream);
    flash_next_hyper_inject(output_view, scratch.injection, hidden_view, device.stream);
    device.synchronize();

    std::array<std::uint16_t, 2'560> actual_input{};
    block_input.copy_to_host(actual_input.data(), sizeof(actual_input));
    if (!std::all_of(actual_input.begin(), actual_input.end(),
                     [](std::uint16_t value) { return value == 0x3F00U; })) {
        std::cerr << "Flash-Next zero-mix hyper input was not exact BF16 0.5\n";
        return 1;
    }
    hidden.copy_to_host(hidden_values.data(), sizeof(hidden_values));
    if (!std::all_of(hidden_values.begin(), hidden_values.end(),
                     [](std::uint16_t value) { return value == 0x4040U; })) {
        std::cerr << "Flash-Next unit injection did not produce exact BF16 3.0\n";
        return 1;
    }

    hidden_values.fill(0x3F80U);
    hidden.copy_from_host(hidden_values.data(), sizeof(hidden_values));
    const HyperMixerWeights mixer{
        .norm           = weights.norm,
        .input_mix_down = weights.input_mix_down,
        .input_mix_up   = weights.input_mix_up,
    };
    flash_next_hyper_mix(hidden_view, mixer, scratch, input_view, device.stream);
    device.synchronize();
    block_input.copy_to_host(actual_input.data(), sizeof(actual_input));
    if (!std::all_of(actual_input.begin(), actual_input.end(),
                     [](std::uint16_t value) { return value == 0x3F00U; })) {
        std::cerr << "Flash-Next final hyper mixer was not exact BF16 0.5\n";
        return 1;
    }
    return 0;
}

int test_synthetic_stage_equivalence(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_weight(-0.05F, 0.05F);
    std::uniform_real_distribution<float> dist_norm(-0.02F, 0.02F);
    std::uniform_real_distribution<float> dist_act(-1.5F, 1.5F);

    // Weights buffers
    std::vector<float> h_norm_f(10'240);
    std::vector<std::uint16_t> h_norm_bf(10'240);
    for (std::size_t i = 0; i < 10'240; ++i) {
        h_norm_f[i]  = dist_norm(rng);
        h_norm_bf[i] = float_to_bf16(h_norm_f[i]);
        h_norm_f[i]  = bf16_to_float(h_norm_bf[i]);
    }

    std::vector<float> h_down_f(320ULL * 10'240);
    std::vector<std::uint16_t> h_down_bf(320ULL * 10'240);
    for (std::size_t i = 0; i < h_down_f.size(); ++i) {
        h_down_f[i]  = dist_weight(rng);
        h_down_bf[i] = float_to_bf16(h_down_f[i]);
        h_down_f[i]  = bf16_to_float(h_down_bf[i]);
    }

    std::vector<float> h_up_f(10'240ULL * 320);
    std::vector<std::uint16_t> h_up_bf(10'240ULL * 320);
    for (std::size_t i = 0; i < h_up_f.size(); ++i) {
        h_up_f[i]  = dist_weight(rng);
        h_up_bf[i] = float_to_bf16(h_up_f[i]);
        h_up_f[i]  = bf16_to_float(h_up_bf[i]);
    }

    std::vector<float> h_inject_f(4ULL * 10'240);
    std::vector<std::uint16_t> h_inject_bf(4ULL * 10'240);
    for (std::size_t i = 0; i < h_inject_f.size(); ++i) {
        h_inject_f[i]  = dist_weight(rng);
        h_inject_bf[i] = float_to_bf16(h_inject_f[i]);
        h_inject_f[i]  = bf16_to_float(h_inject_bf[i]);
    }

    ninfer::DeviceBuffer d_norm(h_norm_bf.size() * 2);
    d_norm.copy_from_host(h_norm_bf.data(), h_norm_bf.size() * 2);

    ninfer::DeviceBuffer d_down(h_down_bf.size() * 2);
    d_down.copy_from_host(h_down_bf.data(), h_down_bf.size() * 2);

    ninfer::DeviceBuffer d_up(h_up_bf.size() * 2);
    d_up.copy_from_host(h_up_bf.data(), h_up_bf.size() * 2);

    ninfer::DeviceBuffer d_inject(h_inject_bf.size() * 2);
    d_inject.copy_from_host(h_inject_bf.data(), h_inject_bf.size() * 2);

    HyperConnectionWeights weights{
        .block_inject   = bf16_weight(d_inject.p, 4, 10'240),
        .norm           = ninfer::Tensor(d_norm.p, ninfer::DType::BF16, {10'240}),
        .input_mix_down = bf16_weight(d_down.p, 320, 10'240),
        .input_mix_up   = bf16_weight(d_up.p, 10'240, 320),
    };

    double max_rel_l2_norm   = 0.0;
    double max_rel_l2_lr     = 0.0;
    double max_rel_l2_inj    = 0.0;
    double max_rel_l2_input  = 0.0;
    double max_rel_l2_injected = 0.0;

    // Decode T=1..8 (fused). Prefill: T=16 one partial 32-col down tile + partial 64-col
    // up tile; T=48 one full down tile + one partial; T=119 production chunk tail
    // (three full 32-col tiles + 23-token partial); T=128 exact down and up tiles.
    std::vector<int> test_tokens = {1, 2, 3, 4, 5, 6, 7, 8, 16, 48, 119, 128};
    constexpr double kMaxRelL2Tolerance = 1.0e-3;

    std::cout << "\n--- Flash-Next Hyper-Connection per-T stage rel-L2 ---\n";
    std::cout << std::scientific << std::setprecision(6);

    for (int tokens : test_tokens) {
        std::vector<float> h_hidden_f(static_cast<std::size_t>(tokens) * 10'240);
        std::vector<std::uint16_t> h_hidden_bf(h_hidden_f.size());
        for (std::size_t i = 0; i < h_hidden_f.size(); ++i) {
            h_hidden_f[i]  = dist_act(rng);
            h_hidden_bf[i] = float_to_bf16(h_hidden_f[i]);
            h_hidden_f[i]  = bf16_to_float(h_hidden_bf[i]);
        }

        std::vector<float> h_output_f(static_cast<std::size_t>(tokens) * 2'560);
        std::vector<std::uint16_t> h_output_bf(h_output_f.size());
        for (std::size_t i = 0; i < h_output_f.size(); ++i) {
            h_output_f[i]  = dist_act(rng);
            h_output_bf[i] = float_to_bf16(h_output_f[i]);
            h_output_f[i]  = bf16_to_float(h_output_bf[i]);
        }

        // Run host reference
        auto ref = evaluate_reference(tokens, h_hidden_f, h_norm_f, h_down_f, h_up_f, h_inject_f, h_output_f);

        // Run GPU fused execution
        ninfer::DeviceBuffer d_hidden(h_hidden_bf.size() * 2);
        d_hidden.copy_from_host(h_hidden_bf.data(), h_hidden_bf.size() * 2);

        ninfer::DeviceBuffer d_block_input(static_cast<std::size_t>(tokens) * 2'560 * 2);
        ninfer::DeviceBuffer d_block_output(h_output_bf.size() * 2);
        d_block_output.copy_from_host(h_output_bf.data(), h_output_bf.size() * 2);

        ninfer::WorkspaceArena workspace(flash_next_hyper_workspace_capacity_bytes(1, tokens));
        auto scope                      = workspace.scope();
        FlashNextHyperWorkspace scratch = allocate_flash_next_hyper_workspace(workspace, tokens);
        CUDA_CHECK(cudaMemsetAsync(scratch.low_rank.data, 0xFF,
                                   static_cast<std::size_t>(tokens) * 320U * 2U, device.stream));
        CUDA_CHECK(cudaMemsetAsync(scratch.up_gemm.data, 0xFF,
                                   static_cast<std::size_t>(tokens) * 10'240U * 2U, device.stream));
        CUDA_CHECK(cudaMemsetAsync(scratch.down_split.data, 0xFF,
                                   static_cast<std::size_t>(tokens) * 320U * 4U * sizeof(float),
                                   device.stream));
        CUDA_CHECK(cudaMemsetAsync(d_block_input.p, 0xFF,
                                   static_cast<std::size_t>(tokens) * 2'560U * 2U, device.stream));

        ninfer::Tensor hidden_view(d_hidden.p, ninfer::DType::BF16, {10'240, tokens});
        ninfer::Tensor input_view(d_block_input.p, ninfer::DType::BF16, {2'560, tokens});
        ninfer::Tensor output_view(d_block_output.p, ninfer::DType::BF16, {2'560, tokens});

        flash_next_hyper_prepare(hidden_view, weights, scratch, input_view, device.stream);
        flash_next_hyper_inject(output_view, scratch.injection, hidden_view, device.stream);
        device.synchronize();

        auto to_float = [](const std::vector<std::uint16_t>& bits) {
            std::vector<float> out(bits.size());
            for (std::size_t i = 0; i < bits.size(); ++i) {
                out[i] = bf16_to_float(bits[i]);
            }
            return out;
        };

        std::vector<std::uint16_t> act_norm_bf(static_cast<std::size_t>(tokens) * 10'240);
        CUDA_CHECK(cudaMemcpy(act_norm_bf.data(), scratch.normalized.data, act_norm_bf.size() * 2,
                              cudaMemcpyDeviceToHost));
        const std::vector<float> act_norm_f = to_float(act_norm_bf);
        const RelL2Stats err_norm =
            rel_l2_stats(act_norm_f.data(), ref.normalized.data(), act_norm_f.size());

        std::vector<std::uint16_t> act_lr_bf(static_cast<std::size_t>(tokens) * 320);
        CUDA_CHECK(cudaMemcpy(act_lr_bf.data(), scratch.low_rank.data, act_lr_bf.size() * 2,
                              cudaMemcpyDeviceToHost));
        const std::vector<float> act_lr_f = to_float(act_lr_bf);
        const RelL2Stats err_lr =
            rel_l2_stats(act_lr_f.data(), ref.low_rank.data(), act_lr_f.size());

        std::vector<float> act_inj_f(static_cast<std::size_t>(tokens) * 4);
        CUDA_CHECK(cudaMemcpy(act_inj_f.data(), scratch.injection.data,
                              act_inj_f.size() * sizeof(float), cudaMemcpyDeviceToHost));
        const RelL2Stats err_inj =
            rel_l2_stats(act_inj_f.data(), ref.injection.data(), act_inj_f.size());

        std::vector<std::uint16_t> act_in_bf(static_cast<std::size_t>(tokens) * 2'560);
        CUDA_CHECK(cudaMemcpy(act_in_bf.data(), input_view.data, act_in_bf.size() * 2,
                              cudaMemcpyDeviceToHost));
        std::vector<float> act_in_f = to_float(act_in_bf);
        const RelL2Stats err_in =
            rel_l2_stats(act_in_f.data(), ref.block_input.data(), act_in_f.size());

        std::vector<std::uint16_t> act_hid_bf(static_cast<std::size_t>(tokens) * 10'240);
        CUDA_CHECK(cudaMemcpy(act_hid_bf.data(), hidden_view.data, act_hid_bf.size() * 2,
                              cudaMemcpyDeviceToHost));
        const std::vector<float> act_hid_f = to_float(act_hid_bf);
        const RelL2Stats err_hid =
            rel_l2_stats(act_hid_f.data(), ref.hidden_injected.data(), act_hid_f.size());

        const HyperMixerWeights mixer{
            .norm           = weights.norm,
            .input_mix_down = weights.input_mix_down,
            .input_mix_up   = weights.input_mix_up,
        };
        flash_next_hyper_mix(hidden_view, mixer, scratch, input_view, device.stream);
        device.synchronize();
        CUDA_CHECK(cudaMemcpy(act_in_bf.data(), input_view.data, act_in_bf.size() * 2,
                              cudaMemcpyDeviceToHost));
        act_in_f = to_float(act_in_bf);
        auto ref_mix = evaluate_reference(tokens, act_hid_f, h_norm_f, h_down_f, h_up_f, h_inject_f,
                                          h_output_f);
        const RelL2Stats err_mix =
            rel_l2_stats(act_in_f.data(), ref_mix.block_input.data(), act_in_f.size());

        std::cout << "  T=" << tokens << " down%32=" << (tokens % 32) << " up%64=" << (tokens % 64)
                  << " norm=" << err_norm.rel << " down=" << err_lr.rel << " inj=" << err_inj.rel
                  << " input=" << err_in.rel << " hid=" << err_hid.rel << " mix=" << err_mix.rel
                  << " down_base_sq=" << err_lr.base_sq << "\n";

        if (!accept_stage("group_norm", tokens, err_norm, &max_rel_l2_norm, kMaxRelL2Tolerance) ||
            !accept_stage("low_rank", tokens, err_lr, &max_rel_l2_lr, kMaxRelL2Tolerance) ||
            !accept_stage("injection", tokens, err_inj, &max_rel_l2_inj, kMaxRelL2Tolerance) ||
            !accept_stage("block_input", tokens, err_in, &max_rel_l2_input, kMaxRelL2Tolerance) ||
            !accept_stage("hidden_injected", tokens, err_hid, &max_rel_l2_injected,
                          kMaxRelL2Tolerance) ||
            !accept_stage("mixer_block_input", tokens, err_mix, &max_rel_l2_input,
                          kMaxRelL2Tolerance)) {
            return 1;
        }

        if (tokens == 48) {
            const RelL2Stats full_tile =
                rel_l2_stats(act_lr_f.data(), ref.low_rank.data(), 32ULL * 320ULL);
            const RelL2Stats partial_tile =
                rel_l2_stats(act_lr_f.data() + 32ULL * 320ULL, ref.low_rank.data() + 32ULL * 320ULL,
                             16ULL * 320ULL);
            print_rel_l2_stats("T=48 low_rank tokens[0,32) full 32-tile", full_tile);
            print_rel_l2_stats("T=48 low_rank tokens[32,48) partial 16-tile", partial_tile);
            if (!accept_stage("low_rank_full_tile", tokens, full_tile, &max_rel_l2_lr,
                              kMaxRelL2Tolerance) ||
                !accept_stage("low_rank_partial_tile", tokens, partial_tile, &max_rel_l2_lr,
                              kMaxRelL2Tolerance)) {
                return 1;
            }
        }
    }

    std::cout << "\n--- Flash-Next Fused Hyper-Connection vs Reference (rel-L2 Maxima) ---\n";
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  Stage 1 (group_norm -> normalized) : " << max_rel_l2_norm << "\n";
    std::cout << "  Stage 2 (down_proj  -> low_rank)   : " << max_rel_l2_lr << "\n";
    std::cout << "  Stage 3 (inject_gate-> injection)  : " << max_rel_l2_inj << "\n";
    std::cout << "  Stage 4 (up_proj+mix-> block_input): " << max_rel_l2_input << "\n";
    std::cout << "  Stage 5 (inject     -> hidden)     : " << max_rel_l2_injected << "\n";
    std::cout << "----------------------------------------------------------------------\n";

    if (max_rel_l2_norm > kMaxRelL2Tolerance || max_rel_l2_lr > kMaxRelL2Tolerance ||
        max_rel_l2_inj > kMaxRelL2Tolerance || max_rel_l2_input > kMaxRelL2Tolerance ||
        max_rel_l2_injected > kMaxRelL2Tolerance) {
        std::cerr << "FAILED: Relative L2 error exceeded tolerance\n";
        return 1;
    }

    std::cout << "PASS: test_synthetic_stage_equivalence\n";
    return 0;
}

int test_kernel_timing_benchmark(ninfer::DeviceContext& device) {
    using namespace ninfer::targets::qwen3_8_flash_next::detail;

    ninfer::DeviceBuffer d_norm(10'240 * 2);
    ninfer::DeviceBuffer d_down(320ULL * 10'240 * 2);
    ninfer::DeviceBuffer d_up(10'240ULL * 320 * 2);
    ninfer::DeviceBuffer d_inject(4ULL * 10'240 * 2);
    d_norm.fill(0);
    d_down.fill(0);
    d_up.fill(0);
    d_inject.fill(0);

    HyperConnectionWeights weights{
        .block_inject   = bf16_weight(d_inject.p, 4, 10'240),
        .norm           = ninfer::Tensor(d_norm.p, ninfer::DType::BF16, {10'240}),
        .input_mix_down = bf16_weight(d_down.p, 320, 10'240),
        .input_mix_up   = bf16_weight(d_up.p, 10'240, 320),
    };

    std::cout << "\n=== Flash-Next Hyper-Connection Kernel Timing Breakdown ===\n";
    std::cout << std::fixed << std::setprecision(2);

    for (int tokens : {1, 16, 128, 2048}) {
        ninfer::DeviceBuffer d_hidden(static_cast<std::size_t>(tokens) * 10'240 * 2);
        ninfer::DeviceBuffer d_block_input(static_cast<std::size_t>(tokens) * 2'560 * 2);
        ninfer::DeviceBuffer d_block_output(static_cast<std::size_t>(tokens) * 2'560 * 2);
        d_hidden.fill(0);
        d_block_input.fill(0);
        d_block_output.fill(0);

        ninfer::WorkspaceArena workspace(flash_next_hyper_workspace_capacity_bytes(1, tokens));
        auto scope                      = workspace.scope();
        FlashNextHyperWorkspace scratch = allocate_flash_next_hyper_workspace(workspace, tokens);

        ninfer::Tensor hidden_view(d_hidden.p, ninfer::DType::BF16, {10'240, tokens});
        ninfer::Tensor input_view(d_block_input.p, ninfer::DType::BF16, {2'560, tokens});
        ninfer::Tensor output_view(d_block_output.p, ninfer::DType::BF16, {2'560, tokens});

        // Warmup
        for (int i = 0; i < 20; ++i) {
            flash_next_hyper_prepare(hidden_view, weights, scratch, input_view, device.stream);
            flash_next_hyper_inject(output_view, scratch.injection, hidden_view, device.stream);
        }
        device.synchronize();

        cudaEvent_t start_event, stop_event;
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));

        constexpr int kIters = 500;

        // Measure prepare stream time
        CUDA_CHECK(cudaEventRecord(start_event, device.stream));
        for (int i = 0; i < kIters; ++i) {
            flash_next_hyper_prepare(hidden_view, weights, scratch, input_view, device.stream);
        }
        CUDA_CHECK(cudaEventRecord(stop_event, device.stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float total_prep_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&total_prep_ms, start_event, stop_event));

        // Measure inject stream time
        CUDA_CHECK(cudaEventRecord(start_event, device.stream));
        for (int i = 0; i < kIters; ++i) {
            flash_next_hyper_inject(output_view, scratch.injection, hidden_view, device.stream);
        }
        CUDA_CHECK(cudaEventRecord(stop_event, device.stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float total_inj_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&total_inj_ms, start_event, stop_event));

        // Measure full chain stream time (Eager)
        CUDA_CHECK(cudaEventRecord(start_event, device.stream));
        for (int i = 0; i < kIters; ++i) {
            flash_next_hyper_prepare(hidden_view, weights, scratch, input_view, device.stream);
            flash_next_hyper_inject(output_view, scratch.injection, hidden_view, device.stream);
        }
        CUDA_CHECK(cudaEventRecord(stop_event, device.stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float total_chain_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&total_chain_ms, start_event, stop_event));

        // Measure full chain stream time (CUDA Graph Replay)
        cudaGraph_t graph;
        cudaGraphExec_t graph_exec;
        CUDA_CHECK(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeGlobal));
        flash_next_hyper_prepare(hidden_view, weights, scratch, input_view, device.stream);
        flash_next_hyper_inject(output_view, scratch.injection, hidden_view, device.stream);
        CUDA_CHECK(cudaStreamEndCapture(device.stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

        CUDA_CHECK(cudaEventRecord(start_event, device.stream));
        for (int i = 0; i < kIters; ++i) {
            CUDA_CHECK(cudaGraphLaunch(graph_exec, device.stream));
        }
        CUDA_CHECK(cudaEventRecord(stop_event, device.stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float total_graph_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&total_graph_ms, start_event, stop_event));

        CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
        CUDA_CHECK(cudaGraphDestroy(graph));

        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));

        const float avg_prep_us   = (total_prep_ms / kIters) * 1000.0F;
        const float avg_inj_us    = (total_inj_ms / kIters) * 1000.0F;
        const float avg_eager_us  = (total_chain_ms / kIters) * 1000.0F;
        const float avg_graph_us  = (total_graph_ms / kIters) * 1000.0F;

        std::cout << "  Tokens T=" << tokens << ":\n";
        std::cout << "    Eager Prepare Launch Stream : " << avg_prep_us << " us\n";
        std::cout << "    Eager Inject Launch Stream  : " << avg_inj_us << " us\n";
        std::cout << "    Eager Chain Stream Total    : " << avg_eager_us << " us\n";
        std::cout << "    CUDA Graph Hardware Replay  : " << avg_graph_us << " us\n";

        if (tokens == 1 && avg_graph_us > 25.0F) {
            std::cerr << "FAILED: T=1 graph chain time (" << avg_graph_us << " us) exceeded 25.0 us threshold!\n";
            return 1;
        }
    }
    std::cout << "===========================================================\n";
    return 0;
}

} // namespace

int main() {
    int device_count              = 0;
    const cudaError_t count_error = cudaGetDeviceCount(&device_count);
    if (cuda_unavailable(count_error) || device_count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_error);

    ninfer::DeviceContext device(0);

    if (test_basic_unit_injection(device) != 0) { return 1; }
    if (test_synthetic_stage_equivalence(device) != 0) { return 1; }
    if (test_kernel_timing_benchmark(device) != 0) { return 1; }

    std::cout << "OK Flash-Next Hyper Connection\n";
    return 0;
}
