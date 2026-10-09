#include "core/weight.h"
#include "core/device.h"
#include "core/decode_graph.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/weight_input.h"

#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

// This criterion belongs to the complete A16 GDN-input-projection Op.
constexpr ReductionCriterion kGdnInputProjA16Tolerance{3.0e-3, 4.0e-3, 3.5e-3};
constexpr ReductionCriterion kFp8GdnInputProjA16Tolerance{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
constexpr ReductionCriterion kFp8GdnInputProjA8Tolerance{0.04, 1.0 / 256.0, 0.06};
constexpr ReductionCriterion kGdnInputProjA4Tolerance{0.16, 4.0e-3, 0.16};
constexpr std::int32_t kA8SampleRows = 31;

int verify_output_range(std::string_view label, const GuardedBf16Tensor& output,
                        std::int32_t full_rows, std::int32_t output_row_offset,
                        std::int32_t output_rows, const quantized_weight::PackedWeight& weight,
                        std::int32_t weight_row_offset, const std::vector<float>& activation,
                        std::int32_t hidden, std::int32_t tokens) {
    const std::vector<double> actual =
        gather_rows(output.values(), full_rows, output_row_offset, output_rows, tokens);
    const std::vector<double> expected =
        projection_oracle(weight, weight_row_offset, output_rows, activation, hidden, tokens);
    return compare(label, actual, expected, kGdnInputProjA16Tolerance);
}

int run_q4_q5_case(DevicePackedWeight& query_key, DevicePackedWeight& value_z_weight,
                   std::int32_t tokens) {
    constexpr std::int32_t kHidden      = 5120;
    constexpr std::int32_t kQkRows      = 4096;
    constexpr std::int32_t kValueRows   = 6144;
    constexpr std::int32_t kZRows       = 6144;
    constexpr std::int32_t kRows        = kQkRows + kValueRows;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 401U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor output   = qkv.tensor();
    Tensor z_output = z.tensor();
    ops::gdn_input_proj(x, query_key.view(), value_z_weight.view(), output, z_output, nullptr);
    cuda_synchronize();

    const std::string suffix = " Q4/Q5 A16 T=" + std::to_string(tokens);
    int failures             = qkv.verify_guards("gdn qkv" + suffix);
    failures += z.verify_guards("gdn z" + suffix);
    failures += qkv.verify_fully_written("gdn qkv" + suffix);
    failures += z.verify_fully_written("gdn z" + suffix);
    failures += verify_output_range("gdn qk" + suffix, qkv, kRows, 0, kQkRows, query_key.host, 0,
                                    activation, kHidden, tokens);
    failures += verify_output_range("gdn value" + suffix, qkv, kRows, kQkRows, kValueRows,
                                    value_z_weight.host, 0, activation, kHidden, tokens);
    failures += verify_output_range("gdn z" + suffix, z, kZRows, 0, kZRows, value_z_weight.host,
                                    kValueRows, activation, kHidden, tokens);
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += query_key.verify_preserved("gdn query/key weight" + suffix);
    failures += value_z_weight.verify_preserved("gdn value/z weight" + suffix);
    return failures;
}

int run_q4_q5_graph_case(DevicePackedWeight& query_key, DevicePackedWeight& value_z_weight,
                         std::int32_t tokens) {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kQkRows    = 4096;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    constexpr std::int32_t kRows      = kQkRows + kValueRows;

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<float> activation = make_bf16_activation(kHidden, tokens, 701U + tokens);
    std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation             = to_device(activation_bits);
    GuardedBf16Tensor qkv(kRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor output     = qkv.tensor();
    Tensor z_output   = z.tensor();
    const auto launch = [&](cudaStream_t launch_stream) {
        ops::gdn_input_proj(x, query_key.view(), value_z_weight.view(), output, z_output,
                            launch_stream);
    };
    launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaGraph_t graph          = nullptr;
    cudaGraphExec_t executable = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));

    // Replay 1, checked on its own: the outputs are poisoned first, so a replay that skipped a
    // range or wrote to a stale address is caught here rather than being masked by replay 2.
    const auto verify_replay = [&](const std::vector<float>& expected, std::string_view tag) {
        int bad = qkv.verify_guards(std::string("gdn qkv") + std::string(tag));
        bad += z.verify_guards(std::string("gdn z") + std::string(tag));
        bad += qkv.verify_fully_written(std::string("gdn qkv") + std::string(tag));
        bad += z.verify_fully_written(std::string("gdn z") + std::string(tag));
        bad += verify_output_range(std::string("gdn qk") + std::string(tag), qkv, kRows, 0, kQkRows,
                                   query_key.host, 0, expected, kHidden, tokens);
        bad += verify_output_range(std::string("gdn value") + std::string(tag), qkv, kRows, kQkRows,
                                   kValueRows, value_z_weight.host, 0, expected, kHidden, tokens);
        bad += verify_output_range(std::string("gdn z") + std::string(tag), z, kZRows, 0, kZRows,
                                   value_z_weight.host, kValueRows, expected, kHidden, tokens);
        return bad;
    };
    const std::string suffix = " Q4/Q5 A16 graph T=" + std::to_string(tokens);
    qkv.repaint(stream);
    z.repaint(stream);
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int failures = verify_replay(activation, suffix + " replay1");

    // Replay 2 against a changed activation at the same captured address: a graph that baked its
    // operands would keep reporting the first input here.
    qkv.repaint(stream);
    z.repaint(stream);
    activation      = make_bf16_activation(kHidden, tokens, 811U + tokens);
    activation_bits = bf16_bits(activation);
    CUDA_CHECK(cudaMemcpyAsync(device_activation.p, activation_bits.data(),
                               activation_bits.size() * sizeof(std::uint16_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaGraphLaunch(executable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));

    failures += verify_replay(activation, suffix + " replay2");
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += query_key.verify_preserved("gdn query/key weight" + suffix);
    failures += value_z_weight.verify_preserved("gdn value/z weight" + suffix);
    return failures;
}

int run_q4_q5() {
    constexpr std::int32_t kHidden = 5120;
    DevicePackedWeight query_key(
        quantized_weight::make_patterned_weight(QType::Q4_G64_FP16, 4096, kHidden, 409U));
    DevicePackedWeight value_z_weight(
        quantized_weight::make_patterned_weight(QType::Q5_G64_FP16, 12288, kHidden, 419U));
    int failures = 0;
    // Every route boundary and both of its neighbours: the Q4/Q5 column catalog hands 1..12 to the
    // per-side small-column kernels (the K-split Q4 parent with the split4 Q5 side at 2..10 and the
    // c4 SIMT Q5 side at 11..12), 13..32 to the 32x32 tile, 33..64 to the 32x64 tile, and 65 upward
    // to the 64x128 tile that also supplies the 128-column tail slices. The Q4 K-split band (7..12)
    // is not changed by this work; its two ends are covered by the same list.
    for (const std::int32_t tokens : {1,  2,  3,  4,  5,  6,  7,  8,  9,   10,  11,  12,
                                      13, 15, 16, 17, 32, 33, 64, 65, 127, 128, 129, 193}) {
        failures += run_q4_q5_case(query_key, value_z_weight, tokens);
    }
    // One captured replay per route, including both ends of the split4 band and a 128-column tail
    // slice (129 = 128 + 1).
    for (const std::int32_t tokens : {7, 8, 9, 10, 12, 13, 33, 65, 129}) {
        failures += run_q4_q5_graph_case(query_key, value_z_weight, tokens);
    }
    return failures;
}

int run_q8_case(DevicePackedWeight& parent, std::int32_t tokens) {
    constexpr std::int32_t kHidden      = 2048;
    constexpr std::int32_t kQkvRows     = 8192;
    constexpr std::int32_t kZRows       = 4096;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 501U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kQkvRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor qkv_output = qkv.tensor();
    Tensor z_output   = z.tensor();
    ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, nullptr);
    cuda_synchronize();

    const std::string suffix = " Q8 A16 T=" + std::to_string(tokens);
    int failures             = qkv.verify_guards("gdn qkv" + suffix);
    failures += z.verify_guards("gdn z" + suffix);
    failures += qkv.verify_fully_written("gdn qkv" + suffix);
    failures += z.verify_fully_written("gdn z" + suffix);
    failures += verify_output_range("gdn qkv" + suffix, qkv, kQkvRows, 0, kQkvRows, parent.host, 0,
                                    activation, kHidden, tokens);
    failures += verify_output_range("gdn z" + suffix, z, kZRows, 0, kZRows, parent.host, kQkvRows,
                                    activation, kHidden, tokens);
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("gdn parent weight" + suffix);
    return failures;
}

int run_q8() {
    constexpr std::int32_t kHidden = 2048;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, 12288, kHidden, 503U));
    int failures = 0;
    for (const std::int32_t tokens : {1, 2, 97}) { failures += run_q8_case(parent, tokens); }
    return failures;
}

int verify_output_range_sampled(std::string_view label, const GuardedBf16Tensor& output,
                                std::int32_t full_rows, std::int32_t output_row_offset,
                                std::int32_t output_rows,
                                const quantized_weight::PackedWeight& weight,
                                std::int32_t weight_row_offset,
                                const std::vector<float>& activation, std::int32_t hidden,
                                std::int32_t tokens, const ReductionCriterion& criterion,
                                std::int32_t sample_count = 7) {
    const std::vector<double> values     = output.values();
    const std::vector<std::int32_t> rows = sampled_rows(output_rows, sample_count);
    std::vector<std::int32_t> selected_tokens;
    for (const std::int32_t token :
         {0, 1, tokens / 4, tokens / 2, (3 * tokens) / 4, tokens - 2, tokens - 1}) {
        if (token >= 0 && token < tokens &&
            std::find(selected_tokens.begin(), selected_tokens.end(), token) ==
                selected_tokens.end()) {
            selected_tokens.push_back(token);
        }
    }
    std::vector<double> actual;
    std::vector<double> expected;
    actual.reserve(rows.size() * selected_tokens.size());
    expected.reserve(rows.size() * selected_tokens.size());
    for (const std::int32_t local_row : rows) {
        const std::int32_t output_row = output_row_offset + local_row;
        const std::int32_t weight_row = weight_row_offset + local_row;
        for (const std::int32_t token : selected_tokens) {
            actual.push_back(values[static_cast<std::size_t>(token) * full_rows + output_row]);
            expected.push_back(quantized_weight::dot_fp64(
                weight, weight_row, activation.data() + static_cast<std::size_t>(token) * hidden,
                hidden));
        }
    }
    return compare(label, actual, expected, criterion);
}

int run_nvfp4_case(DevicePackedWeight& parent, std::int32_t tokens, ops::LinearPolicy policy) {
    constexpr std::int32_t kHidden      = 5120;
    constexpr std::int32_t kQkvRows     = 10240;
    constexpr std::int32_t kZRows       = 6144;
    constexpr std::int32_t kRows        = kQkvRows + kZRows;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 601U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kQkvRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x                   = Tensor(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor qkv_output          = qkv.tensor();
    Tensor z_output            = z.tensor();
    const std::size_t capacity = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::NVFP4, kRows, kHidden, policy, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, policy, workspace, nullptr);
    cuda_synchronize();

    const bool a4                       = policy == ops::LinearPolicy::AllowA4;
    const ReductionCriterion& criterion = a4 ? kGdnInputProjA4Tolerance : kGdnInputProjA16Tolerance;
    const std::string suffix =
        std::string(" NVFP4 ") + (a4 ? "A4" : "A16") + " T=" + std::to_string(tokens);
    int failures = qkv.verify_guards("gdn qkv" + suffix);
    failures += z.verify_guards("gdn z" + suffix);
    failures += qkv.verify_fully_written("gdn qkv" + suffix);
    failures += z.verify_fully_written("gdn z" + suffix);
    failures += verify_output_range_sampled("gdn query" + suffix, qkv, kQkvRows, 0, 2048,
                                            parent.host, 0, activation, kHidden, tokens, criterion);
    failures +=
        verify_output_range_sampled("gdn key" + suffix, qkv, kQkvRows, 2048, 2048, parent.host,
                                    2048, activation, kHidden, tokens, criterion);
    failures +=
        verify_output_range_sampled("gdn value" + suffix, qkv, kQkvRows, 4096, 6144, parent.host,
                                    4096, activation, kHidden, tokens, criterion);
    failures += verify_output_range_sampled("gdn z" + suffix, z, kZRows, 0, kZRows, parent.host,
                                            kQkvRows, activation, kHidden, tokens, criterion);
    if (workspace.peak_used() != capacity) {
        std::cerr << "gdn workspace" << suffix << ": query/execution high-water mismatch\n";
        ++failures;
    }
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("gdn parent weight" + suffix);
    return failures;
}

int run_nvfp4() {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kRows   = 16384;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::NVFP4, kRows, kHidden, 607U, options));
    int failures = 0;
    failures += run_nvfp4_case(parent, 1, ops::LinearPolicy::A16Only);
    failures += run_nvfp4_case(parent, 4, ops::LinearPolicy::AllowA8);
    failures += run_nvfp4_case(parent, 1, ops::LinearPolicy::AllowA4);
    failures += run_nvfp4_case(parent, 2, ops::LinearPolicy::AllowA4);
    failures += run_nvfp4_case(parent, 17, ops::LinearPolicy::AllowA4);
    // 1023, 1024 and 1025 straddle this route's floor. 1024 was the narrowest width it
    // took before; 1025 is the first ragged one it takes now, and its last M tile holds a
    // single real token, which is the emptiest grid this route ever runs.
    failures += run_nvfp4_case(parent, 1023, ops::LinearPolicy::AllowA4);
    failures += run_nvfp4_case(parent, 1024, ops::LinearPolicy::AllowA4);
    failures += run_nvfp4_case(parent, 1025, ops::LinearPolicy::AllowA4);
    return failures;
}

// The registered row-scaled FP8 [16384,K] parents share the route frontier and criteria.
struct Fp8Parent {
    QType qtype;
    std::int32_t hidden;
    const char* label;
};

constexpr Fp8Parent kFp8Bf16K5120{QType::FP8_E4M3FN_ROW_BF16, 5120, "FP8"};
constexpr Fp8Parent kFp8Fp32K2560{QType::FP8_E4M3FN_ROW_FP32, 2560, "FP8-FP32 K2560"};

int run_fp8_case(const Fp8Parent& profile, DevicePackedWeight& parent, std::int32_t tokens,
                 ops::LinearPolicy policy, bool convenience = false, bool replay = false) {
    const std::int32_t kHidden      = profile.hidden;
    constexpr std::int32_t kQkvRows = 10240;
    constexpr std::int32_t kZRows   = 6144;
    constexpr std::int32_t kRows    = kQkvRows + kZRows;
    std::vector<float> activation =
        make_bf16_activation(kHidden, tokens, 617U + static_cast<std::uint32_t>(tokens));
    std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation             = to_device(activation_bits);
    GuardedBf16Tensor qkv(kQkvRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x                   = Tensor(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor qkv_output          = qkv.tensor();
    Tensor z_output            = z.tensor();
    const std::size_t capacity = ops::gdn_input_proj_workspace_capacity_bytes(
        profile.qtype, kRows, kHidden, policy, tokens, tokens);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 256));
    WorkspaceArena workspace(DeviceSpan{scratch.data(), std::max<std::size_t>(capacity, 256)});
    DeviceContext context;
    const auto launch = [&] {
        if (convenience)
            ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, context.stream);
        else
            ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, policy, workspace,
                                context.stream);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    cuda_synchronize();
    if (replay) {
        definition.capture(context.stream, launch);
        graph.instantiate(definition);
    }
    int failures = 0;
    for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
        if (phase) {
            for (auto& value : activation) value = -value;
            activation_bits = bf16_bits(activation);
            device_activation.copy_from_host(activation_bits.data(), activation_bits.size() * 2);
        }
        scratch.fill(phase ? 0xa5 : 0x5a);
        cuda_synchronize();
        CUDA_CHECK(cudaMemsetAsync(qkv_output.data, 0xff, std::size_t(kQkvRows) * tokens * 2,
                                   context.stream));
        CUDA_CHECK(
            cudaMemsetAsync(z_output.data, 0xff, std::size_t(kZRows) * tokens * 2, context.stream));
        if (replay)
            graph.launch(context.stream);
        else
            launch();
        cuda_synchronize(context.stream);
        const bool a8 =
            (policy == ops::LinearPolicy::AllowA8 || policy == ops::LinearPolicy::AllowA4) &&
            tokens >= 17;
        const ReductionCriterion& criterion =
            a8 ? kFp8GdnInputProjA8Tolerance : kFp8GdnInputProjA16Tolerance;
        const std::int32_t sample_count = a8 ? kA8SampleRows : 7;
        const std::string suffix = std::string(" ") + profile.label + (a8 ? " A8" : " A16") +
                                   " T=" + std::to_string(tokens);
        failures += qkv.verify_guards("gdn qkv" + suffix);
        failures += z.verify_guards("gdn z" + suffix);
        failures += qkv.verify_fully_written("gdn qkv" + suffix);
        failures += z.verify_fully_written("gdn z" + suffix);
        failures +=
            verify_output_range_sampled("gdn query" + suffix, qkv, kQkvRows, 0, 2048, parent.host,
                                        0, activation, kHidden, tokens, criterion, sample_count);
        failures +=
            verify_output_range_sampled("gdn key" + suffix, qkv, kQkvRows, 2048, 2048, parent.host,
                                        2048, activation, kHidden, tokens, criterion, sample_count);
        failures += verify_output_range_sampled("gdn value" + suffix, qkv, kQkvRows, 4096, 6144,
                                                parent.host, 4096, activation, kHidden, tokens,
                                                criterion, sample_count);
        failures += verify_output_range_sampled("gdn z" + suffix, z, kZRows, 0, kZRows, parent.host,
                                                kQkvRows, activation, kHidden, tokens, criterion,
                                                sample_count);
        if (workspace.peak_used() != capacity) {
            std::cerr << "gdn workspace" << suffix << ": query/execution high-water mismatch\n";
            ++failures;
        }
        failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
        if (!replay && !convenience && profile.qtype == QType::FP8_E4M3FN_ROW_BF16 &&
            tokens == 128 && policy == ops::LinearPolicy::AllowA8) {
            // Keep both identical weights resident but alternate their addresses. Reusing one
            // hot weight hid intermittent first-token corruption during real-model prefill.
            // Check every represented output, including both sides of the token-tile boundary.
            DeviceBuffer alternate(parent.host.payload.size());
            alternate.copy_from_host(parent.host.payload.data(), parent.host.payload.size());
            const auto alternate_weight = parent.host.device_weight(alternate.p);
            const auto expected_qkv     = qkv.bits();
            const auto expected_z       = z.bits();
            for (int iteration = 0; iteration < 4096; ++iteration) {
                const auto weight = iteration % 2 ? parent.view() : alternate_weight;
                ops::gdn_input_proj(x, weight, qkv_output, z_output, policy, workspace,
                                    context.stream);
                cuda_synchronize(context.stream);
                if (qkv.bits() != expected_qkv || z.bits() != expected_z) {
                    std::cerr << "gdn FP8 repeated weight switch changed the result at iteration "
                              << iteration << '\n';
                    ++failures;
                    break;
                }
            }
            std::vector<std::uint8_t> after(parent.host.payload.size());
            alternate.copy_to_host(after.data(), after.size());
            if (after != parent.host.payload) {
                std::cerr << "gdn alternate weight" << suffix << ": packed weight was modified\n";
                ++failures;
            }
        }
        failures += parent.verify_preserved("gdn parent weight" + suffix);
        failures += scratch.verify_guards(suffix);
    }
    return failures;
}

int run_fp8() {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kRows   = 16384;
    const Fp8Parent& profile       = kFp8Bf16K5120;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, 613U));

    int failures = 0;
    for (int columns : {5, 8, 16, 24, 32, 33, 64, 65, 96, 97, 128, 129}) {
        failures += run_fp8_case(profile, parent, columns, ops::LinearPolicy::A16Only);
    }
    const std::size_t one = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::AllowA8, 1, 1);
    const std::size_t sixteen = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::AllowA8, 16, 16);
    const std::size_t seventeen = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::AllowA8, 17, 17);
    const std::size_t forty_eight = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::AllowA8, 48, 48);
    const std::size_t hot_interval = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::AllowA8, 1, 48);
    const std::size_t exact_1024 = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::AllowA8, 1024, 1024);
    const std::size_t a16 = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, ops::LinearPolicy::A16Only, 1, 2048);
    if (one != 0 || sixteen != 0 || seventeen == 0 || forty_eight <= seventeen ||
        hot_interval != forty_eight || exact_1024 <= forty_eight || a16 != 0) {
        std::cerr << "FP8 gdn input workspace interval contract mismatch\n";
        ++failures;
    }

    failures += run_fp8_case(profile, parent, 1, ops::LinearPolicy::A16Only, true);
    failures += run_fp8_case(profile, parent, 2, ops::LinearPolicy::A16Only);
    for (const std::int32_t tokens :
         {1,   2,   4,   8,   16,  17,  18,  32,  33,  48,  64,  65,  127,  128, 129,
          191, 192, 193, 255, 256, 257, 383, 384, 385, 511, 512, 513, 1024, 1025}) {
        failures += run_fp8_case(profile, parent, tokens,
                                 tokens == 17 ? ops::LinearPolicy::AllowA4
                                              : ops::LinearPolicy::AllowA8);
    }
    for (int tokens : {4, 17, 128, 385, 512, 1025})
        failures += run_fp8_case(profile, parent, tokens, ops::LinearPolicy::AllowA8, false, true);
    return failures;
}

// The Flash-Next q/k/value/z rows of one complete RowScaleFp32 parent resolve through the public
// weight-input entry to the parent's own native operand.
int run_fp8_flash_next_prepared(const DevicePackedWeight& parent) {
    constexpr std::uint64_t kHidden = 2560;
    const std::vector<std::uint64_t> parent_shape{16384, kHidden};
    WeightParent weight_parent;
    weight_parent.geometry =
        weight_geometry(QType::FP8_E4M3FN_ROW_FP32, QuantLayout::RowScaleFp32, parent_shape);
    weight_parent.data = static_cast<const std::byte*>(parent.device.p);
    const auto region  = [&](std::uint64_t row_begin, std::uint64_t rows) {
        return WeightView{{rows, kHidden},
                          {WeightRegion{&weight_parent, row_begin * kHidden,
                                        (row_begin + rows) * kHidden}}};
    };
    const WeightView query = region(0, 2048);
    const WeightView key   = region(2048, 2048);
    const WeightView value = region(4096, 6144);
    const WeightView z     = region(10240, 6144);
    const auto prepared    = ops::prepare_gdn_input_proj_weights(
        {query, ops::LinearPolicy::AllowA8}, {key, ops::LinearPolicy::AllowA8},
        {value, ops::LinearPolicy::AllowA8}, {z, ops::LinearPolicy::AllowA8});
    const auto* single = std::get_if<ops::SingleProjectionWeight>(&prepared);
    const Weight expected = parent.view();
    if (single == nullptr || single->policy != ops::LinearPolicy::AllowA8 ||
        single->weight.qtype != expected.qtype || single->weight.layout != expected.layout ||
        single->weight.scale_dtype != expected.scale_dtype || single->weight.n != expected.n ||
        single->weight.k != expected.k || single->weight.qdata != expected.qdata ||
        single->weight.scales != expected.scales) {
        std::cerr << "FP8-FP32 K2560 GDN parent did not prepare to its native operand\n";
        return 1;
    }
    return 0;
}

int run_fp8_flash_next() {
    constexpr std::int32_t kHidden = 2560;
    constexpr std::int32_t kRows   = 16384;
    const Fp8Parent& profile       = kFp8Fp32K2560;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_FP32, kRows, kHidden, 619U));

    int failures = run_fp8_flash_next_prepared(parent);
    const auto capacity = [&](ops::LinearPolicy policy, std::int32_t first, std::int32_t last) {
        return ops::gdn_input_proj_workspace_capacity_bytes(profile.qtype, kRows, kHidden, policy,
                                                            first, last);
    };
    const std::size_t a8_17   = capacity(ops::LinearPolicy::AllowA8, 17, 17);
    const std::size_t a8_257  = capacity(ops::LinearPolicy::AllowA8, 257, 257);
    const std::size_t a8_8192 = capacity(ops::LinearPolicy::AllowA8, 8192, 8192);
    if (capacity(ops::LinearPolicy::AllowA8, 1, 16) != 0 || a8_17 == 0 ||
        capacity(ops::LinearPolicy::AllowA8, 1, 17) != a8_17 || a8_257 <= a8_17 ||
        capacity(ops::LinearPolicy::AllowA8, 1, 8192) != a8_8192 ||
        capacity(ops::LinearPolicy::A16Only, 1, 8192) != 0) {
        std::cerr << "FP8-FP32 K2560 gdn input workspace interval contract mismatch\n";
        ++failures;
    }
    // A FP32-scale parent is registered only at K=2560 and the BF16-scale one only at K=5120.
    for (const auto [qtype, hidden] : {std::pair{QType::FP8_E4M3FN_ROW_FP32, 5120},
                                       std::pair{QType::FP8_E4M3FN_ROW_BF16, 2560}}) {
        try {
            (void)ops::gdn_input_proj_workspace_capacity_bytes(
                qtype, kRows, hidden, ops::LinearPolicy::A16Only, 1, 1);
            std::cerr << "unregistered FP8 gdn input parent accepted\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    }

    failures += run_fp8_case(profile, parent, 1, ops::LinearPolicy::A16Only, true);
    // A16 contraction boundaries, then the A8 frontier and its tile, TMA and split-K boundaries.
    for (const std::int32_t tokens : {1, 2, 8, 16, 17, 24, 25, 32, 33, 64, 65, 96, 97, 512, 8192})
        failures += run_fp8_case(profile, parent, tokens, ops::LinearPolicy::A16Only);
    for (const std::int32_t tokens : {1,   2,   8,   16,  17,  32,  33,  64,  65,  128,  129,
                                      192, 193, 256, 257, 384, 385, 512, 513, 1024, 8192})
        failures += run_fp8_case(profile, parent, tokens,
                                 tokens == 17 ? ops::LinearPolicy::AllowA4
                                              : ops::LinearPolicy::AllowA8);
    for (int tokens : {1, 17, 512})
        failures += run_fp8_case(profile, parent, tokens, ops::LinearPolicy::AllowA8, false, true);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_q4_q5();
    failures += run_q8();
    failures += run_nvfp4();
    failures += run_fp8();
    failures += run_fp8_flash_next();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_input_proj\n";
    return failures == 0 ? 0 : 1;
}
