#include "ninfer/ops/ple_ngram.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/op_check.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

// Flash-Next per-layer-embedding Ops. ple_ngram_decode is an exact codec checked bit for bit
// against an independent host decode. The gated combine / convolution oracle evaluates the
// contract formula in FP64 from the represented BF16 inputs; the persistent BF16 u columns are
// the one semantic cast inside it.
namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr int kStreamWidth = 2560;
constexpr int kStreams     = 4;
constexpr int kWidth       = 10240;
constexpr int kHistory     = 9;
constexpr int kRowWidth    = 160;

// out = BF16(V + silu(conv)) is dominated by its BF16 store; a stored u can sit one BF16 step from
// the FP64-derived BF16(u) at a rounding tie, which moves one conv tap by <= 2^-8 |u|. A normwise
// criterion with a gross cap absorbs those isolated taps without accepting a corrupted column.
constexpr ReductionCriterion kOutputCriterion{3.0e-3, 1.0e-2, 5.0e-3};
// The stored u columns: one BF16 step of the FP64 value (FP32 statistics can move a tie).
constexpr PointwiseCriterion kStateCriterion{1.0e-6, 1.0 / 128.0};

struct Bf16Host {
    std::vector<std::uint16_t> bits;
    std::vector<double> values;
};

Bf16Host random_bf16(std::size_t count, std::uint32_t seed, float scale) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> normal(0.0F, scale);
    Bf16Host result{std::vector<std::uint16_t>(count), std::vector<double>(count)};
    for (std::size_t i = 0; i < count; ++i) {
        result.bits[i]   = f32_to_bf16(normal(generator));
        result.values[i] = bf16_to_f32(result.bits[i]);
    }
    return result;
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

std::uint16_t bf16_rne(double value) { return f32_to_bf16(static_cast<float>(value)); }

struct Weights {
    Bf16Host query_norm = random_bf16(kWidth, 21U, 0.1F);
    Bf16Host key_norm   = random_bf16(kWidth, 22U, 0.1F);
    Bf16Host conv_norm  = random_bf16(kWidth, 23U, 0.1F);
    Bf16Host conv       = random_bf16(static_cast<std::size_t>(kWidth) * 4, 24U, 0.5F);
    DeviceBuffer query_norm_d = to_device(query_norm.bits);
    DeviceBuffer key_norm_d   = to_device(key_norm.bits);
    DeviceBuffer conv_norm_d  = to_device(conv_norm.bits);
    DeviceBuffer conv_d       = to_device(conv.bits);

    Tensor tensor(const DeviceBuffer& buffer, std::int32_t taps = 1) const {
        return taps == 1 ? Tensor(buffer.p, DType::BF16, {kWidth})
                         : Tensor(buffer.p, DType::BF16, {kWidth, taps});
    }
};

struct Inputs {
    Bf16Host hidden, key, value;
};

Inputs make_inputs(std::int32_t columns, std::uint32_t seed) {
    Inputs in{random_bf16(static_cast<std::size_t>(kWidth) * columns, seed, 1.0F),
              random_bf16(static_cast<std::size_t>(kWidth) * columns, seed + 1, 0.8F),
              random_bf16(static_cast<std::size_t>(kStreamWidth) * columns, seed + 2, 0.5F)};
    return in;
}

// Column oracle: V in FP64 and u = BF16(n(V, conv_norm)).
void oracle_gate(const Weights& w, const Inputs& in, std::int32_t column, std::vector<double>& V,
                 std::vector<std::uint16_t>& u) {
    V.assign(kWidth, 0.0);
    u.assign(kWidth, 0);
    const auto wide   = static_cast<std::size_t>(column) * kWidth;
    const auto narrow = static_cast<std::size_t>(column) * kStreamWidth;
    for (int s = 0; s < kStreams; ++s) {
        const int o = s * kStreamWidth;
        double qq = 0.0, kk = 0.0;
        for (int d = 0; d < kStreamWidth; ++d) {
            qq += in.hidden.values[wide + o + d] * in.hidden.values[wide + o + d];
            kk += in.key.values[wide + o + d] * in.key.values[wide + o + d];
        }
        const double iq = 1.0 / std::sqrt(qq / kStreamWidth + 1e-6);
        const double ik = 1.0 / std::sqrt(kk / kStreamWidth + 1e-6);
        double dot      = 0.0;
        for (int d = 0; d < kStreamWidth; ++d) {
            dot += in.hidden.values[wide + o + d] * iq * (1.0 + w.query_norm.values[o + d]) *
                   in.key.values[wide + o + d] * ik * (1.0 + w.key_norm.values[o + d]);
        }
        const double raw  = dot / std::sqrt(static_cast<double>(kStreamWidth));
        const double sign = raw > 0 ? 1.0 : (raw < 0 ? -1.0 : 0.0);
        const double gate = sigmoid(sign * std::sqrt(std::max(std::abs(raw), 1e-6)));
        double vv         = 0.0;
        for (int d = 0; d < kStreamWidth; ++d) {
            V[o + d] = gate * in.value.values[narrow + d];
            vv += V[o + d] * V[o + d];
        }
        const double iv = 1.0 / std::sqrt(vv / kStreamWidth + 1e-6);
        for (int d = 0; d < kStreamWidth; ++d)
            u[o + d] = bf16_rne(V[o + d] * iv * (1.0 + w.conv_norm.values[o + d]));
    }
}

// Sequence oracle over `columns` from `history` (nine BF16 columns): output and new history.
void oracle_sequence(const Weights& w, const Inputs& in, std::int32_t first, std::int32_t columns,
                     std::vector<std::uint16_t> history, double* out,
                     std::vector<std::uint16_t>& new_history) {
    std::vector<std::vector<std::uint16_t>> window;
    for (int i = 0; i < kHistory; ++i)
        window.emplace_back(history.begin() + static_cast<std::ptrdiff_t>(i) * kWidth,
                            history.begin() + static_cast<std::ptrdiff_t>(i + 1) * kWidth);
    std::vector<double> V;
    std::vector<std::uint16_t> u;
    for (std::int32_t j = 0; j < columns; ++j) {
        oracle_gate(w, in, first + j, V, u);
        window.push_back(u);
        const std::size_t now = window.size() - 1;
        for (int c = 0; c < kWidth; ++c) {
            double conv = 0.0;
            for (int i = 0; i < 4; ++i)
                conv += w.conv.values[static_cast<std::size_t>(i) * kWidth + c] *
                        bf16_to_f32(window[now - 9 + 3 * i][c]);
            out[static_cast<std::size_t>(j) * kWidth + c] = V[c] + conv * sigmoid(conv);
        }
    }
    new_history.clear();
    for (std::size_t i = window.size() - kHistory; i < window.size(); ++i)
        new_history.insert(new_history.end(), window[i].begin(), window[i].end());
}

std::vector<double> bf16_values(const std::vector<std::uint16_t>& bits) {
    std::vector<double> result(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) result[i] = bf16_to_f32(bits[i]);
    return result;
}

int run_decode(std::int64_t rows) {
    const std::string label = "ple decode R=" + std::to_string(rows);
    std::mt19937 generator(static_cast<std::uint32_t>(rows));
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(rows) * kRowWidth / 2);
    std::vector<std::uint16_t> scales(static_cast<std::size_t>(rows) * kRowWidth / 16);
    for (auto& code : codes) code = static_cast<std::uint8_t>(generator());
    // Normal, subnormal, the largest finite and the positive-zero multiplier (zero point only).
    const std::uint16_t specials[]{0x0001U, 0x03ffU, 0x0400U, 0x7bffU, 0x3c00U};
    for (std::size_t g = 0; g < scales.size(); ++g) {
        scales[g] = g % 7 == 3 ? specials[(g / 7) % 5]
                               : static_cast<std::uint16_t>(0x1000U + generator() % 0x2c00U);
        if (g % 23 == 11) {
            scales[g] = 0;
            for (int k = 0; k < 8; ++k) codes[g * 8 + k] = 0x88U;
        }
    }
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(rows) * kRowWidth);
    for (std::int64_t r = 0; r < rows; ++r)
        for (int k = 0; k < kRowWidth; ++k) {
            const std::uint8_t byte = codes[static_cast<std::size_t>(r) * 80 + k / 2];
            const int u             = (byte >> (4 * (k % 2))) & 0xf;
            __half_raw raw;
            raw.x              = scales[static_cast<std::size_t>(r) * 10 + k / 16];
            const float scale  = __half2float(__half(raw));
            const float w_hat  = static_cast<float>(u - 8) * scale;
            expected[static_cast<std::size_t>(r) * kRowWidth + k] = f32_to_bf16(w_hat);
        }
    DeviceBuffer codes_d  = to_device(codes);
    DeviceBuffer scales_d = to_device(scales);
    GuardedDeviceBuffer out_d(expected.size() * 2);
    Tensor codes_t(codes_d.p, DType::U8, {80, static_cast<std::int32_t>(rows)});
    Tensor scales_t(scales_d.p, DType::FP16, {10, static_cast<std::int32_t>(rows)});
    Tensor out(out_d.data(), DType::BF16, {kRowWidth, static_cast<std::int32_t>(rows)});
    ops::ple_ngram_decode(codes_t, scales_t, out, nullptr);
    cuda_synchronize();
    int failures = out_d.verify_guards(label);
    failures += verify_exact(label.c_str(), from_device<std::uint16_t>(out_d.data(), expected.size()),
                             expected);
    return failures;
}

int run_sequence(const Weights& w, std::int32_t T, bool aliased_state, bool graph) {
    const std::string label = "ple_ngram T=" + std::to_string(T) + (aliased_state ? " in-place" : "") +
                              (graph ? " graph" : "");
    const Inputs in = make_inputs(T, 100U + static_cast<std::uint32_t>(T));
    const Bf16Host history = random_bf16(static_cast<std::size_t>(kWidth) * kHistory, 7U + T, 1.0F);
    DeviceBuffer h_d = to_device(in.hidden.bits), k_d = to_device(in.key.bits),
                 v_d = to_device(in.value.bits), s_in = to_device(history.bits);
    DeviceBuffer s_out_buffer(history.bits.size() * 2);
    GuardedDeviceBuffer out_d(static_cast<std::size_t>(kWidth) * T * 2);
    Tensor hidden(h_d.p, DType::BF16, {kWidth, T}), key(k_d.p, DType::BF16, {kWidth, T}),
        value(v_d.p, DType::BF16, {kStreamWidth, T}), out(out_d.data(), DType::BF16, {kWidth, T});
    Tensor state_in(s_in.p, DType::BF16, {kWidth, kHistory});
    Tensor state_out(aliased_state ? s_in.p : s_out_buffer.p, DType::BF16, {kWidth, kHistory});
    const auto capacity = ops::ple_ngram_workspace_capacity_bytes(T, T);
    DeviceArena workspace(capacity);
    DeviceContext context;
    const auto launch = [&] {
        ops::ple_ngram(hidden, key, value, w.tensor(w.query_norm_d), w.tensor(w.key_norm_d),
                       w.tensor(w.conv_norm_d), w.tensor(w.conv_d, 4), state_in, state_out, out,
                       workspace, context.stream);
    };
    if (graph) {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        definition.capture(context.stream, launch);
        executable.instantiate(definition);
        executable.launch(context.stream);
    } else {
        launch();
    }
    cuda_synchronize(context.stream);
    int failures = out_d.verify_guards(label);
    if (workspace.peak_used() > capacity || workspace.used() != 0) {
        std::cerr << label << ": workspace query/scope mismatch\n";
        ++failures;
    }
    std::vector<double> expected(static_cast<std::size_t>(kWidth) * T);
    std::vector<std::uint16_t> new_history;
    oracle_sequence(w, in, 0, T, history.bits, expected.data(), new_history);
    failures += verify_reduction(label + " out",
                                 from_device_bf16(out_d.data(), expected.size()), expected,
                                 kOutputCriterion);
    const auto state = from_device<std::uint16_t>(state_out.data, new_history.size());
    failures += verify_pointwise(label + " state", bf16_values(state), bf16_values(new_history),
                                 kStateCriterion);
    if (!aliased_state && from_device<std::uint16_t>(s_in, history.bits.size()) != history.bits) {
        std::cerr << label << ": distinct input state was modified\n";
        ++failures;
    }
    return failures;
}

int run_snapshot(const Weights& w, std::int32_t W, std::int32_t B, bool ragged, bool graph) {
    const std::string label = "ple_ngram_snapshot W=" + std::to_string(W) + " B=" + std::to_string(B) +
                              (ragged ? " ragged" : "") + (graph ? " graph" : "");
    const std::int32_t slots = 2 + B * (W + 1);
    const Inputs in          = make_inputs(W * B, 500U + static_cast<std::uint32_t>(W * 10 + B));
    const Bf16Host initial_states =
        random_bf16(static_cast<std::size_t>(kWidth) * kHistory * slots, 9U + W + B, 1.0F);
    std::vector<std::int32_t> valid(B), initial(B), base(B);
    for (int b = 0; b < B; ++b) {
        valid[b]   = ragged ? 1 + (b * 3) % W : W;
        base[b]    = 2 + b * (W + 1);
        // Row 0 starts inside its own reservation; the others from a slot outside every
        // reservation or at the unused tail slot of their own interval.
        initial[b] = b == 0 ? base[b] : (b % 2 == 1 ? b % 2 : base[b] + W);
    }
    DeviceBuffer h_d = to_device(in.hidden.bits), k_d = to_device(in.key.bits),
                 v_d = to_device(in.value.bits), states_d = to_device(initial_states.bits),
                 valid_d = to_device(valid), initial_d = to_device(initial), base_d = to_device(base);
    GuardedDeviceBuffer out_d(static_cast<std::size_t>(kWidth) * W * B * 2);
    Tensor hidden(h_d.p, DType::BF16, {kWidth, W, B}), key(k_d.p, DType::BF16, {kWidth, W, B}),
        value(v_d.p, DType::BF16, {kStreamWidth, W, B}),
        out(out_d.data(), DType::BF16, {kWidth, W, B}),
        states(states_d.p, DType::BF16, {kWidth, kHistory, slots}),
        valid_t = ragged ? Tensor(valid_d.p, DType::I32, {B}) : Tensor{},
        initial_t(initial_d.p, DType::I32, {B}), base_t(base_d.p, DType::I32, {B});
    const auto capacity = ops::ple_ngram_workspace_capacity_bytes(W * B, W * B);
    DeviceArena workspace(capacity);
    DeviceContext context;
    out_d.fill(0x7f);
    cuda_synchronize();
    const auto launch = [&] {
        ops::ple_ngram_snapshot(hidden, key, value, w.tensor(w.query_norm_d),
                                w.tensor(w.key_norm_d), w.tensor(w.conv_norm_d),
                                w.tensor(w.conv_d, 4), states, valid_t, initial_t, base_t, out,
                                workspace, context.stream);
    };
    if (graph) {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        definition.capture(context.stream, launch);
        executable.instantiate(definition);
        executable.launch(context.stream);
    } else {
        launch();
    }
    cuda_synchronize(context.stream);
    int failures = out_d.verify_guards(label);
    const auto got_out   = from_device_bf16(out_d.data(), static_cast<std::size_t>(kWidth) * W * B);
    const auto got_state = from_device<std::uint16_t>(states_d, initial_states.bits.size());
    std::vector<bool> written(slots, false);
    constexpr std::size_t kSlot = static_cast<std::size_t>(kWidth) * kHistory;
    for (int b = 0; b < B; ++b) {
        // Each row is its own sequence over the columns (b, 0..W-1).
        Inputs row{{}, {}, {}};
        const auto slice = [&](const Bf16Host& source, std::size_t width) {
            Bf16Host result;
            result.bits.assign(source.bits.begin() + static_cast<std::ptrdiff_t>(b * W * width),
                               source.bits.begin() + static_cast<std::ptrdiff_t>((b + 1) * W * width));
            result.values.assign(source.values.begin() + static_cast<std::ptrdiff_t>(b * W * width),
                                 source.values.begin() + static_cast<std::ptrdiff_t>((b + 1) * W * width));
            return result;
        };
        row.hidden = slice(in.hidden, kWidth);
        row.key    = slice(in.key, kWidth);
        row.value  = slice(in.value, kStreamWidth);
        std::vector<std::uint16_t> history(
            initial_states.bits.begin() + static_cast<std::ptrdiff_t>(initial[b] * kSlot),
            initial_states.bits.begin() + static_cast<std::ptrdiff_t>((initial[b] + 1) * kSlot));
        for (int j = 0; j < valid[b]; ++j) {
            std::vector<double> expected(kWidth);
            std::vector<std::uint16_t> new_history;
            oracle_sequence(w, row, j, 1, history, expected.data(), new_history);
            const std::string at = label + " row " + std::to_string(b) + " col " + std::to_string(j);
            failures += verify_reduction(
                at + " out",
                std::vector<double>(got_out.begin() + static_cast<std::ptrdiff_t>((b * W + j) * kWidth),
                                    got_out.begin() + static_cast<std::ptrdiff_t>((b * W + j + 1) * kWidth)),
                expected, kOutputCriterion);
            const std::size_t slot = static_cast<std::size_t>(base[b] + j);
            written[slot]          = true;
            failures += verify_pointwise(
                at + " snapshot",
                bf16_values(std::vector<std::uint16_t>(got_state.begin() + static_cast<std::ptrdiff_t>(slot * kSlot),
                                                       got_state.begin() + static_cast<std::ptrdiff_t>((slot + 1) * kSlot))),
                bf16_values(new_history), kStateCriterion);
            history = new_history;
        }
        for (int j = valid[b]; j < W; ++j) {
            for (int c = 0; c < kWidth; ++c) {
                if (got_out[static_cast<std::size_t>(b * W + j) * kWidth + c] != 0.0) {
                    std::cerr << label << ": invalid tail column was not zero\n";
                    ++failures;
                    break;
                }
            }
        }
    }
    for (int slot = 0; slot < slots; ++slot) {
        if (written[slot]) continue;
        if (!std::equal(got_state.begin() + static_cast<std::ptrdiff_t>(slot * kSlot),
                        got_state.begin() + static_cast<std::ptrdiff_t>((slot + 1) * kSlot),
                        initial_states.bits.begin() + static_cast<std::ptrdiff_t>(slot * kSlot))) {
            std::cerr << label << ": untouched state slot " << slot << " changed\n";
            ++failures;
        }
    }
    return failures;
}

int run_ple_ngram() {
    int failures = 0;
    for (std::int64_t rows : {1, 16, 16 * 7, 16 * 4096}) failures += run_decode(rows);
    const Weights weights;
    for (std::int32_t T : {1, 2, 8, 9, 10, 17, 256, 8192})
        failures += run_sequence(weights, T, false, false);
    for (std::int32_t T : {1, 5, 9, 12}) failures += run_sequence(weights, T, true, false);
    failures += run_sequence(weights, 9, true, true);
    for (std::int32_t B : {1, 5, 8}) {
        failures += run_snapshot(weights, 1, B, false, false);
        failures += run_snapshot(weights, 4, B, true, false);
    }
    failures += run_snapshot(weights, 1, 8, false, true);
    failures += run_snapshot(weights, 4, 3, true, true);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_ple_ngram();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " ple_ngram\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "ple_ngram: " << error.what() << '\n';
        return 1;
    }
}
