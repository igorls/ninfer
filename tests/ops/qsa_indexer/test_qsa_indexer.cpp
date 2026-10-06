// Qualification of ops::qsa_indexer_append and ops::qsa_indexer_select.
//
// Append: published block keys against an FP64 oracle (after the contract's BF16 mean cast),
// forming-state entries and positions bit for bit, untouched plane/state regions exact.
// Select: exact selections on inputs whose ideal scores are separated at the 512 boundary
// (quantized key levels, so boundary ties are bitwise ties resolved by the lowest id), plus a
// random-key case checked for ideal-score consistency within the score criterion.

#include "ninfer/ops/qsa_indexer.h"
#include "core/decode_graph.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kDim          = 128;
constexpr int kProjection   = 640;
constexpr int kBlocksPage   = 16;
constexpr int kMaxSelected  = 512;
constexpr double kEpsilon   = 1.0e-6;
constexpr double kTheta     = 1.0e7;
constexpr std::uint16_t kSentinel = 0x7fa5; // BF16 NaN payload: never produced by the Op

// Suite criteria. Block keys are BF16-stored normalized/rotated vectors (pair-scaled pointwise
// plus relative L2); scores only gate the random-key consistency check.
constexpr double kKeyPairRelative  = 6.9e-3;
constexpr double kKeyRelativeL2    = 2.5e-3; // BF16 store bound (1.95e-3) plus FP32 arithmetic
constexpr double kScoreTolerance   = 1.5e-2; // relative to the column's maximum ideal score

std::vector<std::uint16_t> random_bf16(std::size_t count, std::mt19937& rng, float low, float high) {
    std::uniform_real_distribution<float> d(low, high);
    std::vector<std::uint16_t> bits(count);
    for (auto& b : bits) b = f32_to_bf16(d(rng));
    return bits;
}

double bf(std::uint16_t bits) { return bf16_to_f32(bits); }

// FP64 one-centered RMSNorm + interleaved MRoPE (64 of 128 dims). Also returns per-dim criterion
// scales (pair magnitude for rotated dims).
void norm_rotate(const double* x, const std::vector<std::uint16_t>& weight, const std::int32_t* axes,
                 double* out, double* scale) {
    double sum = 0.0;
    for (int d = 0; d < kDim; ++d) sum += x[d] * x[d];
    const double inverse = 1.0 / std::sqrt(sum / kDim + kEpsilon);
    double n[kDim];
    for (int d = 0; d < kDim; ++d) n[d] = x[d] * inverse * (1.0 + bf(weight[static_cast<std::size_t>(d)]));
    for (int d = 64; d < kDim; ++d) {
        out[d] = n[d];
        if (scale) scale[d] = std::abs(n[d]);
    }
    for (int i = 0; i < 32; ++i) {
        const double phase = static_cast<double>(axes[i % 3]) * std::pow(kTheta, -2.0 * i / 64.0);
        const double c = std::cos(phase), s = std::sin(phase);
        out[i]      = n[i] * c - n[i + 32] * s;
        out[i + 32] = n[i + 32] * c + n[i] * s;
        if (scale) scale[i] = scale[i + 32] = std::hypot(n[i], n[i + 32]);
    }
}

struct Plane {
    int pages = 0, logical = 0, rows = 0;
    std::vector<std::int32_t> tables; // [logical, rows]
    std::vector<std::uint16_t> keys;  // [128,16,pages]
    std::size_t offset(int row, int block) const {
        const int page = tables[static_cast<std::size_t>(row) * logical + block / kBlocksPage];
        return (static_cast<std::size_t>(page) * kBlocksPage + block % kBlocksPage) * kDim;
    }
};

Plane make_plane(int rows, int blocks_per_row, std::mt19937& rng) {
    Plane plane;
    plane.rows    = rows;
    plane.logical = (blocks_per_row + kBlocksPage - 1) / kBlocksPage;
    plane.pages   = plane.logical * rows + 2;
    std::vector<int> ids(static_cast<std::size_t>(plane.pages));
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    plane.tables.assign(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(plane.logical) * rows);
    plane.keys.assign(static_cast<std::size_t>(plane.pages) * kBlocksPage * kDim, kSentinel);
    return plane;
}

Tensor plane_tensor(DeviceBuffer& keys, const Plane& plane) {
    return Tensor(keys.p, DType::BF16, {kDim, kBlocksPage, plane.pages});
}

int compare_keys(const std::string& label, const std::vector<double>& got,
                 const std::vector<double>& expected, const std::vector<double>& scale) {
    double error2 = 0.0, reference2 = 0.0, worst = 0.0;
    int violations = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) { ++violations; continue; }
        const double e = std::abs(got[i] - expected[i]);
        const double limit = kKeyPairRelative * scale[i];
        worst = std::max(worst, limit > 0 ? e / limit : (e > 0 ? 1e30 : 0.0));
        if (e > limit) ++violations;
        error2 += e * e;
        reference2 += expected[i] * expected[i];
    }
    const double l2 = std::sqrt(error2 / std::max(reference2, 1e-30));
    if (error_stats_enabled()) {
        std::cout << "OP_ERROR_STATS kind=qsa_block_key count=" << got.size() << " relative_l2=" << l2
                  << " max_pair_limit_ratio=" << worst << " case=" << label << '\n';
    }
    if (violations != 0 || l2 > kKeyRelativeL2) {
        std::cerr << label << ": block keys violations=" << violations << " rel_l2=" << l2 << '\n';
        return 1;
    }
    return 0;
}

// Oracle of one published key from four raw BF16 keys and the first token's axes.
void oracle_block(const std::uint16_t* k0, const std::uint16_t* k1, const std::uint16_t* k2,
                  const std::uint16_t* k3, const std::int32_t* axes,
                  const std::vector<std::uint16_t>& norm, double* out, double* scale) {
    double mean[kDim];
    for (int d = 0; d < kDim; ++d) {
        const double m = (bf(k0[d]) + bf(k1[d]) + bf(k2[d]) + bf(k3[d])) / 4.0;
        mean[d]        = bf(f32_to_bf16(static_cast<float>(m))); // the contract's BF16 mean cast
    }
    norm_rotate(mean, norm, axes, out, scale);
}

// ---------------------------------------------------------------- append, shared row
int run_append_shared(int first_position, int tokens, bool same_slot, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const int slots = 4, source = 1, destination = same_slot ? 1 : 3;
    const int last  = first_position + tokens - 1;
    Plane plane     = make_plane(1, last / 4 + 2, rng);
    auto raw_keys   = random_bf16(static_cast<std::size_t>(slots) * 4 * kDim, rng, -3.0F, 3.0F);
    std::vector<std::int32_t> raw_positions(static_cast<std::size_t>(slots) * 12);
    for (auto& p : raw_positions) p = static_cast<std::int32_t>(rng() % 200000U);
    const auto projected = random_bf16(static_cast<std::size_t>(tokens) * kProjection, rng, -3.0F, 3.0F);
    const auto norm      = random_bf16(kDim, rng, -0.7F, 0.7F);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens)), rope(static_cast<std::size_t>(tokens) * 3);
    for (int t = 0; t < tokens; ++t) {
        positions[static_cast<std::size_t>(t)] = first_position + t;
        for (int a = 0; a < 3; ++a)
            rope[static_cast<std::size_t>(a) * tokens + t] = first_position + t + a * 311 - (t % 5) * a;
    }
    // Host model of the forming block: ordinal s over [leftover entries of source] + tokens.
    const int leftover = first_position & 3;
    auto raw_at = [&](int s) -> const std::uint16_t* {
        return s < leftover ? &raw_keys[(static_cast<std::size_t>(source) * 4 + s) * kDim]
                            : &projected[static_cast<std::size_t>(s - leftover) * kProjection + 512];
    };
    auto axes_at = [&](int s, std::int32_t* axes) {
        for (int a = 0; a < 3; ++a)
            axes[a] = s < leftover ? raw_positions[(static_cast<std::size_t>(source) * 4 + s) * 3 + a]
                                   : rope[static_cast<std::size_t>(a) * tokens + (s - leftover)];
    };
    const int complete = (leftover + tokens) / 4;
    const int out      = (leftover + tokens) & 3;
    std::vector<std::uint16_t> expected_keys_plane = plane.keys;

    DeviceBuffer keys_d = to_device(plane.keys), tables_d = to_device(plane.tables),
                 raw_d = to_device(raw_keys), rawpos_d = to_device(raw_positions),
                 proj_d = to_device(projected), pos_d = to_device(positions), rope_d = to_device(rope),
                 norm_d = to_device(norm);
    ops::QsaIndexerKeyState state{Tensor(raw_d.p, DType::BF16, {kDim, 4, slots}),
                                  Tensor(rawpos_d.p, DType::I32, {3, 4, slots})};
    ops::QsaIndexerBlockKeys blocks{plane_tensor(keys_d, plane),
                                    Tensor(tables_d.p, DType::I32, {plane.logical, 1})};
    ops::qsa_indexer_append(Tensor(proj_d.p, DType::BF16, {kProjection, tokens}),
                            Tensor(pos_d.p, DType::I32, {tokens}),
                            Tensor(rope_d.p, DType::I32, {tokens, 3}),
                            Tensor(norm_d.p, DType::BF16, {kDim}), source, destination, state,
                            blocks, nullptr);
    cuda_synchronize();

    const std::string label = "qsa_indexer_append shared p0=" + std::to_string(first_position) +
                              " T=" + std::to_string(tokens) + " same=" + std::to_string(same_slot);
    int failures = 0;
    const auto got_plane = from_device<std::uint16_t>(keys_d, plane.keys.size());
    std::vector<double> got, expected, scale;
    const int first_block = (first_position - leftover) / 4;
    for (int b = 0; b < complete; ++b) {
        std::int32_t axes[3];
        axes_at(4 * b, axes);
        double o[kDim], s[kDim];
        oracle_block(raw_at(4 * b), raw_at(4 * b + 1), raw_at(4 * b + 2), raw_at(4 * b + 3), axes,
                     norm, o, s);
        const std::size_t off = plane.offset(0, first_block + b);
        for (int d = 0; d < kDim; ++d) {
            got.push_back(bf(got_plane[off + d]));
            expected.push_back(o[d]);
            scale.push_back(s[d]);
            expected_keys_plane[off + d] = got_plane[off + d]; // mark as written
        }
    }
    if (!got.empty()) failures += compare_keys(label, got, expected, scale);
    failures += verify_exact((label + " untouched plane").c_str(), got_plane, expected_keys_plane);
    // Forming state of the destination: entries [0,out) are the trailing tokens, bit exact.
    const auto got_raw    = from_device<std::uint16_t>(raw_d, raw_keys.size());
    const auto got_rawpos = from_device<std::int32_t>(rawpos_d, raw_positions.size());
    std::vector<std::uint16_t> want_raw;
    std::vector<std::uint16_t> have_raw;
    std::vector<std::int32_t> want_pos, have_pos;
    for (int e = 0; e < out; ++e) {
        const int s = complete * 4 + e;
        std::int32_t axes[3];
        axes_at(s, axes);
        for (int d = 0; d < kDim; ++d) {
            want_raw.push_back(raw_at(s)[d]);
            have_raw.push_back(got_raw[(static_cast<std::size_t>(destination) * 4 + e) * kDim + d]);
        }
        for (int a = 0; a < 3; ++a) {
            want_pos.push_back(axes[a]);
            have_pos.push_back(got_rawpos[(static_cast<std::size_t>(destination) * 4 + e) * 3 + a]);
        }
    }
    failures += verify_exact((label + " forming keys").c_str(), have_raw, want_raw);
    failures += verify_exact((label + " forming positions").c_str(), have_pos, want_pos);
    // Slots other than the destination are untouched.
    for (int slot = 0; slot < slots; ++slot) {
        if (slot == destination) continue;
        std::vector<std::uint16_t> a(raw_keys.begin() + slot * 4 * kDim, raw_keys.begin() + (slot + 1) * 4 * kDim);
        std::vector<std::uint16_t> b(got_raw.begin() + slot * 4 * kDim, got_raw.begin() + (slot + 1) * 4 * kDim);
        failures += verify_exact((label + " other slot " + std::to_string(slot)).c_str(), b, a);
    }
    return failures;
}

// ---------------------------------------------------------------- append, snapshots
int run_append_snapshot(int width, const std::vector<int>& first_positions, std::uint32_t seed, bool graph, bool ragged = false) {
    std::mt19937 rng(seed);
    const int batch = static_cast<int>(first_positions.size());
    const int slots = 2 + batch * (width + 1);
    int max_position = 0;
    for (int p : first_positions) max_position = std::max(max_position, p + width);
    Plane plane   = make_plane(batch, max_position / 4 + 2, rng);
    auto raw_keys = random_bf16(static_cast<std::size_t>(slots) * 4 * kDim, rng, -3.0F, 3.0F);
    std::vector<std::int32_t> raw_positions(static_cast<std::size_t>(slots) * 12);
    for (auto& p : raw_positions) p = static_cast<std::int32_t>(rng() % 200000U);
    const int columns    = width * batch;
    const auto projected = random_bf16(static_cast<std::size_t>(columns) * kProjection, rng, -3.0F, 3.0F);
    const auto norm      = random_bf16(kDim, rng, -0.7F, 0.7F);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(columns)), rope(static_cast<std::size_t>(columns) * 3),
        table_rows(static_cast<std::size_t>(batch)), initial(static_cast<std::size_t>(batch)),
        base(static_cast<std::size_t>(batch)), valid(static_cast<std::size_t>(batch), width);
    for (int b = 0; b < batch; ++b) {
        valid[b] = ragged ? 1 + b % width : width;
        table_rows[static_cast<std::size_t>(b)] = batch - 1 - b;
        base[static_cast<std::size_t>(b)]       = 1 + b * (width + 1);
        // Row 0's initial slot lies inside its own snapshot interval.
        initial[static_cast<std::size_t>(b)] = b == 0 ? base[0] : base[static_cast<std::size_t>(b)] + width;
        for (int w = 0; w < width; ++w) {
            const int c = w + width * b;
            positions[static_cast<std::size_t>(c)] = w < valid[b] ? first_positions[static_cast<std::size_t>(b)] + w : -1001;
            for (int a = 0; a < 3; ++a)
                rope[static_cast<std::size_t>(a) * columns + c] = positions[static_cast<std::size_t>(c)] * (a + 1) + 7;
        }
    }
    // Host replay of the state machine.
    std::vector<std::uint16_t> want_raw = raw_keys;
    std::vector<std::int32_t> want_pos  = raw_positions;
    struct Published { int row, block; double key[kDim], scale[kDim]; };
    std::vector<Published> published;
    std::vector<int> defined(static_cast<std::size_t>(slots), 0); // defined entries per snapshot slot
    std::vector<bool> is_snapshot(static_cast<std::size_t>(slots), false);
    for (int b = 0; b < batch; ++b) {
        std::uint16_t forming[4][kDim];
        std::int32_t forming_pos[4][3];
        const int init = initial[static_cast<std::size_t>(b)];
        for (int r = 0; r < 4; ++r) {
            for (int d = 0; d < kDim; ++d) forming[r][d] = raw_keys[(static_cast<std::size_t>(init) * 4 + r) * kDim + d];
            for (int a = 0; a < 3; ++a) forming_pos[r][a] = raw_positions[(static_cast<std::size_t>(init) * 4 + r) * 3 + a];
        }
        for (int w = 0; w < valid[b]; ++w) {
            const int c = w + width * b;
            const int p = positions[static_cast<std::size_t>(c)];
            const int e = p & 3;
            for (int d = 0; d < kDim; ++d) forming[e][d] = projected[static_cast<std::size_t>(c) * kProjection + 512 + d];
            for (int a = 0; a < 3; ++a) forming_pos[e][a] = rope[static_cast<std::size_t>(a) * columns + c];
            const int snap = base[static_cast<std::size_t>(b)] + w;
            is_snapshot[static_cast<std::size_t>(snap)] = true;
            defined[static_cast<std::size_t>(snap)]     = (p + 1) & 3;
            for (int r = 0; r < 4; ++r) {
                for (int d = 0; d < kDim; ++d) want_raw[(static_cast<std::size_t>(snap) * 4 + r) * kDim + d] = forming[r][d];
                for (int a = 0; a < 3; ++a) want_pos[(static_cast<std::size_t>(snap) * 4 + r) * 3 + a] = forming_pos[r][a];
            }
            if (e == 3) {
                Published item{};
                item.row   = table_rows[static_cast<std::size_t>(b)];
                item.block = p / 4;
                oracle_block(forming[0], forming[1], forming[2], forming[3], forming_pos[0], norm,
                             item.key, item.scale);
                published.push_back(item);
            }
        }
    }

    DeviceBuffer keys_d = to_device(plane.keys), tables_d = to_device(plane.tables),
                 raw_d = to_device(raw_keys), rawpos_d = to_device(raw_positions),
                 proj_d = to_device(projected), pos_d = to_device(positions), rope_d = to_device(rope),
                 norm_d = to_device(norm), rows_d = to_device(table_rows), init_d = to_device(initial),
                 base_d = to_device(base), valid_d = to_device(valid);
    ops::QsaIndexerKeyState state{Tensor(raw_d.p, DType::BF16, {kDim, 4, slots}),
                                  Tensor(rawpos_d.p, DType::I32, {3, 4, slots})};
    ops::QsaIndexerBlockKeys blocks{plane_tensor(keys_d, plane),
                                    Tensor(tables_d.p, DType::I32, {plane.logical, batch})};
    const auto launch = [&](cudaStream_t stream) {
        ops::qsa_indexer_append(Tensor(proj_d.p, DType::BF16, {kProjection, width, batch}),
                                Tensor(pos_d.p, DType::I32, {width, batch}),
                                Tensor(rope_d.p, DType::I32, {columns, 3}),
                                Tensor(rows_d.p, DType::I32, {batch}), Tensor(init_d.p, DType::I32, {batch}),
                                Tensor(base_d.p, DType::I32, {batch}),
                                ragged ? Tensor(valid_d.p, DType::I32, {batch}) : Tensor{}, Tensor(norm_d.p, DType::BF16, {kDim}),
                                state, blocks, stream);
    };
    if (graph) {
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        {
            DecodeGraphDefinition definition;
            DecodeGraphExecutable executable;
            definition.capture(stream, [&] { launch(stream); });
            executable.instantiate(definition);
            executable.launch(stream);
            cuda_synchronize(stream);
        }
        cuda_check(cudaStreamDestroy(stream), "destroy stream");
    } else {
        launch(nullptr);
        cuda_synchronize();
    }
    const std::string label = "qsa_indexer_append snapshot W=" + std::to_string(width) +
                              " B=" + std::to_string(batch) + " ragged=" + std::to_string(ragged) + " graph=" + std::to_string(graph);
    int failures        = 0;
    const auto got_raw    = from_device<std::uint16_t>(raw_d, raw_keys.size());
    const auto got_rawpos = from_device<std::int32_t>(rawpos_d, raw_positions.size());
    std::vector<std::uint16_t> a16, b16;
    std::vector<std::int32_t> a32, b32;
    for (int slot = 0; slot < slots; ++slot) {
        // Snapshot slots: defined entries exact. Every other slot: completely untouched.
        const int entries = is_snapshot[static_cast<std::size_t>(slot)] ? defined[static_cast<std::size_t>(slot)] : 4;
        const auto& reference_raw = is_snapshot[static_cast<std::size_t>(slot)] ? want_raw : raw_keys;
        const auto& reference_pos = is_snapshot[static_cast<std::size_t>(slot)] ? want_pos : raw_positions;
        for (int r = 0; r < entries; ++r) {
            for (int d = 0; d < kDim; ++d) {
                const std::size_t i = (static_cast<std::size_t>(slot) * 4 + r) * kDim + d;
                a16.push_back(reference_raw[i]);
                b16.push_back(got_raw[i]);
            }
            for (int a = 0; a < 3; ++a) {
                const std::size_t i = (static_cast<std::size_t>(slot) * 4 + r) * 3 + a;
                a32.push_back(reference_pos[i]);
                b32.push_back(got_rawpos[i]);
            }
        }
    }
    failures += verify_exact((label + " state keys").c_str(), b16, a16);
    failures += verify_exact((label + " state positions").c_str(), b32, a32);
    const auto got_plane = from_device<std::uint16_t>(keys_d, plane.keys.size());
    std::vector<std::uint16_t> untouched = plane.keys;
    std::vector<double> got, expected, scale;
    for (const auto& item : published) {
        const std::size_t off = plane.offset(item.row, item.block);
        for (int d = 0; d < kDim; ++d) {
            got.push_back(bf(got_plane[off + d]));
            expected.push_back(item.key[d]);
            scale.push_back(item.scale[d]);
            untouched[off + d] = got_plane[off + d];
        }
    }
    if (!got.empty()) failures += compare_keys(label, got, expected, scale);
    failures += verify_exact((label + " untouched plane").c_str(), got_plane, untouched);
    return failures;
}

// ---------------------------------------------------------------- select
struct SelectCase {
    int rows_form = 0;         // 1: batched table_rows form, 0: shared row
    int columns   = 1;
    int first_position = 0;    // shared row: consecutive positions; batched: per-column spread
    int envelope  = 0;
    bool leveled  = true;      // quantized key levels (exact) vs random keys (consistency)
    bool graph    = false;
    int levels    = 24;        // distinct key levels (1: every block key identical)
};

int run_select(const SelectCase& cs, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const int columns = cs.columns;
    const int rows    = cs.rows_form ? columns : 1;
    std::vector<std::int32_t> positions(static_cast<std::size_t>(columns)), table_rows(static_cast<std::size_t>(columns));
    for (int c = 0; c < columns; ++c) {
        positions[static_cast<std::size_t>(c)] =
            cs.rows_form ? std::max(0, cs.envelope * 4 + 2 - static_cast<int>(rng() % 3000U))
                         : cs.first_position + c;
        table_rows[static_cast<std::size_t>(c)] = cs.rows_form ? (columns - 1 - c) : 0;
    }
    if (cs.rows_form) positions[0] = cs.envelope * 4 + 2; // one column exactly at the envelope
    int max_complete = 0;
    for (int p : positions) max_complete = std::max(max_complete, (p + 1) / 4);
    if (max_complete > cs.envelope) throw std::runtime_error("test envelope too small");
    Plane plane = make_plane(rows, std::max(cs.envelope, 1), rng);

    // Projected queries: rotated dims random, unrotated dims positive so q_h . v > 0.
    std::vector<std::uint16_t> projected(static_cast<std::size_t>(columns) * kProjection);
    std::uniform_real_distribution<float> unit(-1.0F, 1.0F), positive(0.3F, 1.5F);
    for (int c = 0; c < columns; ++c)
        for (int r = 0; r < kProjection; ++r)
            projected[static_cast<std::size_t>(c) * kProjection + r] =
                f32_to_bf16((r % kDim) >= 64 && cs.leveled ? positive(rng) : unit(rng));
    std::vector<std::int32_t> rope(static_cast<std::size_t>(columns) * 3);
    for (int c = 0; c < columns; ++c)
        for (int a = 0; a < 3; ++a)
            rope[static_cast<std::size_t>(a) * columns + c] = positions[static_cast<std::size_t>(c)] + a * 13;
    const auto norm = random_bf16(kDim, rng, -0.5F, 0.5F);
    // Block keys. Leveled: key = level * v with v zero on rotated dims and positive elsewhere; 24
    // levels 6% apart, so every tie at the 512 boundary is a bitwise tie.
    std::vector<std::uint16_t> v(kDim);
    for (int d = 0; d < kDim; ++d) v[static_cast<std::size_t>(d)] = f32_to_bf16(d < 64 ? 0.0F : positive(rng));
    std::uniform_int_distribution<int> level(0, cs.levels - 1);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < cs.envelope; ++b) {
            const std::size_t off = plane.offset(r, b);
            const float scale     = 0.25F * std::pow(1.06F, static_cast<float>(level(rng)));
            for (int d = 0; d < kDim; ++d) {
                plane.keys[off + d] = cs.leveled ? f32_to_bf16(scale * bf16_to_f32(v[static_cast<std::size_t>(d)]))
                                                 : f32_to_bf16(unit(rng));
            }
        }
    }
    // Ideal selections.
    std::vector<std::int32_t> want_sel(static_cast<std::size_t>(columns) * kMaxSelected, -1), want_count(static_cast<std::size_t>(columns));
    std::vector<std::vector<double>> ideal_scores(static_cast<std::size_t>(columns));
    for (int c = 0; c < columns; ++c) {
        const int complete = (positions[static_cast<std::size_t>(c)] + 1) / 4;
        const int n        = std::min(complete, kMaxSelected);
        want_count[static_cast<std::size_t>(c)] = n;
        std::vector<int> chosen(static_cast<std::size_t>(complete));
        std::iota(chosen.begin(), chosen.end(), 0);
        if (complete > kMaxSelected) {
            std::int32_t axes[3] = {rope[static_cast<std::size_t>(c)], rope[static_cast<std::size_t>(columns) + c],
                                    rope[2 * static_cast<std::size_t>(columns) + c]};
            double q[4][kDim];
            for (int h = 0; h < 4; ++h) {
                double x[kDim];
                for (int d = 0; d < kDim; ++d) x[d] = bf(projected[static_cast<std::size_t>(c) * kProjection + h * kDim + d]);
                norm_rotate(x, norm, axes, q[h], nullptr);
            }
            auto& scores = ideal_scores[static_cast<std::size_t>(c)];
            scores.resize(static_cast<std::size_t>(complete));
            const int row = table_rows[static_cast<std::size_t>(c)];
            for (int b = 0; b < complete; ++b) {
                const std::size_t off = plane.offset(row, b);
                double s = 0.0;
                for (int h = 0; h < 4; ++h) {
                    double dot = 0.0;
                    for (int d = 0; d < kDim; ++d) dot += q[h][d] * bf(plane.keys[off + d]);
                    s += std::max(dot, 0.0);
                }
                scores[static_cast<std::size_t>(b)] = s / std::sqrt(128.0);
            }
            std::stable_sort(chosen.begin(), chosen.end(), [&](int a, int b) {
                return scores[static_cast<std::size_t>(a)] > scores[static_cast<std::size_t>(b)];
            });
            chosen.resize(kMaxSelected);
            std::sort(chosen.begin(), chosen.end());
        }
        for (int j = 0; j < n; ++j) want_sel[static_cast<std::size_t>(c) * kMaxSelected + j] = chosen[static_cast<std::size_t>(j)];
    }

    DeviceBuffer keys_d = to_device(plane.keys), tables_d = to_device(plane.tables),
                 proj_d = to_device(projected), pos_d = to_device(positions), rope_d = to_device(rope),
                 norm_d = to_device(norm), rows_d = to_device(table_rows);
    GuardedDeviceBuffer sel_d(static_cast<std::size_t>(columns) * kMaxSelected * 4);
    GuardedDeviceBuffer count_d(static_cast<std::size_t>(columns) * 4);
    const ops::QsaIndexerSelectEnvelope envelope{cs.envelope};
    const std::size_t capacity = ops::qsa_indexer_select_workspace_capacity_bytes(1, columns, envelope);
    GuardedDeviceBuffer workspace_storage(std::max<std::size_t>(capacity, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_storage.data(), std::max<std::size_t>(capacity, 256)});
    ops::QsaIndexerBlockKeys blocks{plane_tensor(keys_d, plane),
                                    Tensor(tables_d.p, DType::I32, {plane.logical, rows})};
    Tensor sel_t(sel_d.data(), DType::I32, {kMaxSelected, columns});
    Tensor count_t(count_d.data(), DType::I32, {columns});
    const auto launch = [&](cudaStream_t stream) {
        const Tensor proj(proj_d.p, DType::BF16, {kProjection, columns});
        const Tensor pos(pos_d.p, DType::I32, {columns});
        const Tensor rp(rope_d.p, DType::I32, {columns, 3});
        const Tensor nm(norm_d.p, DType::BF16, {kDim});
        if (cs.rows_form) {
            ops::qsa_indexer_select(proj, pos, rp, Tensor(rows_d.p, DType::I32, {columns}), nm, blocks,
                                    envelope, workspace, sel_t, count_t, stream);
        } else {
            ops::qsa_indexer_select(proj, pos, rp, nm, blocks, envelope, workspace, sel_t, count_t, stream);
        }
    };
    if (cs.graph) {
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        {
            DecodeGraphDefinition definition;
            DecodeGraphExecutable executable;
            definition.capture(stream, [&] { launch(stream); });
            executable.instantiate(definition);
            executable.launch(stream);
            cuda_synchronize(stream);
        }
        cuda_check(cudaStreamDestroy(stream), "destroy stream");
    } else {
        launch(nullptr);
        cuda_synchronize();
    }
    const std::string label = std::string("qsa_indexer_select ") + (cs.rows_form ? "rows" : "shared") +
                              " C=" + std::to_string(columns) + " env=" + std::to_string(cs.envelope) +
                              " leveled=" + std::to_string(cs.leveled) + " graph=" + std::to_string(cs.graph);
    const auto got_sel   = from_device<std::int32_t>(sel_d.data(), want_sel.size());
    const auto got_count = from_device<std::int32_t>(count_d.data(), want_count.size());
    int failures = verify_exact((label + " counts").c_str(), got_count, want_count);
    failures += sel_d.verify_guards(label + " selection guards");
    failures += count_d.verify_guards(label + " count guards");
    failures += workspace_storage.verify_guards(label + " workspace guards");
    if (cs.leveled) {
        failures += verify_exact((label + " selections").c_str(), got_sel, want_sel);
        return failures;
    }
    // Random keys: the selection is a valid ascending set whose ideal scores dominate the rest
    // within the score criterion.
    for (int c = 0; c < columns; ++c) {
        const int complete = (positions[static_cast<std::size_t>(c)] + 1) / 4;
        const std::int32_t* row = &got_sel[static_cast<std::size_t>(c) * kMaxSelected];
        if (complete <= kMaxSelected) {
            for (int j = 0; j < kMaxSelected; ++j)
                if (row[j] != (j < complete ? j : -1)) { ++failures; break; }
            continue;
        }
        const auto& scores = ideal_scores[static_cast<std::size_t>(c)];
        std::vector<char> in(static_cast<std::size_t>(complete), 0);
        double min_in = INFINITY, max_out = -INFINITY, top = 0.0;
        bool ordered = true;
        for (int j = 0; j < kMaxSelected; ++j) {
            if (row[j] < 0 || row[j] >= complete || (j > 0 && row[j] <= row[j - 1])) ordered = false;
            else in[static_cast<std::size_t>(row[j])] = 1;
        }
        for (int b = 0; b < complete; ++b) {
            top = std::max(top, scores[static_cast<std::size_t>(b)]);
            if (in[static_cast<std::size_t>(b)]) min_in = std::min(min_in, scores[static_cast<std::size_t>(b)]);
            else max_out = std::max(max_out, scores[static_cast<std::size_t>(b)]);
        }
        if (!ordered || min_in < max_out - kScoreTolerance * top) {
            std::cerr << label << ": column " << c << " selection inconsistent (ordered=" << ordered
                      << " min_in=" << min_in << " max_out=" << max_out << ")\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "qsa_indexer: SKIP (CUDA unavailable)\n";
        return 77;
    }
    int failures = 0;
    // Append, shared row: every leftover phase, single token, tile-free chunks, same/distinct slots.
    for (int p0 : {0, 1, 2, 3, 4097})
        for (int t : {1, 3, 5, 64})
            failures += run_append_shared(p0, t, (p0 + t) % 2 == 0, 100U + p0 * 7U + t);
    failures += run_append_shared(29'998, 2'050, false, 7U);
    // Append, snapshots: decode (W=1) and speculative verify (W=4), B up to 8.
    failures += run_append_snapshot(1, {0, 3, 6, 7'501, 29'999}, 31U, false);
    failures += run_append_snapshot(4, {2, 5, 8'190}, 32U, false);
    failures += run_append_snapshot(4, {1, 2, 3, 4, 5, 6, 7, 30'001}, 33U, true);
    for (int residue = 0; residue < 4; ++residue) {
        failures += run_append_snapshot(6, {residue, residue + 4, residue + 8, residue + 12,
            residue + 16, residue + 20, residue + 24, residue + 28}, 34U + residue, residue % 2 != 0, true);
    }
    // Select: identity envelope, batched rows at ~30K context, shared-row chunks incl. tiling.
    failures += run_select({1, 5, 0, 400, true, false}, 41U);
    failures += run_select({1, 8, 0, 7'500, true, false}, 42U);
    failures += run_select({1, 8, 0, 7'500, true, true}, 43U);
    failures += run_select({1, 3, 0, 2'000, false, false}, 44U);
    failures += run_select({0, 1, 29'999, 7'500, true, false}, 45U);
    failures += run_select({0, 200, 2'000, 550, true, false}, 46U);
    failures += run_select({0, 300, 29'700, 7'500, true, false}, 47U);
    failures += run_select({0, 300, 262'000, 65'575, true, false}, 48U);
    failures += run_select({0, 120, 8'000, 2'030, false, false}, 49U);
    // Every score equal: the lowest 512 ids win.
    failures += run_select({0, 20, 29'970, 7'500, true, false, 1}, 50U);
    if (failures != 0) {
        std::cerr << "qsa_indexer failures=" << failures << '\n';
        return 1;
    }
    std::cout << "qsa_indexer: PASS\n";
    return 0;
}
