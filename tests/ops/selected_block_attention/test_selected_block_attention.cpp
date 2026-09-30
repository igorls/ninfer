// Qualification of ops::selected_block_attention against an independent FP64 oracle.
//
// The cache is written directly in the paged v3 layouts ([256,64,2,N] page-major, shuffled
// physical pages). Every cache token outside a column's visible set holds NaN, so any read of an
// unselected value poisons the output and fails the criterion.

#include "ninfer/ops/selected_block_attention.h"
#include "core/decode_graph.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <bit>
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

constexpr int kDim         = 256;
constexpr int kQueryHeads  = 24;
constexpr int kKvHeads     = 2;
constexpr int kPage        = 64;
constexpr int kMaxSelected = 512;

// Named criterion of the suite: BF16 output store, FP32 softmax statistics, and FP16/BF16
// staging of probabilities, rotated queries and decoded FP8 tiles on the tensor-core route.
constexpr ReductionCriterion kAttentionCriterion{
    .relative_l2                     = 4.0e-3,
    .gross_absolute                  = 2.0e-3,
    .gross_relative_to_max_reference = 1.2e-2,
};

enum class Profile { Bf16, Fp8 };

double fp16_to_double(std::uint16_t bits) {
    const int sign     = bits >> 15;
    const int exponent = (bits >> 10) & 31;
    const int mantissa = bits & 1023;
    double value;
    if (exponent == 0) {
        value = std::ldexp(static_cast<double>(mantissa), -24);
    } else if (exponent == 31) {
        value = mantissa ? std::nan("") : INFINITY;
    } else {
        value = std::ldexp(1.0 + mantissa / 1024.0, exponent - 15);
    }
    return sign ? -value : value;
}

// Exact encode of a value representable in FP16 (zero or a normal FP16 value with at most 11
// significant bits; the test only produces such values).
std::uint16_t double_to_fp16_exact(double value) {
    if (value == 0.0) { return 0; }
    const std::uint16_t sign = value < 0.0 ? 0x8000U : 0U;
    int exponent             = 0;
    const double fraction    = std::frexp(std::abs(value), &exponent); // [0.5,1)
    const int biased         = exponent - 1 + 15;
    const double mantissa    = (fraction * 2.0 - 1.0) * 1024.0;
    if (biased < 1 || biased > 30 || mantissa != std::floor(mantissa)) {
        throw std::runtime_error("value not exactly representable as a normal FP16");
    }
    return static_cast<std::uint16_t>(sign | (biased << 10) | static_cast<int>(mantissa));
}

double e4m3_to_double(std::uint8_t code) {
    const int sign     = code >> 7;
    const int exponent = (code >> 3) & 15;
    const int mantissa = code & 7;
    double value;
    if (exponent == 15 && mantissa == 7) { return std::nan(""); }
    if (exponent == 0) {
        value = std::ldexp(mantissa / 8.0, -6);
    } else {
        value = std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
    }
    return sign ? -value : value;
}

// Normalized Sylvester H256 applied to one vector.
std::vector<double> hadamard(const std::vector<double>& x) {
    std::vector<double> y(kDim, 0.0);
    for (int i = 0; i < kDim; ++i) {
        double sum = 0.0;
        for (int j = 0; j < kDim; ++j) {
            const int sign = std::popcount(static_cast<unsigned>(i & j)) & 1;
            sum += sign ? -x[static_cast<std::size_t>(j)] : x[static_cast<std::size_t>(j)];
        }
        y[static_cast<std::size_t>(i)] = sum / 16.0;
    }
    return y;
}

struct Column {
    int position = 0;
    int table_row = 0;
    std::vector<int> selected; // any order
};

// Host copy of one paged cache and its decoded view.
struct Cache {
    Profile profile;
    int physical_pages = 0;
    int logical_pages  = 0;
    int rows           = 0;
    std::vector<std::int32_t> tables;   // [logical, rows]
    std::vector<std::uint16_t> k16, v16; // BF16 K / FP16 V words
    std::vector<std::uint8_t> k8, v8;   // FP8 codes
    std::vector<std::uint16_t> ks, vs;  // FP16 scales [64,2,N]

    std::int64_t row_of(int table_row, int token, int head) const {
        const int page = tables[static_cast<std::size_t>(token / kPage) * 1 +
                                static_cast<std::size_t>(table_row) * logical_pages];
        return (static_cast<std::int64_t>(page) * kKvHeads + head) * kPage + token % kPage;
    }
    double key(std::int64_t row, int d) const {
        if (profile == Profile::Bf16) return bf16_to_f32(k16[static_cast<std::size_t>(row * kDim + d)]);
        return e4m3_to_double(k8[static_cast<std::size_t>(row * kDim + d)]) *
               fp16_to_double(ks[static_cast<std::size_t>(row)]);
    }
    double value(std::int64_t row, int d) const {
        if (profile == Profile::Bf16) return fp16_to_double(v16[static_cast<std::size_t>(row * kDim + d)]);
        return e4m3_to_double(v8[static_cast<std::size_t>(row * kDim + d)]) *
               fp16_to_double(vs[static_cast<std::size_t>(row)]);
    }
};

std::vector<int> visible_tokens(const Column& column) {
    std::vector<int> tokens;
    for (int block : column.selected) {
        for (int r = 0; r < 4; ++r) tokens.push_back(block * 4 + r);
    }
    const int complete = (column.position + 1) / 4;
    for (int t = complete * 4; t <= column.position; ++t) tokens.push_back(t);
    return tokens;
}

// Fill every token of every table row with NaN, then write finite random values at the visible
// tokens of all columns. Rows are distinct sequences (disjoint physical pages).
Cache make_cache(Profile profile, int rows, int context, const std::vector<Column>& columns,
                 std::mt19937& rng) {
    Cache cache;
    cache.profile        = profile;
    cache.rows           = rows;
    cache.logical_pages  = (context + kPage - 1) / kPage;
    cache.physical_pages = cache.logical_pages * rows + 3;
    std::vector<int> pages(static_cast<std::size_t>(cache.physical_pages));
    std::iota(pages.begin(), pages.end(), 0);
    std::shuffle(pages.begin(), pages.end(), rng);
    cache.tables.resize(static_cast<std::size_t>(cache.logical_pages) * rows);
    for (int r = 0; r < rows; ++r)
        for (int l = 0; l < cache.logical_pages; ++l)
            cache.tables[static_cast<std::size_t>(r) * cache.logical_pages + l] =
                pages[static_cast<std::size_t>(r) * cache.logical_pages + l];
    const std::size_t elements = static_cast<std::size_t>(cache.physical_pages) * kKvHeads * kPage * kDim;
    const std::size_t scale_rows = static_cast<std::size_t>(cache.physical_pages) * kKvHeads * kPage;
    if (profile == Profile::Bf16) {
        cache.k16.assign(elements, 0x7fc0U); // BF16 NaN
        cache.v16.assign(elements, 0x7e00U); // FP16 NaN
    } else {
        cache.k8.assign(elements, 0x7fU); // E4M3FN NaN
        cache.v8.assign(elements, 0x7fU);
        cache.ks.assign(scale_rows, 0x7e00U);
        cache.vs.assign(scale_rows, 0x7e00U);
    }
    std::uniform_real_distribution<float> unit(-1.0F, 1.0F);
    std::uniform_int_distribution<int> code(0, 0xfe);
    std::uniform_real_distribution<double> scale(0.6, 1.4);
    for (const Column& column : columns) {
        for (int token : visible_tokens(column)) {
            for (int head = 0; head < kKvHeads; ++head) {
                const std::int64_t row = cache.row_of(column.table_row, token, head);
                const auto base        = static_cast<std::size_t>(row * kDim);
                if (profile == Profile::Bf16) {
                    for (int d = 0; d < kDim; ++d) {
                        cache.k16[base + d] = f32_to_bf16(2.0F * unit(rng));
                        // V = FP16_RNE(BF16 value): exact for normal-range BF16 inputs.
                        const float v = bf16_to_f32(f32_to_bf16(unit(rng)));
                        cache.v16[base + d] = double_to_fp16_exact(std::abs(v) < 6.2e-5F ? 0.0 : v);
                    }
                } else {
                    for (int d = 0; d < kDim; ++d) {
                        int c = code(rng);
                        if ((c & 0x7f) == 0x7f) c = 0x7e;
                        cache.k8[base + d] = static_cast<std::uint8_t>(c);
                        c = code(rng);
                        if ((c & 0x7f) == 0x7f) c = 0x7e;
                        cache.v8[base + d] = static_cast<std::uint8_t>(c);
                    }
                    cache.ks[static_cast<std::size_t>(row)] =
                        double_to_fp16_exact(std::ldexp(std::round(scale(rng) * 1024.0), -18));
                    cache.vs[static_cast<std::size_t>(row)] =
                        double_to_fp16_exact(std::ldexp(std::round(scale(rng) * 1024.0), -19));
                }
            }
        }
    }
    return cache;
}

// FP64 oracle of the complete formula in the cache's stored basis.
std::vector<double> oracle(const Cache& cache, const std::vector<float>& q,
                           const std::vector<Column>& columns) {
    std::vector<double> out(q.size(), 0.0);
    for (std::size_t c = 0; c < columns.size(); ++c) {
        const auto tokens = visible_tokens(columns[c]);
        for (int h = 0; h < kQueryHeads; ++h) {
            const int g = h / 12;
            std::vector<double> qv(kDim);
            for (int d = 0; d < kDim; ++d) qv[static_cast<std::size_t>(d)] = q[(c * kQueryHeads + h) * kDim + d];
            if (cache.profile == Profile::Fp8) qv = hadamard(qv);
            if (tokens.empty()) continue;
            std::vector<double> scores(tokens.size());
            double maximum = -INFINITY;
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                const std::int64_t row = cache.row_of(columns[c].table_row, tokens[i], g);
                double dot             = 0.0;
                for (int d = 0; d < kDim; ++d) dot += qv[static_cast<std::size_t>(d)] * cache.key(row, d);
                scores[i] = dot / 16.0;
                maximum   = std::max(maximum, scores[i]);
            }
            double denominator = 0.0;
            for (double& s : scores) {
                s = std::exp(s - maximum);
                denominator += s;
            }
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                const std::int64_t row = cache.row_of(columns[c].table_row, tokens[i], g);
                for (int d = 0; d < kDim; ++d)
                    out[(c * kQueryHeads + h) * kDim + d] += scores[i] / denominator * cache.value(row, d);
            }
        }
    }
    return out;
}

struct DeviceCache {
    DeviceBuffer k, v, ks, vs, tables;
};

DeviceCache upload(const Cache& cache) {
    DeviceCache d;
    if (cache.profile == Profile::Bf16) {
        d.k = to_device(cache.k16);
        d.v = to_device(cache.v16);
    } else {
        d.k  = to_device(cache.k8);
        d.v  = to_device(cache.v8);
        d.ks = to_device(cache.ks);
        d.vs = to_device(cache.vs);
    }
    d.tables = to_device(cache.tables);
    return d;
}

template <class View>
View make_view(const Cache& cache, DeviceCache& d, int table_rows) {
    View view{};
    const int n = cache.physical_pages;
    if (cache.profile == Profile::Bf16) {
        view.k_pages = Tensor(d.k.p, DType::BF16, {kDim, kPage, kKvHeads, n});
        view.v_pages = Tensor(d.v.p, DType::FP16, {kDim, kPage, kKvHeads, n});
        view.storage = KvCacheStorage::BFloat16;
    } else {
        view.k_pages       = Tensor(d.k.p, DType::FP8_E4M3FN, {kDim, kPage, kKvHeads, n});
        view.v_pages       = Tensor(d.v.p, DType::FP8_E4M3FN, {kDim, kPage, kKvHeads, n});
        view.k_scale_pages = Tensor(d.ks.p, DType::FP16, {1, kPage, kKvHeads, n});
        view.v_scale_pages = Tensor(d.vs.p, DType::FP16, {1, kPage, kKvHeads, n});
        view.storage       = KvCacheStorage::Fp8E4M3Row256;
    }
    const Tensor tables(d.tables.p, DType::I32, {cache.logical_pages, table_rows});
    if constexpr (requires { view.block_tables; }) {
        view.block_tables = tables;
    } else {
        view.block_table = tables;
    }
    view.head_dim     = kDim;
    view.num_kv_heads = kKvHeads;
    return view;
}

// Selection generator: identity, a random subset, or a random 512-subset of a long prefix.
Column make_column(int position, int table_row, int mode, std::mt19937& rng) {
    Column column;
    column.position    = position;
    column.table_row   = table_row;
    const int complete = (position + 1) / 4;
    std::vector<int> ids(static_cast<std::size_t>(complete));
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    int count = std::min(complete, kMaxSelected);
    if (mode == 1) count = complete == 0 ? 0 : static_cast<int>(rng() % static_cast<unsigned>(count + 1));
    if (mode == 2) count = 0;
    ids.resize(static_cast<std::size_t>(count));
    column.selected = ids;
    return column;
}

void pack_columns(const std::vector<Column>& columns, std::vector<std::int32_t>& positions,
                  std::vector<std::int32_t>& rows, std::vector<std::int32_t>& selections,
                  std::vector<std::int32_t>& counts) {
    const std::size_t n = columns.size();
    positions.assign(n, 0);
    rows.assign(n, 0);
    counts.assign(n, 0);
    selections.assign(n * kMaxSelected, -7);
    for (std::size_t c = 0; c < n; ++c) {
        positions[c] = columns[c].position;
        rows[c]      = columns[c].table_row;
        counts[c]    = static_cast<std::int32_t>(columns[c].selected.size());
        for (std::size_t j = 0; j < columns[c].selected.size(); ++j)
            selections[c * kMaxSelected + j] = columns[c].selected[j];
    }
}

std::vector<float> make_query(std::size_t columns, float magnitude, std::mt19937& rng) {
    std::vector<float> q(columns * kQueryHeads * kDim);
    std::uniform_real_distribution<float> unit(-magnitude, magnitude);
    for (float& x : q) x = bf16_to_f32(f32_to_bf16(unit(rng)));
    return q;
}

std::vector<std::uint16_t> to_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

int check_empty_rows(const std::string& label, const std::vector<double>& got,
                     const std::vector<Column>& columns) {
    int failures = 0;
    for (std::size_t c = 0; c < columns.size(); ++c) {
        if (!visible_tokens(columns[c]).empty()) continue;
        for (int i = 0; i < kQueryHeads * kDim; ++i) {
            if (got[c * kQueryHeads * kDim + i] != 0.0) {
                std::cerr << label << ": empty visible set is not exact zero at column " << c << '\n';
                return 1;
            }
        }
    }
    return failures;
}

// Batched form, B columns on B distinct rows; optionally captured and replayed on fresh inputs.
int run_batched(Profile profile, int batch, int context, int mode, float magnitude,
                std::uint32_t seed, bool graph) {
    std::mt19937 rng(seed);
    std::vector<Column> columns;
    for (int b = 0; b < batch; ++b) {
        int position = context - 1 - static_cast<int>(rng() % 7U) - (b == 3 ? 1 : 0);
        // Row 1 of a subset case has no selected block and an empty tail: exact zero.
        if (b == 1 && mode == 1) position = (context / 4) * 4 - 1;
        columns.push_back(make_column(std::max(position, 0), b, b == 1 && mode == 1 ? 2 : mode, rng));
    }
    std::vector<Column> replay_columns;
    if (graph) {
        for (int b = 0; b < batch; ++b)
            replay_columns.push_back(make_column(std::max(context - 5 - b, 0), b, 1, rng));
    }
    std::vector<Column> all = columns;
    all.insert(all.end(), replay_columns.begin(), replay_columns.end());
    const Cache cache = make_cache(profile, batch, context, all, rng);
    DeviceCache device = upload(cache);
    const auto view    = make_view<PagedKVBatchLayerView>(cache, device, batch);

    const auto q = make_query(static_cast<std::size_t>(batch), magnitude, rng);
    const auto q_bits = to_bits(q);
    std::vector<std::int32_t> positions, rows, selections, counts;
    pack_columns(columns, positions, rows, selections, counts);
    DeviceBuffer q_device = to_device(q_bits), positions_device = to_device(positions),
                 rows_device = to_device(rows), selections_device = to_device(selections),
                 counts_device = to_device(counts);
    GuardedDeviceBuffer out(static_cast<std::size_t>(batch) * kQueryHeads * kDim * 2);
    const std::size_t capacity = ops::selected_block_attention_workspace_capacity_bytes(1, batch);
    GuardedDeviceBuffer workspace_storage(capacity);
    WorkspaceArena workspace(DeviceSpan{workspace_storage.data(), capacity});
    Tensor q_t(q_device.p, DType::BF16, {kDim, kQueryHeads, batch});
    Tensor positions_t(positions_device.p, DType::I32, {batch});
    Tensor rows_t(rows_device.p, DType::I32, {batch});
    Tensor selections_t(selections_device.p, DType::I32, {kMaxSelected, batch});
    Tensor counts_t(counts_device.p, DType::I32, {batch});
    Tensor out_t(out.data(), DType::BF16, {kDim, kQueryHeads, batch});
    const std::string base = std::string("selected_block_attention batched ") +
                             (profile == Profile::Bf16 ? "bf16" : "fp8") + " B=" +
                             std::to_string(batch) + " ctx=" + std::to_string(context) +
                             " mode=" + std::to_string(mode) + " graph=" + std::to_string(graph);
    int failures = 0;
    const auto verify = [&](const std::vector<Column>& cols, const std::string& label) {
        const auto got      = from_device_bf16(out.data(), q.size());
        const auto expected = oracle(cache, q, cols);
        failures += verify_reduction(label, got, expected, kAttentionCriterion);
        failures += check_empty_rows(label, got, cols);
        failures += out.verify_guards(label + " out guards");
        failures += workspace_storage.verify_guards(label + " workspace guards");
    };
    if (!graph) {
        ops::selected_block_attention(q_t, positions_t, rows_t, selections_t, counts_t, view,
                                      workspace, out_t, nullptr);
        cuda_synchronize();
        verify(columns, base);
        return failures;
    }
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        definition.capture(stream, [&] {
            ops::selected_block_attention(q_t, positions_t, rows_t, selections_t, counts_t, view,
                                          workspace, out_t, stream);
        });
        executable.instantiate(definition);
        executable.launch(stream);
        cuda_synchronize(stream);
        verify(columns, base + " replay0");
        // Fresh device inputs, same graph.
        pack_columns(replay_columns, positions, rows, selections, counts);
        positions_device.copy_from_host(positions.data(), positions.size() * 4);
        selections_device.copy_from_host(selections.data(), selections.size() * 4);
        counts_device.copy_from_host(counts.data(), counts.size() * 4);
        executable.launch(stream);
        cuda_synchronize(stream);
        verify(replay_columns, base + " replay1");
    }
    cuda_check(cudaStreamDestroy(stream), "destroy stream");
    return failures;
}

// Shared-row form: T consecutive positions of one sequence, each with its own selection.
int run_shared_row(Profile profile, int tokens, int first_position, int mode, float magnitude,
                   std::uint32_t seed, bool graph) {
    std::mt19937 rng(seed);
    std::vector<Column> columns;
    for (int t = 0; t < tokens; ++t)
        columns.push_back(make_column(first_position + t, 0, t % 5 == 4 && mode == 1 ? 2 : mode, rng));
    const int context  = first_position + tokens;
    const Cache cache  = make_cache(profile, 1, context, columns, rng);
    DeviceCache device = upload(cache);
    const auto view    = make_view<PagedKVLayerView>(cache, device, 1);
    const auto q       = make_query(static_cast<std::size_t>(tokens), magnitude, rng);
    const auto q_bits  = to_bits(q);
    std::vector<std::int32_t> positions, rows, selections, counts;
    pack_columns(columns, positions, rows, selections, counts);
    DeviceBuffer q_device = to_device(q_bits), positions_device = to_device(positions),
                 selections_device = to_device(selections), counts_device = to_device(counts);
    GuardedDeviceBuffer out(static_cast<std::size_t>(tokens) * kQueryHeads * kDim * 2);
    Tensor q_t(q_device.p, DType::BF16, {kDim, kQueryHeads, tokens});
    Tensor positions_t(positions_device.p, DType::I32, {tokens});
    Tensor selections_t(selections_device.p, DType::I32, {kMaxSelected, tokens});
    Tensor counts_t(counts_device.p, DType::I32, {tokens});
    Tensor out_t(out.data(), DType::BF16, {kDim, kQueryHeads, tokens});
    const auto launch = [&](cudaStream_t stream) {
        ops::selected_block_attention(q_t, positions_t, selections_t, counts_t, view, out_t, stream);
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
    const std::string label = std::string("selected_block_attention shared-row ") +
                              (profile == Profile::Bf16 ? "bf16" : "fp8") + " T=" +
                              std::to_string(tokens) + " p0=" + std::to_string(first_position) +
                              " mode=" + std::to_string(mode) + " graph=" + std::to_string(graph);
    const auto got      = from_device_bf16(out.data(), q.size());
    const auto expected = oracle(cache, q, columns);
    int failures        = verify_reduction(label, got, expected, kAttentionCriterion);
    failures += check_empty_rows(label, got, columns);
    failures += out.verify_guards(label + " guards");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "selected_block_attention: SKIP (CUDA unavailable)\n";
        return 77;
    }
    int failures = 0;
    for (Profile profile : {Profile::Bf16, Profile::Fp8}) {
        // Decode: B = 1, 2, 5, 8; dense short contexts, subsets with empty rows, and 512 of many.
        failures += run_batched(profile, 1, 3, 0, 1.0F, 11, false);
        failures += run_batched(profile, 1, 1'203, 0, 3.0F, 12, false);
        failures += run_batched(profile, 2, 2'049, 1, 2.0F, 13, false);
        failures += run_batched(profile, 5, 8'192, 0, 4.0F, 14, false);
        failures += run_batched(profile, 8, 9'000, 1, 1.0F, 15, false);
        failures += run_batched(profile, 8, 4'100, 0, 6.0F, 16, true);
        // Shared-row tensor-core route: single column, tile boundaries, empty rows, long prefix.
        failures += run_shared_row(profile, 1, 0, 0, 1.0F, 21, false);
        failures += run_shared_row(profile, 1, 2'047, 0, 3.0F, 22, false);
        failures += run_shared_row(profile, 17, 60, 0, 2.0F, 23, false);
        failures += run_shared_row(profile, 40, 3'000, 1, 4.0F, 24, false);
        failures += run_shared_row(profile, 96, 8'100, 0, 2.0F, 25, false);
        failures += run_shared_row(profile, 9, 5'000, 1, 1.0F, 26, true);
    }
    if (failures != 0) {
        std::cerr << "selected_block_attention failures=" << failures << '\n';
        return 1;
    }
    std::cout << "selected_block_attention: PASS\n";
    return 0;
}
