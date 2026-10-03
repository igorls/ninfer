// Host-only checks for the Flash-Next 1M context envelope and static YaRN table.
// Does not call finalize_flash_next_runtime_plan: that sizes CUB scratch through CUDA.

#include "targets/qwen3_8_flash_next/impl/long_context.h"
#include "targets/qwen3_8_flash_next/impl/qsa_indexer_workspace.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {

using ninfer::targets::qwen3_8_flash_next::detail::flash_next_context_allowed;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_default_inv_freq;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_indexer_blocks;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_qsa_indexer_tile_size;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_rope_scaling;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_rotate_pair;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_yarn_attention_factor;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_yarn_correction_bounds;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_yarn_enabled;
using ninfer::targets::qwen3_8_flash_next::detail::flash_next_yarn_inv_freq;
using ninfer::targets::qwen3_8_flash_next::detail::kMaxContextTokens;
using ninfer::targets::qwen3_8_flash_next::detail::kMaxIndexerBlocks;
using ninfer::targets::qwen3_8_flash_next::detail::kMaxScoreWorkspaceBytes;
using ninfer::targets::qwen3_8_flash_next::detail::kNativeContextTokens;

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        ++failures;
    }
}

bool near(double got, double want, double rel = 1.0e-12) {
    const double scale = std::max(std::abs(want), 1.0);
    return std::abs(got - want) <= rel * scale;
}

} // namespace

int main() {
    expect(kNativeContextTokens == 262144U, "native window");
    expect(kMaxContextTokens == 1000000U, "ceiling");
    expect(kMaxIndexerBlocks == 250000, "block ceiling");

    expect(flash_next_context_allowed(1), "context 1");
    expect(flash_next_context_allowed(262144), "context 262144");
    expect(flash_next_context_allowed(262145), "context 262145");
    expect(flash_next_context_allowed(1000000), "context 1000000");
    expect(!flash_next_context_allowed(0), "context 0");
    expect(!flash_next_context_allowed(1000001), "context 1000001");
    expect(!flash_next_context_allowed(1500000), "context 1500000");

    expect(flash_next_indexer_blocks(262144) == 65536U, "256k blocks stay 65536");
    expect(flash_next_indexer_blocks(262145) == 65537U, "first block past the old cap");
    expect(flash_next_indexer_blocks(1000000) == 250000U, "1M is 250000 blocks of 4");
    expect(flash_next_indexer_blocks(1000000) * 4U == 1000000U, "1M token envelope");
    // The previous planner stored min(65536, ceil(max_context/4)). That clamp
    // covers only the native window, so a 1M request would leave the tail unindexed.
    expect(std::min(65536U, flash_next_indexer_blocks(1000000)) * 4U == 262144U,
           "old 65536 clamp covers only 262144 tokens");
    expect(std::min(65536U, flash_next_indexer_blocks(262145)) == 65536U,
           "old clamp truncates 262145");

    expect(!flash_next_yarn_enabled(32768), "32k stays default RoPE");
    expect(!flash_next_yarn_enabled(262144), "256k stays default RoPE");
    expect(flash_next_yarn_enabled(262145), "YaRN starts past the native window");
    expect(flash_next_yarn_enabled(1000000), "1M uses YaRN");

    const auto native = flash_next_rope_scaling(262144);
    const auto empty  = decltype(native){};
    expect(native.yarn == 0 && native.attention_factor == 1.0F, "native scaling flag");
    expect(native.yarn == empty.yarn && native.attention_factor == empty.attention_factor,
           "native scaling matches the default-constructed table");
    bool native_freq_clear = true;
    for (float value : native.inv_freq) { native_freq_clear = native_freq_clear && value == 0.0F; }
    expect(native_freq_clear, "native plan does not install a YaRN frequency table");

    const auto extended = flash_next_rope_scaling(1000000);
    const auto just_over = flash_next_rope_scaling(262145);
    expect(extended.yarn == 1 && just_over.yarn == 1, "extended plans enable YaRN");
    expect(near(flash_next_yarn_attention_factor(), 1.1386294361119891), "attention factor");
    expect(extended.attention_factor == just_over.attention_factor, "factor is not retuned");
    bool same_table = extended.attention_factor == just_over.attention_factor;
    for (int pair = 0; pair < 32; ++pair) {
        same_table = same_table && extended.inv_freq[pair] == just_over.inv_freq[pair];
    }
    expect(same_table, "262145 and 1000000 share the factor-4 table");

    double low = 0;
    double high = 0;
    flash_next_yarn_correction_bounds(low, high);
    expect(low == 14.0 && high == 22.0, "truncated correction bounds");

    const double yarn_inv[] = {
        1.0000000000000000e+00, 8.6596432336006539e-04, 4.7423982268010459e-04,
        2.5693505988868084e-04, 1.3734974507600040e-04, 7.2173874043091133e-05,
        3.7072249820680398e-05, 1.8449222025000475e-05, 8.7597700711790042e-06,
        3.8498163151487305e-06, 4.1370427498579540e-08,
    };
    const int yarn_pairs[] = {0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 31};
    const double yarn_ratios[] = {1.0, 1.0, 0.90625, 0.8125, 0.71875, 0.625,
                                  0.53125, 0.4375, 0.34375, 0.25, 0.25};
    for (int i = 0; i < 11; ++i) {
        const int pair = yarn_pairs[i];
        const double got = flash_next_yarn_inv_freq(pair);
        const double base = flash_next_default_inv_freq(pair);
        if (!near(got, yarn_inv[i]) || !near(got / base, yarn_ratios[i], 1.0e-9)) {
            std::cerr << "FAIL: yarn inv_freq pair " << pair << " got " << got << "\n";
            ++failures;
        }
        const float stored = extended.inv_freq[pair];
        if (stored != static_cast<float>(got)) {
            std::cerr << "FAIL: stored inv_freq pair " << pair << "\n";
            ++failures;
        }
    }
    expect(near(flash_next_yarn_inv_freq(0), flash_next_default_inv_freq(0)),
           "pair 0 is not interpolated");
    expect(near(flash_next_yarn_inv_freq(14), flash_next_default_inv_freq(14)),
           "pair 14 stays on the extrapolation side of the ramp");

    const double factor = flash_next_yarn_attention_factor();
    const auto origin_default = flash_next_rotate_pair(1.0, 0.5, 0, flash_next_default_inv_freq(0), 1.0);
    const auto origin_yarn =
        flash_next_rotate_pair(1.0, 0.5, 0, flash_next_yarn_inv_freq(0), factor);
    expect(near(origin_default.first, 1.0) && near(origin_default.second, 0.5),
           "position 0 default rotation");
    expect(near(origin_yarn.first, 1.1386294361119891) && near(origin_yarn.second, 0.5693147180559945),
           "position 0 YaRN is a pure magnitude scale");

    struct Rotation {
        std::int32_t position;
        int pair;
        double x0;
        double x1;
        bool yarn;
        double first;
        double second;
    };
    const Rotation rotations[] = {
        {100, 0, 1.0, 0.5, false, 1.1155016928425634, -0.07520620496591685},
        {100, 0, 1.0, 0.5, true, 1.270143063503297, -0.08563199875246463},
        {100, 31, 1.0, 0.5, false, 0.9999917257775797, 0.5000165481025381},
        {100, 31, 1.0, 0.5, true, 1.1386270808229184, 0.569319428609776},
        {262144, 31, 1.0, 0.5, false, 0.9773760172024913, 0.5428960499003426},
        {262144, 31, 1.0, 0.5, true, 1.132388374847845, 0.5816294434477498},
        {300000, 31, 1.0, -0.25, false, 1.011173995079255, -0.20006786767359366},
        {300000, 31, 1.0, -0.25, true, 1.1420745705075712, -0.27050412238900834},
        {999999, 22, 0.125, 0.5, false, -0.271007217613324, -0.4383834942165186},
        {999999, 22, 0.125, 0.5, true, 0.2622277521568631, -0.5249888638877341},
    };
    for (const Rotation& row : rotations) {
        const double inv = row.yarn ? flash_next_yarn_inv_freq(row.pair)
                                    : flash_next_default_inv_freq(row.pair);
        const auto got = flash_next_rotate_pair(row.x0, row.x1, row.position, inv, row.yarn ? factor : 1.0);
        if (!near(got.first, row.first, 1.0e-9) || !near(got.second, row.second, 1.0e-9)) {
            std::cerr << "FAIL: rotation pos " << row.position << " pair " << row.pair
                      << " got (" << got.first << ", " << got.second << ")\n";
            ++failures;
        }
    }

    constexpr std::int32_t blocks = 250000;
    constexpr std::int32_t chunk  = 2048;
    const std::int32_t tile       = flash_next_qsa_indexer_tile_size(blocks, chunk);
    expect(tile == 67, "1M prefill tile");
    const std::int64_t score_bytes =
        static_cast<std::int64_t>(blocks) * tile * static_cast<std::int64_t>(sizeof(float));
    expect(score_bytes <= static_cast<std::int64_t>(kMaxScoreWorkspaceBytes), "score plane cap");
    const std::int64_t tile_items = static_cast<std::int64_t>(blocks) * tile;
    const std::int64_t decode_items = static_cast<std::int64_t>(blocks) * 8;
    const std::int64_t uncapped = static_cast<std::int64_t>(blocks) * chunk;
    const std::int64_t int32_max = std::numeric_limits<std::int32_t>::max();
    expect(tile_items == 16'750'000 && tile_items <= int32_max, "tiled item count");
    expect(decode_items == 2'000'000 && decode_items <= int32_max, "decode item count");
    expect(uncapped == 512'000'000 && uncapped <= int32_max, "uncapped item count still fits int32");
    expect(999999 <= (1 << 24), "positions through 999999 are exact in float32");

    if (failures != 0) {
        std::cerr << failures << " long-context checks failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PASS: flash-next long context\n";
    return EXIT_SUCCESS;
}
