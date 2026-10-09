#pragma once

#include "core/layout.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::models::qwen4_exp {

// Gathered per-layer n-gram embedding rows of one token: 16 heads of 160 u4 codes (80 bytes) and
// ten FP16 scales each.
inline constexpr std::size_t kPleHeads         = 16;
inline constexpr std::size_t kPleCodeRowBytes  = 80;
inline constexpr std::size_t kPleScaleRowWords = 10;

struct RoundStateSpec {
    std::int32_t stream_hidden   = 0;
    std::int32_t output_rows     = 0;
    std::uint32_t batch_capacity = 1;
    bool causal_scoring          = false;
};

struct PrefillRoundHost {
    TokenId sampled_token = 0;
    ops::SamplingConfig sampling;
};

// Stable pinned/device transfer format for ordinary decode. The full fixed-size object is copied
// once per round; only its exact-B prefixes are consumed by the model schedule. rope_positions
// is planar [B,3] for the round's B (axis a of row b at a * B + b).
struct OrdinaryDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> tokens{};
    std::array<std::int32_t, kMaximumConcurrency> cache_positions{};
    std::array<std::int32_t, 3 * kMaximumConcurrency> rope_positions{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
    std::array<std::uint8_t, kMaximumConcurrency * kPleHeads * kPleCodeRowBytes> ple_codes{};
    std::array<std::uint16_t, kMaximumConcurrency * kPleHeads * kPleScaleRowWords> ple_scales{};
};

struct OrdinaryDecodeEgress {
    std::array<TokenId, kMaximumConcurrency> sampled_tokens{};
};

struct OrdinaryDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion logits;
    TensorRegion hidden;
};

struct RoundStateLayout {
    RoundStateSpec spec;
    std::optional<OrdinaryDecodeStateLayout> ordinary;
    TensorRegion token;
    TensorRegion pos;
    TensorRegion logits;
    TensorRegion prefill_tail;
};

// Device views of the ordinary ingress for the round's exact B.
struct OrdinaryDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    std::uint32_t batch_capacity        = 0;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor logits;
    Tensor hidden;

    OrdinaryDecodeState() = default;
    OrdinaryDecodeState(DeviceSpan backing, const OrdinaryDecodeStateLayout& layout,
                        std::uint32_t batch_capacity);

    [[nodiscard]] Tensor tokens(std::int32_t batch) const;
    [[nodiscard]] Tensor cache_positions(std::int32_t batch) const;
    [[nodiscard]] Tensor rope_positions(std::int32_t batch) const;
    [[nodiscard]] Tensor text_kv_table_rows(std::int32_t batch) const;
    [[nodiscard]] Tensor state_source_slots(std::int32_t batch) const;
    [[nodiscard]] Tensor state_destination_slots(std::int32_t batch) const;
    [[nodiscard]] Tensor ple_codes(std::int32_t batch) const;
    [[nodiscard]] Tensor ple_scales(std::int32_t batch) const;
    [[nodiscard]] Tensor sampled_tokens(std::int32_t batch) const;
};

[[nodiscard]] RoundStateLayout plan_round_state_layout(LayoutBuilder& builder,
                                                       const RoundStateSpec& spec);

struct RoundState {
    std::optional<OrdinaryDecodeState> ordinary;
    Tensor token;
    Tensor pos;
    Tensor logits;
    // Four-stream hidden of the last column of the latest prefill chunk.
    Tensor prefill_tail;

    RoundState() = default;
    RoundState(DeviceSpan backing, const RoundStateLayout& layout);
};

} // namespace ninfer::models::qwen4_exp
