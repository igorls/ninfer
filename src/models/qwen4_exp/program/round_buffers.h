#pragma once

#include "core/layout.h"
#include "core/tensor.h"
#include "models/qwen4_exp/config.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::models::qwen4_exp {

// Gathered PLE rows carried by a decode transfer: per PLE layer and token, `heads` u4z8_g16 rows
// of code_row_bytes codes and scale_row_bytes FP16 scales. A transfer of T tokens is planar by
// layer (execution::PleGather): its code plane holds layers x T x heads rows and its scale plane
// the matching scales, so each layer's rows are one ops::ple_ngram_decode operand.
struct PleRowGeometry {
    std::uint32_t layers        = 0;
    std::uint32_t heads         = 0;
    std::size_t code_row_bytes  = 0;
    std::size_t scale_row_bytes = 0;

    [[nodiscard]] std::size_t rows(std::size_t tokens) const noexcept {
        return std::size_t(layers) * heads * tokens;
    }
};

[[nodiscard]] PleRowGeometry ple_row_geometry(const TextConfig& config);

// Offsets of the PLE planes that follow a fixed control record in one transfer region.
struct PleTransferLayout {
    PleRowGeometry geometry;
    std::size_t token_capacity = 0;
    std::size_t codes_offset   = 0;
    std::size_t scales_offset  = 0;
    std::size_t bytes          = 0; // the whole region, control record included
};

[[nodiscard]] PleTransferLayout plan_ple_transfer(std::size_t control_bytes,
                                                  const PleRowGeometry& geometry,
                                                  std::size_t token_capacity);

// Device views of the first `tokens` planar tokens of a transfer region.
[[nodiscard]] Tensor ple_transfer_codes(const MemorySpan& region, const PleTransferLayout& layout,
                                        std::int32_t tokens);
[[nodiscard]] Tensor ple_transfer_scales(const MemorySpan& region, const PleTransferLayout& layout,
                                         std::int32_t tokens);

struct RoundStateSpec {
    std::int32_t stream_hidden   = 0;
    std::int32_t output_rows     = 0;
    std::uint32_t batch_capacity = 1;
    bool causal_scoring          = false;
    PleRowGeometry ple;
};

struct PrefillRoundHost {
    TokenId sampled_token = 0;
    ops::SamplingConfig sampling;
};

// Stable pinned/device transfer format for ordinary decode: this control record followed by the
// round's PLE planes (OrdinaryDecodeStateLayout::ple). The whole fixed-size region is copied once
// per round; only its exact-B prefixes are consumed by the model schedule. rope_positions is
// planar [B,3] for the round's B (axis a of row b at a * B + b), and the PLE rows are planar by
// layer for the round's B.
struct OrdinaryDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> tokens{};
    std::array<std::int32_t, kMaximumConcurrency> cache_positions{};
    std::array<std::int32_t, 3 * kMaximumConcurrency> rope_positions{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> state_source_slots{};
    std::array<std::int32_t, kMaximumConcurrency> state_destination_slots{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
};

struct OrdinaryDecodeEgress {
    std::array<TokenId, kMaximumConcurrency> sampled_tokens{};
};

struct OrdinaryDecodeStateLayout {
    LayoutRegion ingress; // OrdinaryDecodeIngress and the PLE planes, ple.bytes long
    PleTransferLayout ple;
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
    MemorySpan ingress;
    MemorySpan egress;
    PleTransferLayout ple;
    std::uint32_t batch_capacity        = 0;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor logits;
    Tensor hidden;

    OrdinaryDecodeState() = default;
    OrdinaryDecodeState(MemorySpan backing, const OrdinaryDecodeStateLayout& layout,
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
    RoundState(MemorySpan backing, const RoundStateLayout& layout);
};

} // namespace ninfer::models::qwen4_exp
