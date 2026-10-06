#pragma once
#include "core/gdn_replay_records.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/program/round_buffers.h"
#include "ninfer/ops/gdn_replay.h"

namespace ninfer::models::qwen4_exp {
inline constexpr std::int32_t kMtpMaximumDrafts = 5;
inline constexpr std::int32_t kMtpMaximumWidth  = kMtpMaximumDrafts + 1;
inline constexpr std::size_t kMtpMaximumColumns = kMaximumConcurrency * kMtpMaximumWidth;

struct MtpIngress {
    alignas(16) std::array<std::int32_t, kMtpMaximumColumns> ids{}, positions{}, rows{}, mtp_rows{};
    alignas(16) std::array<std::int32_t, 3 * kMtpMaximumColumns> rope{};
    alignas(16) std::array<std::int32_t, kMtpMaximumColumns> teacher_positions{};
    alignas(
        16) std::array<std::int32_t, kMaximumConcurrency * kMtpMaximumDrafts> proposal_positions{};
    alignas(
        16) std::array<std::int32_t, 3 * kMaximumConcurrency * kMtpMaximumDrafts> proposal_rope{};
    alignas(16) std::array<std::int32_t, kMaximumConcurrency> sequence_rows{}, mtp_sequence_rows{};
    alignas(16) std::array<std::int32_t, kMaximumConcurrency> sources{},
        destinations{}, snapshots{};
    alignas(16) std::array<std::int32_t, kMaximumConcurrency> frontiers{}, anchors{}, extents{},
        valid{};
    alignas(16) std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
    alignas(16) std::array<std::uint8_t, kMtpMaximumColumns * kPleHeads * kPleCodeRowBytes> codes{};
    alignas(
        16) std::array<std::uint16_t, kMtpMaximumColumns * kPleHeads * kPleScaleRowWords> scales{};
};

struct MtpEgress {
    std::array<std::int32_t, kMaximumConcurrency * kMtpMaximumDrafts> drafts{};
    std::array<std::int32_t, kMtpMaximumColumns> licensed{};
    std::array<std::int32_t, kMaximumConcurrency> counts{}, accepted{};
};

struct MtpFrameLayout {
    LayoutRegion ingress;
    TensorRegion hidden, logits, argmax, drafts, licensed, counts, accepted;
    TensorRegion previous, teacher_output, teacher_positions, teacher_rope;
    TensorRegion ar_hidden, next_hidden, embedding, proposal_logits;
    TensorRegion proposal_ids, proposal_positions, proposal_rope, proposal_sources;
    GdnReplayRecordLayout records;
    std::int32_t width = 0, batch_capacity = 0;
    std::size_t bytes = 0;
};

MtpFrameLayout plan_mtp_frame(const TextConfig& config, std::int32_t width, std::int32_t batch);

// Planned for the startup maximum verification width. A round may use any narrower width: its
// width-shaped tensors and replay records are repacked densely from the same bases. Per-row valid
// extents bound mutations; records and token-mixer snapshots remain live until the whole pending
// round is committed or cancelled.
struct MtpFrame {
    DeviceSpan ingress;
    Tensor hidden, logits, argmax, drafts, licensed, counts, accepted;
    Tensor previous, teacher_output, teacher_positions, teacher_rope;
    Tensor ar_hidden, next_hidden, embedding, proposal_logits;
    Tensor proposal_ids, proposal_positions, proposal_rope, proposal_sources;
    GdnReplayRecords records;
    std::int32_t width = 0, batch_capacity = 0;
    MtpFrame(DeviceSpan backing, const MtpFrameLayout& layout);
    [[nodiscard]] MtpFrame batch(std::int32_t count) const { return round(count, width); }
    [[nodiscard]] MtpFrame round(std::int32_t count, std::int32_t round_width) const;
    [[nodiscard]] Tensor i32(std::size_t offset, std::int32_t count) const;
    [[nodiscard]] const ops::SamplingConfig* sampling() const;
};
} // namespace ninfer::models::qwen4_exp
