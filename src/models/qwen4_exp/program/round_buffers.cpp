#include "models/qwen4_exp/program/round_buffers.h"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ninfer::models::qwen4_exp {
namespace {

constexpr std::size_t kArenaAlign = 256;

TensorRegion add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

void validate_spec(const RoundStateSpec& spec) {
    if (spec.stream_hidden <= 0 || spec.output_rows <= 0) {
        throw std::invalid_argument("RoundState widths must be positive");
    }
    if (spec.batch_capacity == 0 || spec.batch_capacity > kMaximumConcurrency ||
        (spec.causal_scoring && spec.batch_capacity != 1)) {
        throw std::invalid_argument("RoundState batch capacity is outside its domain");
    }
}

void require_batch(std::int32_t batch, std::uint32_t capacity) {
    if (batch <= 0 || static_cast<std::uint32_t>(batch) > capacity) {
        throw std::invalid_argument("ordinary decode batch exceeds the round capacity");
    }
}

} // namespace

RoundStateLayout plan_round_state_layout(LayoutBuilder& builder, const RoundStateSpec& spec) {
    validate_spec(spec);
    RoundStateLayout layout;
    layout.spec = spec;
    if (!spec.causal_scoring) {
        OrdinaryDecodeStateLayout& ordinary = layout.ordinary.emplace();
        ordinary.ingress =
            builder.add(sizeof(OrdinaryDecodeIngress), 256, "ordinary decode ingress");
        ordinary.egress = builder.add(sizeof(OrdinaryDecodeEgress), 256, "ordinary decode egress");
        const auto batch = static_cast<std::int32_t>(spec.batch_capacity);
        ordinary.logits =
            add_tensor(builder, DType::BF16, {spec.output_rows, batch}, "ordinary decode logits");
        ordinary.hidden = add_tensor(builder, DType::BF16, {spec.stream_hidden, batch},
                                     "ordinary decode stream hidden");
        layout.token = add_tensor(builder, DType::I32, {1}, "step token");
        layout.pos   = add_tensor(builder, DType::I32, {1}, "step position");
    }
    // Prompt readouts tile their logits through this buffer; scoring projects its own tiles.
    layout.logits       = add_tensor(builder, DType::BF16, {spec.output_rows, 1}, "step logits");
    layout.prefill_tail = add_tensor(builder, DType::BF16, {spec.stream_hidden, 1},
                                     "prefill tail stream hidden");
    return layout;
}

OrdinaryDecodeState::OrdinaryDecodeState(DeviceSpan backing,
                                         const OrdinaryDecodeStateLayout& layout,
                                         std::uint32_t capacity)
    : batch_capacity(capacity) {
    if (capacity == 0 || capacity > kMaximumConcurrency) {
        throw std::invalid_argument("ordinary decode batch capacity must be in [1,8]");
    }
    static_assert(std::is_standard_layout_v<OrdinaryDecodeIngress>);
    static_assert(std::is_standard_layout_v<OrdinaryDecodeEgress>);
    static_assert(offsetof(OrdinaryDecodeIngress, ple_codes) % 16 == 0);
    static_assert(offsetof(OrdinaryDecodeIngress, ple_scales) % 16 == 0);
    ingress  = layout.ingress.bind(backing);
    egress   = layout.egress.bind(backing);
    sampling = reinterpret_cast<const ops::SamplingConfig*>(
        static_cast<const unsigned char*>(ingress.data) +
        offsetof(OrdinaryDecodeIngress, sampling));
    logits = layout.logits.bind(backing);
    hidden = layout.hidden.bind(backing);
}

#define NINFER_ORDINARY_I32_VIEW(name)                                                             \
    Tensor OrdinaryDecodeState::name(std::int32_t batch) const {                                   \
        require_batch(batch, batch_capacity);                                                      \
        return Tensor(static_cast<unsigned char*>(ingress.data) +                                  \
                          offsetof(OrdinaryDecodeIngress, name),                                   \
                      DType::I32, {batch});                                                        \
    }

NINFER_ORDINARY_I32_VIEW(tokens)
NINFER_ORDINARY_I32_VIEW(cache_positions)
NINFER_ORDINARY_I32_VIEW(text_kv_table_rows)
NINFER_ORDINARY_I32_VIEW(state_source_slots)
NINFER_ORDINARY_I32_VIEW(state_destination_slots)

#undef NINFER_ORDINARY_I32_VIEW

Tensor OrdinaryDecodeState::rope_positions(std::int32_t batch) const {
    require_batch(batch, batch_capacity);
    return Tensor(static_cast<unsigned char*>(ingress.data) +
                      offsetof(OrdinaryDecodeIngress, rope_positions),
                  DType::I32, {batch, 3});
}

Tensor OrdinaryDecodeState::ple_codes(std::int32_t batch) const {
    require_batch(batch, batch_capacity);
    return Tensor(static_cast<unsigned char*>(ingress.data) +
                      offsetof(OrdinaryDecodeIngress, ple_codes),
                  DType::U8,
                  {static_cast<std::int32_t>(kPleCodeRowBytes),
                   batch * static_cast<std::int32_t>(kPleHeads)});
}

Tensor OrdinaryDecodeState::ple_scales(std::int32_t batch) const {
    require_batch(batch, batch_capacity);
    return Tensor(static_cast<unsigned char*>(ingress.data) +
                      offsetof(OrdinaryDecodeIngress, ple_scales),
                  DType::FP16,
                  {static_cast<std::int32_t>(kPleScaleRowWords),
                   batch * static_cast<std::int32_t>(kPleHeads)});
}

Tensor OrdinaryDecodeState::sampled_tokens(std::int32_t batch) const {
    require_batch(batch, batch_capacity);
    return Tensor(static_cast<unsigned char*>(egress.data) +
                      offsetof(OrdinaryDecodeEgress, sampled_tokens),
                  DType::I32, {batch});
}

RoundState::RoundState(DeviceSpan backing, const RoundStateLayout& layout) {
    if (layout.ordinary) { ordinary.emplace(backing, *layout.ordinary, layout.spec.batch_capacity); }
    if (!layout.spec.causal_scoring) {
        token = layout.token.bind(backing);
        pos   = layout.pos.bind(backing);
    }
    logits       = layout.logits.bind(backing);
    prefill_tail = layout.prefill_tail.bind(backing);
}

} // namespace ninfer::models::qwen4_exp
