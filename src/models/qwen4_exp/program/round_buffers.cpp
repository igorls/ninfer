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

constexpr std::size_t kTransferAlign = 16;

std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::overflow_error("PLE transfer size overflows");
    }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("PLE transfer size overflows");
    }
    return a * b;
}

std::size_t align_transfer(std::size_t value) {
    return checked_add(value, kTransferAlign - 1) / kTransferAlign * kTransferAlign;
}

std::int32_t transfer_extent(std::size_t value) {
    if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("PLE transfer extent is outside the Tensor domain");
    }
    return static_cast<std::int32_t>(value);
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

PleRowGeometry ple_row_geometry(const TextConfig& config) {
    return {.layers          = static_cast<std::uint32_t>(config.ple.layers.size()),
            .heads           = config.ple.heads(),
            .code_row_bytes  = config.ple.code_row_bytes(),
            .scale_row_bytes = config.ple.scale_row_bytes()};
}

PleTransferLayout plan_ple_transfer(std::size_t control_bytes, const PleRowGeometry& geometry,
                                    std::size_t token_capacity) {
    if (geometry.layers == 0 || geometry.heads == 0 || geometry.code_row_bytes == 0 ||
        geometry.scale_row_bytes == 0 || geometry.scale_row_bytes % sizeof(std::uint16_t) != 0 ||
        token_capacity == 0) {
        throw std::invalid_argument("PLE transfer geometry is invalid");
    }
    PleTransferLayout out;
    out.geometry       = geometry;
    out.token_capacity = token_capacity;
    const std::size_t rows =
        checked_mul(checked_mul(geometry.layers, geometry.heads), token_capacity);
    out.codes_offset = align_transfer(control_bytes);
    out.scales_offset =
        align_transfer(checked_add(out.codes_offset, checked_mul(rows, geometry.code_row_bytes)));
    out.bytes =
        align_transfer(checked_add(out.scales_offset, checked_mul(rows, geometry.scale_row_bytes)));
    return out;
}

Tensor ple_transfer_codes(const MemorySpan& region, const PleTransferLayout& layout,
                          std::int32_t tokens) {
    if (tokens <= 0 || static_cast<std::size_t>(tokens) > layout.token_capacity ||
        region.bytes < layout.bytes) {
        throw std::invalid_argument("PLE transfer tokens exceed the region capacity");
    }
    return Tensor(static_cast<std::byte*>(region.data) + layout.codes_offset, region.device, DType::U8,
                  {transfer_extent(layout.geometry.code_row_bytes),
                   transfer_extent(layout.geometry.rows(static_cast<std::size_t>(tokens)))});
}

Tensor ple_transfer_scales(const MemorySpan& region, const PleTransferLayout& layout,
                           std::int32_t tokens) {
    if (tokens <= 0 || static_cast<std::size_t>(tokens) > layout.token_capacity ||
        region.bytes < layout.bytes) {
        throw std::invalid_argument("PLE transfer tokens exceed the region capacity");
    }
    return Tensor(static_cast<std::byte*>(region.data) + layout.scales_offset, region.device, DType::FP16,
                  {transfer_extent(layout.geometry.scale_row_bytes / sizeof(std::uint16_t)),
                   transfer_extent(layout.geometry.rows(static_cast<std::size_t>(tokens)))});
}

RoundStateLayout plan_round_state_layout(LayoutBuilder& builder, const RoundStateSpec& spec) {
    validate_spec(spec);
    RoundStateLayout layout;
    layout.spec = spec;
    if (!spec.causal_scoring) {
        OrdinaryDecodeStateLayout& ordinary = layout.ordinary.emplace();
        ordinary.ple =
            plan_ple_transfer(sizeof(OrdinaryDecodeIngress), spec.ple, spec.batch_capacity);
        ordinary.ingress = builder.add(ordinary.ple.bytes, 256, "ordinary decode ingress");
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

OrdinaryDecodeState::OrdinaryDecodeState(MemorySpan backing,
                                         const OrdinaryDecodeStateLayout& layout,
                                         std::uint32_t capacity)
    : ple(layout.ple), batch_capacity(capacity) {
    if (capacity == 0 || capacity > kMaximumConcurrency || capacity > layout.ple.token_capacity) {
        throw std::invalid_argument("ordinary decode batch capacity must be in [1,8]");
    }
    static_assert(std::is_standard_layout_v<OrdinaryDecodeIngress>);
    static_assert(std::is_standard_layout_v<OrdinaryDecodeEgress>);
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
                      ingress.device, DType::I32, {batch});                                        \
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
                      offsetof(OrdinaryDecodeIngress, rope_positions), ingress.device,
                  DType::I32, {batch, 3});
}

Tensor OrdinaryDecodeState::ple_codes(std::int32_t batch) const {
    require_batch(batch, batch_capacity);
    return ple_transfer_codes(ingress, ple, batch);
}

Tensor OrdinaryDecodeState::ple_scales(std::int32_t batch) const {
    require_batch(batch, batch_capacity);
    return ple_transfer_scales(ingress, ple, batch);
}

Tensor OrdinaryDecodeState::sampled_tokens(std::int32_t batch) const {
    require_batch(batch, batch_capacity);
    return Tensor(static_cast<unsigned char*>(egress.data) +
                      offsetof(OrdinaryDecodeEgress, sampled_tokens), egress.device,
                  DType::I32, {batch});
}

RoundState::RoundState(MemorySpan backing, const RoundStateLayout& layout) {
    if (layout.ordinary) { ordinary.emplace(backing, *layout.ordinary, layout.spec.batch_capacity); }
    if (!layout.spec.causal_scoring) {
        token = layout.token.bind(backing);
        pos   = layout.pos.bind(backing);
    }
    logits       = layout.logits.bind(backing);
    prefill_tail = layout.prefill_tail.bind(backing);
}

} // namespace ninfer::models::qwen4_exp
