#include "models/qwen4_exp/state/state_image.h"

#include "core/device.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen4_exp {
namespace {

constexpr std::size_t kStateImageAlignment = 256;
constexpr std::int32_t kMropeAxes          = 3;

std::size_t checked_mul(std::size_t left, std::size_t right, const char* label) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(label);
    }
    return left * right;
}

std::size_t checked_add(std::size_t left, std::size_t right, const char* label) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::overflow_error(label);
    }
    return left + right;
}

bool same_linear_spec(const LinearAttentionStatePoolSpec& left,
                      const LinearAttentionStatePoolSpec& right) noexcept {
    return left.layers == right.layers && left.conv_channels == right.conv_channels &&
           left.conv_width == right.conv_width && left.value_heads == right.value_heads &&
           left.value_head_dim == right.value_head_dim && left.key_head_dim == right.key_head_dim &&
           left.slot_count == right.slot_count && left.conv_dtype == right.conv_dtype;
}

bool same_token_mixer_spec(const TokenMixerStateSpec& left,
                           const TokenMixerStateSpec& right) noexcept {
    return left.indexer_layers == right.indexer_layers && left.indexer_dim == right.indexer_dim &&
           left.indexer_block == right.indexer_block && left.ple_channels == right.ple_channels &&
           left.ple_history == right.ple_history && left.transient_slots == right.transient_slots;
}

bool same_region(const LayoutRegion& left, const LayoutRegion& right) noexcept {
    return left.offset == right.offset && left.bytes == right.bytes &&
           left.alignment == right.alignment;
}

bool same_host_layout(const StateImageHostLayout& left,
                      const StateImageHostLayout& right) noexcept {
    return same_linear_spec(left.spec.linear, right.spec.linear) &&
           same_token_mixer_spec(left.spec.token_mixer, right.spec.token_mixer) &&
           left.spec.hidden == right.spec.hidden &&
           same_region(left.linear_conv, right.linear_conv) &&
           left.linear_conv_layer_bytes == right.linear_conv_layer_bytes &&
           same_region(left.linear_recurrent, right.linear_recurrent) &&
           left.linear_recurrent_layer_bytes == right.linear_recurrent_layer_bytes &&
           same_region(left.indexer_keys, right.indexer_keys) &&
           left.indexer_keys_layer_bytes == right.indexer_keys_layer_bytes &&
           same_region(left.indexer_positions, right.indexer_positions) &&
           left.indexer_positions_layer_bytes == right.indexer_positions_layer_bytes &&
           same_region(left.ple_history, right.ple_history) &&
           same_region(left.continuation_hidden, right.continuation_hidden) &&
           same_region(left.continuation_positions, right.continuation_positions) &&
           left.image_bytes == right.image_bytes;
}

void validate_spec(const StateImageSpec& spec) {
    const auto& linear = spec.linear;
    const auto& mixer  = spec.token_mixer;
    if (linear.layers == 0 || linear.conv_channels <= 0 || linear.conv_width <= 0 ||
        linear.value_heads <= 0 || linear.value_head_dim <= 0 || linear.key_head_dim <= 0 ||
        linear.slot_count <= 0 || linear.conv_dtype != DType::BF16 || spec.hidden <= 0 ||
        mixer.indexer_layers == 0 || mixer.indexer_dim <= 0 || mixer.indexer_block <= 0 ||
        mixer.ple_channels <= 0 || mixer.ple_history <= 0 || mixer.transient_slots < 0 ||
        mixer.transient_slots > std::numeric_limits<std::int32_t>::max() - linear.slot_count) {
        throw std::invalid_argument("Qwen4Exp StateImage geometry is invalid");
    }
}

StateImageHostLayout plan_host_state_image(const StateImageSpec& spec) {
    validate_spec(spec);
    const auto& mixer = spec.token_mixer;
    LayoutBuilder builder;
    StateImageHostLayout host;
    host.spec = spec;
    const Tensor conv_slot(nullptr, spec.linear.conv_dtype,
                           {spec.linear.conv_channels, spec.linear.conv_width});
    const Tensor recurrent_slot(
        nullptr, DType::FP32,
        {spec.linear.key_head_dim, spec.linear.value_head_dim, spec.linear.value_heads});
    const Tensor keys_slot(nullptr, DType::BF16, {mixer.indexer_dim, mixer.indexer_block});
    const Tensor positions_slot(nullptr, DType::I32, {kMropeAxes, mixer.indexer_block});
    const Tensor ple_slot(nullptr, DType::BF16, {mixer.ple_channels, mixer.ple_history});
    const Tensor hidden_slot(nullptr, DType::BF16, {spec.hidden});

    host.linear_conv_layer_bytes = conv_slot.bytes();
    host.linear_conv = builder.add(checked_mul(host.linear_conv_layer_bytes, spec.linear.layers,
                                               "StateImage host convolution bytes overflow"),
                                   kStateImageAlignment, "StateImage host convolution");
    host.linear_recurrent_layer_bytes = recurrent_slot.bytes();
    host.linear_recurrent =
        builder.add(checked_mul(host.linear_recurrent_layer_bytes, spec.linear.layers,
                                "StateImage host recurrent bytes overflow"),
                    kStateImageAlignment, "StateImage host recurrent");
    host.indexer_keys_layer_bytes = keys_slot.bytes();
    host.indexer_keys = builder.add(checked_mul(host.indexer_keys_layer_bytes, mixer.indexer_layers,
                                                "StateImage host indexer keys bytes overflow"),
                                    kStateImageAlignment, "StateImage host indexer keys");
    host.indexer_positions_layer_bytes = positions_slot.bytes();
    host.indexer_positions =
        builder.add(checked_mul(host.indexer_positions_layer_bytes, mixer.indexer_layers,
                                "StateImage host indexer positions bytes overflow"),
                    kStateImageAlignment, "StateImage host indexer positions");
    host.ple_history =
        builder.add(ple_slot.bytes(), kStateImageAlignment, "StateImage host PLE history");
    host.continuation_hidden    = builder.add(hidden_slot.bytes(), kStateImageAlignment,
                                              "StateImage host continuation hidden");
    host.continuation_positions = builder.add(3 * sizeof(std::int32_t), kStateImageAlignment,
                                              "StateImage host continuation positions");
    host.image_bytes            = builder.finish(kStateImageAlignment, "StateImage host image");
    return host;
}

std::byte* byte_offset(std::byte* base, std::size_t offset) noexcept { return base + offset; }

const std::byte* byte_offset(const std::byte* base, std::size_t offset) noexcept {
    return base + offset;
}

void validate_slot(std::int32_t slot, std::int32_t slot_count, const char* label) {
    if (slot < 0 || slot >= slot_count) { throw std::out_of_range(label); }
}

// One slot of a [..., slots] tensor whose leading extents are contiguous.
Tensor slot_of(const Tensor& tensor, int axis, std::int32_t slot) {
    return tensor.slice(axis, slot, 1);
}

void copy_device(const Tensor& destination, const Tensor& source, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, destination.bytes(),
                               cudaMemcpyDeviceToDevice, stream));
}

} // namespace

StateImageDeviceLayout plan_state_image_device_pool(LayoutBuilder& builder,
                                                    const StateImageSpec& spec) {
    validate_spec(spec);
    const auto& mixer = spec.token_mixer;
    StateImageDeviceLayout out;
    out.linear = plan_linear_attention_state_pool(builder, spec.linear);
    out.indexer_keys.reserve(mixer.indexer_layers);
    out.indexer_positions.reserve(mixer.indexer_layers);
    for (std::uint32_t layer = 0; layer < mixer.indexer_layers; ++layer) {
        out.indexer_keys.push_back(
            builder.add_tensor(DType::BF16,
                               {mixer.indexer_dim, mixer.indexer_block,
                                spec.linear.slot_count + mixer.transient_slots},
                               kStateImageAlignment, "StateImage indexer forming keys"));
        out.indexer_positions.push_back(builder.add_tensor(
            DType::I32,
            {kMropeAxes, mixer.indexer_block, spec.linear.slot_count + mixer.transient_slots},
            kStateImageAlignment, "StateImage indexer forming positions"));
    }
    out.ple_history = builder.add_tensor(
        DType::BF16,
        {mixer.ple_channels, mixer.ple_history, spec.linear.slot_count + mixer.transient_slots},
        kStateImageAlignment, "StateImage PLE history");
    out.continuation_hidden =
        builder.add_tensor(DType::BF16, {spec.hidden, spec.linear.slot_count}, kStateImageAlignment,
                           "StateImage continuation hidden");
    out.continuation_positions =
        builder.add_tensor(DType::I32, {3, spec.linear.slot_count}, kStateImageAlignment,
                           "StateImage continuation positions");
    out.host = plan_host_state_image(spec);
    return out;
}

TransferWork state_image_transfer_work(const StateImageHostLayout& layout) {
    std::size_t payload = checked_add(layout.linear_conv.bytes, layout.linear_recurrent.bytes,
                                      "StateImage transfer payload overflow");
    payload =
        checked_add(payload, layout.indexer_keys.bytes, "StateImage transfer payload overflow");
    payload = checked_add(payload, layout.indexer_positions.bytes,
                          "StateImage transfer payload overflow");
    payload =
        checked_add(payload, layout.ple_history.bytes, "StateImage transfer payload overflow");
    payload = checked_add(payload, layout.continuation_hidden.bytes,
                          "StateImage transfer payload overflow");
    payload = checked_add(payload, layout.continuation_positions.bytes,
                          "StateImage transfer payload overflow");
    const std::uint64_t operations =
        2ULL * layout.spec.linear.layers + 2ULL * layout.spec.token_mixer.indexer_layers + 3ULL;
    if (operations > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("StateImage transfer operation count exceeds uint32");
    }
    return TransferWork{.payload_bytes   = static_cast<std::uint64_t>(payload),
                        .copy_operations = static_cast<std::uint32_t>(operations)};
}

HostStatePool::HostStatePool(HostContextArena& arena, StateImageHostLayout layout)
    : layout_(std::move(layout)), arena_(&arena) {
    if (!same_host_layout(layout_, plan_host_state_image(layout_.spec))) {
        throw std::invalid_argument("HostStatePool image layout is invalid");
    }
    if (layout_.image_bytes < arena_->minimum_allocation_bytes()) {
        throw std::invalid_argument("Host state geometry is below the shared arena minimum");
    }
    const std::size_t maximum_slots = arena_->capacity_bytes() / layout_.image_bytes;
    if (maximum_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Host state descriptor capacity exceeds uint32");
    }
    const auto capacity = static_cast<std::uint32_t>(maximum_slots);
    slots_.resize(capacity);
    free_slots_.resize(capacity);
    free_count_ = capacity;
    for (std::uint32_t index = 0; index < capacity; ++index) {
        free_slots_[index] = capacity - 1U - index;
    }
}

std::optional<HostStateSlotHandle> HostStatePool::allocate() noexcept {
    if (free_count_ == 0) { return std::nullopt; }
    auto storage = arena_->allocate(layout_.image_bytes);
    if (!storage) { return std::nullopt; }
    const std::uint32_t index = free_slots_[--free_count_];
    Slot& slot                = slots_[index];
    slot.storage              = std::move(*storage);
    ++occupied_;
    return HostStateSlotHandle{.index = index, .generation = slot.generation, .owner = this};
}

bool HostStatePool::can_allocate() const noexcept {
    return free_count_ != 0 && arena_->can_allocate(layout_.image_bytes);
}

bool HostStatePool::publish(HostStateSlotHandle handle) noexcept {
    if (!valid(handle)) { return false; }
    slots_[handle.index].storage.publish();
    return true;
}

bool HostStatePool::release(HostStateSlotHandle handle) noexcept {
    if (!valid(handle)) { return false; }
    Slot& slot = slots_[handle.index];
    (void)slot.storage.release();
    if (++slot.generation == 0) { ++slot.generation; }
    free_slots_[free_count_++] = handle.index;
    --occupied_;
    return true;
}

HostStateImageView HostStatePool::writable_view(HostStateSlotHandle handle) {
    if (!valid(handle)) { throw std::invalid_argument("HostStatePool handle is stale"); }
    return {.data = slot_data(handle.index), .layout = &layout_};
}

HostStateImageConstView HostStatePool::view(HostStateSlotHandle handle) const {
    if (!valid(handle)) { throw std::invalid_argument("HostStatePool handle is stale"); }
    return {.data = slot_data(handle.index), .layout = &layout_};
}

std::uint32_t HostStatePool::capacity() const noexcept {
    return static_cast<std::uint32_t>(slots_.size());
}

bool HostStatePool::valid(HostStateSlotHandle handle) const noexcept {
    return handle.owner == this && handle.index < slots_.size() &&
           slots_[handle.index].storage.valid() &&
           slots_[handle.index].generation == handle.generation;
}

std::byte* HostStatePool::slot_data(std::uint32_t index) const noexcept {
    return slots_[index].storage.data();
}

StateImageDevicePool::StateImageDevicePool(DeviceSpan backing, const StateImageDeviceLayout& layout)
    : linear_(backing, layout.linear), ple_history_(layout.ple_history.bind(backing)),
      continuation_hidden_(layout.continuation_hidden.bind(backing)),
      continuation_positions_(layout.continuation_positions.bind(backing)),
      host_layout_(layout.host) {
    const StateImageSpec& spec = host_layout_.spec;
    if (layout.indexer_keys.size() != spec.token_mixer.indexer_layers ||
        layout.indexer_positions.size() != spec.token_mixer.indexer_layers) {
        throw std::invalid_argument("StateImage indexer layout is inconsistent");
    }
    indexer_keys_.reserve(layout.indexer_keys.size());
    indexer_positions_.reserve(layout.indexer_positions.size());
    for (std::size_t layer = 0; layer < layout.indexer_keys.size(); ++layer) {
        indexer_keys_.push_back(layout.indexer_keys[layer].bind(backing));
        indexer_positions_.push_back(layout.indexer_positions[layer].bind(backing));
        if (indexer_keys_.back().ne[2] != linear_.slot_count() + spec.token_mixer.transient_slots ||
            indexer_positions_.back().ne[2] !=
                linear_.slot_count() + spec.token_mixer.transient_slots) {
            throw std::invalid_argument("StateImage components do not share one slot geometry");
        }
    }
    if (continuation_hidden_.dtype != DType::BF16 || !continuation_hidden_.is_contiguous() ||
        continuation_hidden_.ne[0] != spec.hidden ||
        continuation_hidden_.ne[1] != linear_.slot_count() ||
        ple_history_.ne[2] != linear_.slot_count() + spec.token_mixer.transient_slots ||
        continuation_positions_.dtype != DType::I32 || continuation_positions_.ne[0] != 3 ||
        continuation_positions_.ne[1] != linear_.slot_count()) {
        throw std::invalid_argument("StateImage continuation layout is inconsistent");
    }
    StateImageSpec device_spec = spec;
    device_spec.linear         = layout.linear.spec;
    if (!same_host_layout(host_layout_, plan_host_state_image(device_spec))) {
        throw std::invalid_argument("StateImage host layout does not match its device components");
    }
}

Tensor StateImageDevicePool::continuation_hidden_slot(std::int32_t slot) const {
    validate_slot(slot, slot_count(), "StateImage slot is out of range");
    return continuation_hidden_.slice(1, slot, 1).view({host_layout_.spec.hidden});
}

Tensor StateImageDevicePool::continuation_positions_slot(std::int32_t slot) const {
    validate_slot(slot, slot_count(), "StateImage positions slot is out of range");
    return continuation_positions_.slice(1, slot, 1).view({3});
}

void StateImageDevicePool::commit_token_mixer(std::int32_t snapshot, std::int32_t destination,
                                              cudaStream_t stream) {
    validate_slot(snapshot, ple_history_.ne[2], "token mixer snapshot is out of range");
    validate_slot(destination, slot_count(), "token mixer destination is out of range");
    for (std::size_t layer = 0; layer < indexer_keys_.size(); ++layer) {
        copy_device(slot_of(indexer_keys_[layer], 2, destination),
                    slot_of(indexer_keys_[layer], 2, snapshot), stream);
        copy_device(slot_of(indexer_positions_[layer], 2, destination),
                    slot_of(indexer_positions_[layer], 2, snapshot), stream);
    }
    copy_device(slot_of(ple_history_, 2, destination), slot_of(ple_history_, 2, snapshot), stream);
}

IndexerStateView StateImageDevicePool::indexer(std::uint32_t layer) const {
    if (layer >= indexer_keys_.size()) {
        throw std::out_of_range("StateImage indexer layer is out of range");
    }
    return {.raw_keys = indexer_keys_[layer], .raw_positions = indexer_positions_[layer]};
}

Tensor StateImageDevicePool::ple_history_slot(std::int32_t slot) const {
    validate_slot(slot, slot_count(), "StateImage PLE slot is out of range");
    return slot_of(ple_history_, 2, slot)
        .view({host_layout_.spec.token_mixer.ple_channels,
               host_layout_.spec.token_mixer.ple_history});
}

void StateImageDevicePool::zero_slot(std::int32_t slot, cudaStream_t stream) {
    validate_slot(slot, slot_count(), "StateImage zero slot is out of range");
    linear_.zero_slot(slot, stream);
    for (std::size_t layer = 0; layer < indexer_keys_.size(); ++layer) {
        const Tensor keys      = slot_of(indexer_keys_[layer], 2, slot);
        const Tensor positions = slot_of(indexer_positions_[layer], 2, slot);
        CUDA_CHECK(cudaMemsetAsync(keys.data, 0, keys.bytes(), stream));
        CUDA_CHECK(cudaMemsetAsync(positions.data, 0, positions.bytes(), stream));
    }
    const Tensor history = ple_history_slot(slot);
    CUDA_CHECK(cudaMemsetAsync(history.data, 0, history.bytes(), stream));
    const Tensor hidden = continuation_hidden_slot(slot);
    CUDA_CHECK(cudaMemsetAsync(hidden.data, 0, hidden.bytes(), stream));
    const Tensor positions = continuation_positions_slot(slot);
    CUDA_CHECK(cudaMemsetAsync(positions.data, 0, positions.bytes(), stream));
}

void StateImageDevicePool::zero_all(cudaStream_t stream) {
    linear_.zero_all(stream);
    for (std::size_t layer = 0; layer < indexer_keys_.size(); ++layer) {
        CUDA_CHECK(
            cudaMemsetAsync(indexer_keys_[layer].data, 0, indexer_keys_[layer].bytes(), stream));
        CUDA_CHECK(cudaMemsetAsync(indexer_positions_[layer].data, 0,
                                   indexer_positions_[layer].bytes(), stream));
    }
    CUDA_CHECK(cudaMemsetAsync(ple_history_.data, 0, ple_history_.bytes(), stream));
    CUDA_CHECK(cudaMemsetAsync(continuation_hidden_.data, 0, continuation_hidden_.bytes(), stream));
    CUDA_CHECK(
        cudaMemsetAsync(continuation_positions_.data, 0, continuation_positions_.bytes(), stream));
}

void StateImageDevicePool::copy_slot(std::int32_t source, std::int32_t destination,
                                     cudaStream_t stream) {
    validate_slot(source, slot_count(), "StateImage copy source is out of range");
    validate_slot(destination, slot_count(), "StateImage copy destination is out of range");
    if (source == destination) { return; }
    linear_.copy_slot(source, destination, stream);
    for (std::size_t layer = 0; layer < indexer_keys_.size(); ++layer) {
        copy_device(slot_of(indexer_keys_[layer], 2, destination),
                    slot_of(indexer_keys_[layer], 2, source), stream);
        copy_device(slot_of(indexer_positions_[layer], 2, destination),
                    slot_of(indexer_positions_[layer], 2, source), stream);
    }
    copy_device(ple_history_slot(destination), ple_history_slot(source), stream);
    copy_device(continuation_hidden_slot(destination), continuation_hidden_slot(source), stream);
    copy_device(continuation_positions_slot(destination), continuation_positions_slot(source),
                stream);
}

void StateImageDevicePool::validate_host_layout(const StateImageHostLayout* layout,
                                                const std::byte* data) const {
    if (layout == nullptr || data == nullptr || !same_host_layout(*layout, host_layout_)) {
        throw std::invalid_argument("Host and device StateImage layouts do not match");
    }
}

void StateImageDevicePool::copy_to_host(std::int32_t source, HostStateImageView destination,
                                        cudaStream_t stream) const {
    validate_slot(source, slot_count(), "StateImage copy-to-host source is out of range");
    validate_host_layout(destination.layout, destination.data);
    const auto to_host = [&](std::size_t offset, const Tensor& tensor) {
        CUDA_CHECK(cudaMemcpyAsync(byte_offset(destination.data, offset), tensor.data,
                                   tensor.bytes(), cudaMemcpyDeviceToHost, stream));
    };
    for (std::uint32_t layer = 0; layer < linear_.layer_count(); ++layer) {
        to_host(host_layout_.linear_conv.offset + layer * host_layout_.linear_conv_layer_bytes,
                linear_.conv_slot(layer, source));
        to_host(host_layout_.linear_recurrent.offset +
                    layer * host_layout_.linear_recurrent_layer_bytes,
                linear_.recurrent_slot(layer, source));
    }
    for (std::size_t layer = 0; layer < indexer_keys_.size(); ++layer) {
        to_host(host_layout_.indexer_keys.offset + layer * host_layout_.indexer_keys_layer_bytes,
                slot_of(indexer_keys_[layer], 2, source));
        to_host(host_layout_.indexer_positions.offset +
                    layer * host_layout_.indexer_positions_layer_bytes,
                slot_of(indexer_positions_[layer], 2, source));
    }
    to_host(host_layout_.ple_history.offset, ple_history_slot(source));
    to_host(host_layout_.continuation_hidden.offset, continuation_hidden_slot(source));
    to_host(host_layout_.continuation_positions.offset, continuation_positions_slot(source));
}

void StateImageDevicePool::copy_from_host(HostStateImageConstView source, std::int32_t destination,
                                          cudaStream_t stream) {
    validate_slot(destination, slot_count(),
                  "StateImage copy-from-host destination is out of range");
    validate_host_layout(source.layout, source.data);
    const auto from_host = [&](const Tensor& tensor, std::size_t offset) {
        CUDA_CHECK(cudaMemcpyAsync(tensor.data, byte_offset(source.data, offset), tensor.bytes(),
                                   cudaMemcpyHostToDevice, stream));
    };
    for (std::uint32_t layer = 0; layer < linear_.layer_count(); ++layer) {
        from_host(linear_.conv_slot(layer, destination),
                  host_layout_.linear_conv.offset + layer * host_layout_.linear_conv_layer_bytes);
        from_host(linear_.recurrent_slot(layer, destination),
                  host_layout_.linear_recurrent.offset +
                      layer * host_layout_.linear_recurrent_layer_bytes);
    }
    for (std::size_t layer = 0; layer < indexer_keys_.size(); ++layer) {
        from_host(slot_of(indexer_keys_[layer], 2, destination),
                  host_layout_.indexer_keys.offset + layer * host_layout_.indexer_keys_layer_bytes);
        from_host(slot_of(indexer_positions_[layer], 2, destination),
                  host_layout_.indexer_positions.offset +
                      layer * host_layout_.indexer_positions_layer_bytes);
    }
    from_host(ple_history_slot(destination), host_layout_.ple_history.offset);
    from_host(continuation_hidden_slot(destination), host_layout_.continuation_hidden.offset);
    from_host(continuation_positions_slot(destination), host_layout_.continuation_positions.offset);
}

} // namespace ninfer::models::qwen4_exp
