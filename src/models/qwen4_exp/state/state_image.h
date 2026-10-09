#pragma once

#include "core/arena.h"
#include "core/host_context_arena.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "core/transfer_work.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::models::qwen4_exp {

// Per-sequence state of the QSA indexer and the per-layer n-gram embedding that lives beside the
// GDN state: the raw keys and MRoPE positions of each attention layer's forming four-token block
// (ops::QsaIndexerKeyState) and the nine BF16 history columns of the PLE dilated convolution.
struct TokenMixerStateSpec {
    std::uint32_t indexer_layers = 0;
    std::int32_t indexer_dim     = 0;
    std::int32_t indexer_block   = 0;
    std::int32_t ple_channels    = 0;
    std::int32_t ple_history     = 0;
    // Extra QSA/PLE slots for a pending verification block; never cache-owned.
    std::int32_t transient_slots = 0;
};

struct StateImageSpec {
    LinearAttentionStatePoolSpec linear;
    TokenMixerStateSpec token_mixer;
    // The four-stream residual of the last executed position (before the final mixer).
    std::int32_t hidden = 0;
};

struct StateImageHostLayout {
    StateImageSpec spec;
    LayoutRegion linear_conv;
    std::size_t linear_conv_layer_bytes = 0;
    LayoutRegion linear_recurrent;
    std::size_t linear_recurrent_layer_bytes = 0;
    LayoutRegion indexer_keys;
    std::size_t indexer_keys_layer_bytes = 0;
    LayoutRegion indexer_positions;
    std::size_t indexer_positions_layer_bytes = 0;
    LayoutRegion ple_history;
    LayoutRegion continuation_hidden;
    LayoutRegion continuation_positions;
    std::size_t image_bytes = 0;
};

struct StateImageDeviceLayout {
    LinearAttentionStatePoolLayout linear;
    std::vector<TensorRegion> indexer_keys;      // BF16 [dim, block, slots] per attention layer
    std::vector<TensorRegion> indexer_positions; // I32 [3, block, slots] per attention layer
    TensorRegion ple_history;                    // BF16 [channels, history, slots]
    TensorRegion continuation_hidden;            // BF16 [hidden, slots]
    TensorRegion continuation_positions;         // I32 [3, slots]
    StateImageHostLayout host;
};

[[nodiscard]] TransferWork state_image_transfer_work(const StateImageHostLayout& layout);

[[nodiscard]] StateImageDeviceLayout plan_state_image_device_pool(LayoutBuilder& builder,
                                                                  const StateImageSpec& spec);

struct HostStateImageView {
    std::byte* data                    = nullptr;
    const StateImageHostLayout* layout = nullptr;
};

struct HostStateImageConstView {
    const std::byte* data              = nullptr;
    const StateImageHostLayout* layout = nullptr;
};

class HostStatePool;

struct HostStateSlotHandle {
    std::uint32_t index        = 0;
    std::uint32_t generation   = 0;
    const HostStatePool* owner = nullptr;
};

/** Typed StateImage descriptors backed on demand by the shared Host arena; owns no cache policy.
 */
class HostStatePool {
public:
    HostStatePool(HostContextArena& arena, StateImageHostLayout layout);

    HostStatePool(const HostStatePool&)            = delete;
    HostStatePool& operator=(const HostStatePool&) = delete;
    HostStatePool(HostStatePool&&)                 = delete;
    HostStatePool& operator=(HostStatePool&&)      = delete;

    [[nodiscard]] std::optional<HostStateSlotHandle> allocate() noexcept;
    [[nodiscard]] bool can_allocate() const noexcept;
    [[nodiscard]] bool publish(HostStateSlotHandle handle) noexcept;
    [[nodiscard]] bool release(HostStateSlotHandle handle) noexcept;

    [[nodiscard]] HostStateImageView writable_view(HostStateSlotHandle handle);
    [[nodiscard]] HostStateImageConstView view(HostStateSlotHandle handle) const;

    [[nodiscard]] std::uint32_t capacity() const noexcept;

    [[nodiscard]] std::uint32_t occupied() const noexcept { return occupied_; }

    [[nodiscard]] std::size_t occupied_bytes() const noexcept {
        return static_cast<std::size_t>(occupied_) * layout_.image_bytes;
    }

    [[nodiscard]] const StateImageHostLayout& layout() const noexcept { return layout_; }

private:
    struct Slot {
        HostContextAllocation storage;
        std::uint32_t generation = 1;
    };

    [[nodiscard]] bool valid(HostStateSlotHandle handle) const noexcept;
    [[nodiscard]] std::byte* slot_data(std::uint32_t index) const noexcept;

    StateImageHostLayout layout_;
    HostContextArena* arena_ = nullptr;
    std::vector<Slot> slots_;
    std::vector<std::uint32_t> free_slots_;
    std::uint32_t free_count_ = 0;
    std::uint32_t occupied_   = 0;
};

// The QSA indexer forming-block state of one attention layer across every slot, the form
// ops::qsa_indexer_append consumes.
struct IndexerStateView {
    Tensor raw_keys;
    Tensor raw_positions;
};

/**
 * Caller-backed fixed storage for Qwen4Exp continuation state.
 *
 * Every absolute slot holds the GDN convolution and recurrent state, the QSA indexer forming
 * blocks, the PLE convolution history and the continuation hidden of one frontier. The pool owns
 * neither slot roles nor logical checkpoint identity.
 */
class StateImageDevicePool {
public:
    StateImageDevicePool(DeviceSpan backing, const StateImageDeviceLayout& layout);

    StateImageDevicePool(const StateImageDevicePool&)            = delete;
    StateImageDevicePool& operator=(const StateImageDevicePool&) = delete;
    StateImageDevicePool(StateImageDevicePool&&)                 = delete;
    StateImageDevicePool& operator=(StateImageDevicePool&&)      = delete;

    [[nodiscard]] std::int32_t slot_count() const noexcept { return linear_.slot_count(); }

    [[nodiscard]] Tensor continuation_hidden_slot(std::int32_t slot) const;
    [[nodiscard]] Tensor continuation_positions_slot(std::int32_t slot) const;

    // Publish just the token-mixer part of one transient verification snapshot.
    void commit_token_mixer(std::int32_t snapshot, std::int32_t destination, cudaStream_t stream);

    [[nodiscard]] LinearAttentionStatePool& linear() noexcept { return linear_; }

    [[nodiscard]] const LinearAttentionStatePool& linear() const noexcept { return linear_; }

    [[nodiscard]] IndexerStateView indexer(std::uint32_t layer) const;

    [[nodiscard]] const Tensor& ple_history() const noexcept { return ple_history_; }

    [[nodiscard]] Tensor ple_history_slot(std::int32_t slot) const;

    [[nodiscard]] Tensor& continuation_hidden_store() noexcept { return continuation_hidden_; }

    [[nodiscard]] const Tensor& continuation_hidden_store() const noexcept {
        return continuation_hidden_;
    }

    [[nodiscard]] const StateImageHostLayout& host_layout() const noexcept { return host_layout_; }

    void zero_slot(std::int32_t slot, cudaStream_t stream = nullptr);
    void zero_all(cudaStream_t stream = nullptr);
    void copy_slot(std::int32_t source, std::int32_t destination, cudaStream_t stream = nullptr);
    void copy_to_host(std::int32_t source, HostStateImageView destination,
                      cudaStream_t stream = nullptr) const;
    void copy_from_host(HostStateImageConstView source, std::int32_t destination,
                        cudaStream_t stream = nullptr);

private:
    void validate_host_layout(const StateImageHostLayout* layout, const std::byte* data) const;

    LinearAttentionStatePool linear_;
    std::vector<Tensor> indexer_keys_;
    std::vector<Tensor> indexer_positions_;
    Tensor ple_history_;
    Tensor continuation_hidden_;
    Tensor continuation_positions_;
    StateImageHostLayout host_layout_;
};

} // namespace ninfer::models::qwen4_exp
