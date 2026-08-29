#include "targets/qwen3_8_flash_next/impl/text_executor.h"

#include "targets/qwen3_8_flash_next/impl/text_decode.h"
#include "ninfer/ops/embedding.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// ---------------------------------------------------------------------------
// PendingRound
// ---------------------------------------------------------------------------

PendingRound::PendingRound(FlashNextTextExecutor* owner, std::uint64_t transaction_id,
                           std::uint32_t batch_size, Tensor logits, Tensor final_hidden) noexcept
    : owner_(owner), transaction_id_(transaction_id), batch_size_(batch_size), logits_(logits),
      final_hidden_(final_hidden) {}

PendingRound::PendingRound(PendingRound&& other) noexcept
    : owner_(other.owner_), transaction_id_(other.transaction_id_), batch_size_(other.batch_size_),
      logits_(other.logits_), final_hidden_(other.final_hidden_) {
    other.owner_          = nullptr;
    other.transaction_id_ = 0;
    other.batch_size_     = 0;
}

PendingRound& PendingRound::operator=(PendingRound&& other) noexcept {
    if (this != &other) {
        if (valid()) { abort(); }
        owner_                = other.owner_;
        transaction_id_       = other.transaction_id_;
        batch_size_           = other.batch_size_;
        logits_               = other.logits_;
        final_hidden_         = other.final_hidden_;
        other.owner_          = nullptr;
        other.transaction_id_ = 0;
        other.batch_size_     = 0;
    }
    return *this;
}

bool PendingRound::valid() const noexcept { return owner_ != nullptr && transaction_id_ != 0; }

std::uint32_t PendingRound::batch_size() const {
    if (!valid()) { throw std::logic_error("PendingRound: transaction is not valid"); }
    return batch_size_;
}

Tensor PendingRound::logits() const {
    if (!valid()) { throw std::logic_error("PendingRound: transaction is not valid"); }
    return logits_;
}

Tensor PendingRound::final_hidden() const {
    if (!valid()) { throw std::logic_error("PendingRound: transaction is not valid"); }
    return final_hidden_;
}

void PendingRound::commit(std::span<const LaneCommitDecision> decisions) {
    if (!valid()) { throw std::logic_error("PendingRound: cannot commit invalid transaction"); }
    auto* owner      = owner_;
    const auto tx_id = transaction_id_;
    owner->commit_transaction(tx_id, decisions);
    owner_          = nullptr;
    transaction_id_ = 0;
    batch_size_     = 0;
}

void PendingRound::abort() noexcept {
    if (!valid()) { return; }
    auto* owner      = owner_;
    const auto tx_id = transaction_id_;
    owner->abort_transaction(tx_id);
    owner_          = nullptr;
    transaction_id_ = 0;
    batch_size_     = 0;
}

// ---------------------------------------------------------------------------
// FlashNextTextExecutor
// ---------------------------------------------------------------------------

FlashNextTextExecutor::FlashNextTextExecutor(const TextModelView& model,
                                             PleIndexMetadata ple_metadata, DeviceContext& device,
                                             FlashNextRuntimeAllocation& allocation)
    : model_(model), ple_metadata_(ple_metadata), device_(device), alloc_(allocation),
      ple_pipeline_(model.ple.table, device, allocation.plan().config.max_concurrency),
      ledger_(allocation.plan()) {
    const auto& plan                = alloc_.plan();
    const std::uint32_t concurrency = plan.config.max_concurrency;

    host_token_ids_.resize(concurrency);
    host_token_indices_.resize(concurrency);
    host_mrope_positions_.resize(concurrency * 3);
    host_table_rows_.resize(concurrency);
    host_source_slots_.resize(concurrency);
    host_destination_slots_.resize(concurrency);
}

LaneHandle FlashNextTextExecutor::allocate_lane() {
    auto handle = ledger_.allocate_lane(this);
    try {
        alloc_.zero_lane_slots(handle.lane_index(), device_.stream);
    } catch (...) {
        ledger_.release_lane(handle);
        throw;
    }
    return handle;
}

void FlashNextTextExecutor::release_lane(LaneHandle handle) {
    if (handle.owner() != this) {
        throw std::invalid_argument(
            "FlashNextTextExecutor: cross-executor or invalid owner handle");
    }
    ledger_.release_lane(handle);
}

std::int32_t FlashNextTextExecutor::committed_frontier(LaneHandle handle) const {
    if (handle.owner() != this) {
        throw std::invalid_argument(
            "FlashNextTextExecutor: cross-executor or invalid owner handle");
    }
    return ledger_.committed_frontier(handle);
}

std::size_t FlashNextTextExecutor::active_lanes_count() const noexcept {
    return ledger_.active_lanes_count();
}

PendingRound FlashNextTextExecutor::execute_round(std::span<const LaneStepRequest> requests,
                                                   const FlashNextDecodeStateSink* sink) {
    for (const auto& req : requests) {
        if (req.handle.owner() != this) {
            throw std::invalid_argument(
                "FlashNextTextExecutor: cross-executor or invalid owner handle");
        }
    }

    const auto batch_size = static_cast<std::uint32_t>(requests.size());
    auto prepared         = ledger_.begin_round(requests, ple_metadata_);

    try {
        alloc_.workspace().reset();

        // Sync dirty tables to device
        ledger_.sync_tables_if_dirty(alloc_, device_.stream);

        // Prepare and enqueue PLE gather
        auto ticket = ple_pipeline_.prepare(prepared.ple_indices);
        Tensor gathered_ple(alloc_.round_tensors().gathered_ple_embedding.data, DType::BF16,
                            {2'560, static_cast<std::int32_t>(batch_size)});
        ple_pipeline_.enqueue_copy(std::move(ticket), gathered_ple);

        // Populate and upload host round buffers
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            const auto lane            = requests[i].handle.lane_index();
            host_token_ids_[i]         = requests[i].token_id;
            host_token_indices_[i]     = requests[i].token_index;
            for (std::uint32_t d = 0; d < 3; ++d) {
                host_mrope_positions_[d * batch_size + i] = requests[i].mrope_positions[d];
            }
            host_table_rows_[i]        = static_cast<std::int32_t>(lane);
            host_source_slots_[i]      = alloc_.current_source_slot(lane);
            host_destination_slots_[i] = alloc_.current_destination_slot(lane);
        }

        Tensor token_ids(alloc_.round_tensors().token_ids.data, DType::I32,
                         {static_cast<std::int32_t>(batch_size)});
        Tensor token_indices(alloc_.round_tensors().token_indices.data, DType::I32,
                             {static_cast<std::int32_t>(batch_size)});
        Tensor mrope_positions(alloc_.round_tensors().mrope_positions.data, DType::I32,
                               {static_cast<std::int32_t>(batch_size), 3});
        Tensor table_rows(alloc_.round_tensors().table_rows.data, DType::I32,
                          {static_cast<std::int32_t>(batch_size)});
        Tensor source_slots(alloc_.round_tensors().source_slots.data, DType::I32,
                            {static_cast<std::int32_t>(batch_size)});
        Tensor destination_slots(alloc_.round_tensors().destination_slots.data, DType::I32,
                                 {static_cast<std::int32_t>(batch_size)});

        CUDA_CHECK(cudaMemcpyAsync(token_ids.data, host_token_ids_.data(),
                                   batch_size * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   device_.stream));
        CUDA_CHECK(cudaMemcpyAsync(token_indices.data, host_token_indices_.data(),
                                   batch_size * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   device_.stream));
        CUDA_CHECK(cudaMemcpyAsync(mrope_positions.data, host_mrope_positions_.data(),
                                   batch_size * 3 * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   device_.stream));
        CUDA_CHECK(cudaMemcpyAsync(table_rows.data, host_table_rows_.data(),
                                   batch_size * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   device_.stream));
        CUDA_CHECK(cudaMemcpyAsync(source_slots.data, host_source_slots_.data(),
                                   batch_size * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   device_.stream));
        CUDA_CHECK(cudaMemcpyAsync(destination_slots.data, host_destination_slots_.data(),
                                   batch_size * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   device_.stream));

        Tensor final_hidden(alloc_.round_tensors().final_hidden.data, DType::BF16,
                            {2'560, static_cast<std::int32_t>(batch_size)});
        Tensor logits(alloc_.round_tensors().logits.data, DType::BF16,
                      {248'320, static_cast<std::int32_t>(batch_size)});

        // Allocate embedding buffer in workspace
        Tensor embedding = alloc_.workspace().alloc(DType::BF16, {2'560, static_cast<std::int32_t>(batch_size)}, 256);
        ops::embedding(token_ids, model_.token_embedding, embedding, device_.stream);

        // Replace custom embeddings if specified (multimodal vision tokens)
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            if (requests[i].custom_embedding != nullptr) {
                const auto& custom = *requests[i].custom_embedding;
                if (custom.dtype != DType::BF16 || custom.ne[0] != 2'560 || !custom.is_contiguous() || custom.data == nullptr) {
                    throw std::invalid_argument("LaneStepRequest custom_embedding tensor view is invalid");
                }
                CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<std::uint16_t*>(embedding.data) + static_cast<std::size_t>(i) * 2'560,
                    custom.data,
                    2'560 * sizeof(std::uint16_t),
                    cudaMemcpyDeviceToDevice,
                    device_.stream));
            }
        }

        // Launch full target decode core
        flash_next_text_decode_core(model_, embedding, token_indices, mrope_positions, table_rows,
                                    source_slots, destination_slots, gathered_ple,
                                    static_cast<std::int32_t>(alloc_.plan().maximum_blocks),
                                    prepared.max_active_blocks, alloc_.state_view(), alloc_.workspace(),
                                    final_hidden, logits, device_.stream, sink);

        return PendingRound(this, prepared.transaction_id, batch_size, logits, final_hidden);
    } catch (...) {
        ledger_.rollback_prepared_round(prepared.transaction_id);
        throw;
    }
}

void FlashNextTextExecutor::commit_transaction(std::uint64_t tx_id,
                                               std::span<const LaneCommitDecision> decisions) {
    ledger_.commit_round(tx_id, decisions, alloc_, device_.stream);
}

void FlashNextTextExecutor::abort_transaction(std::uint64_t tx_id) noexcept {
    ledger_.abort_round(tx_id);
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
