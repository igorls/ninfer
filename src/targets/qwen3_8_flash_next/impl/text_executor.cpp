#include "targets/qwen3_8_flash_next/impl/text_executor.h"

#include "targets/qwen3_8_flash_next/impl/mtp_forward.h"
#include "targets/qwen3_8_flash_next/impl/stage_ledger.h"
#include "targets/qwen3_8_flash_next/impl/text_decode.h"
#include "targets/qwen3_8_flash_next/impl/text_decode_workspace.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scatter.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// ---------------------------------------------------------------------------
// PendingRound
// ---------------------------------------------------------------------------

PendingRound::PendingRound(FlashNextTextExecutor* owner, std::uint64_t transaction_id,
                           std::uint32_t batch_size, Tensor logits, Tensor final_hidden,
                           Tensor hyper_hidden,
                           std::span<const std::int32_t> sampled_tokens) noexcept
    : owner_(owner), transaction_id_(transaction_id), batch_size_(batch_size), logits_(logits),
      final_hidden_(final_hidden), hyper_hidden_(hyper_hidden) {
    const std::size_t n = std::min<std::size_t>(batch_size, sampled_tokens.size());
    for (std::size_t i = 0; i < n; ++i) {
        sampled_tokens_[i] = sampled_tokens[i];
    }
}

PendingRound::PendingRound(PendingRound&& other) noexcept
    : owner_(other.owner_), transaction_id_(other.transaction_id_), batch_size_(other.batch_size_),
      logits_(other.logits_), final_hidden_(other.final_hidden_),
      hyper_hidden_(other.hyper_hidden_), sampled_tokens_(other.sampled_tokens_) {
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
        hyper_hidden_         = other.hyper_hidden_;
        sampled_tokens_       = other.sampled_tokens_;
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

Tensor PendingRound::hyper_hidden() const {
    if (!valid()) { throw std::logic_error("PendingRound: transaction is not valid"); }
    return hyper_hidden_;
}

std::span<const std::int32_t> PendingRound::sampled_tokens() const {
    if (!valid()) { throw std::logic_error("PendingRound: transaction is not valid"); }
    return std::span(sampled_tokens_.data(), batch_size_);
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

void PendingRound::commit_speculative(std::uint32_t lane_index,
                                      std::span<const std::int32_t> accepted_tokens) {
    if (!valid()) { throw std::logic_error("PendingRound: cannot commit invalid transaction"); }
    auto* owner      = owner_;
    const auto tx_id = transaction_id_;
    owner->commit_speculative_transaction(tx_id, lane_index, accepted_tokens);
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
      ple_pipeline_(model.ple.table, device,
                    std::max(allocation.plan().config.max_concurrency,
                             allocation.plan().config.prefill_chunk)),
      ledger_(allocation.plan()),
      use_cuda_graph_(allocation.plan().config.use_cuda_graph),
      sampling_workspace_(std::max<std::size_t>(
          256,
          ops::sampling_workspace_capacity_bytes(
              248'077, 1,
              static_cast<std::int32_t>(std::max(
                  allocation.plan().config.max_concurrency,
                  allocation.plan().config.speculative_draft_tokens > 0
                      ? (allocation.plan().config.speculative_draft_tokens + 1U)
                      : 1U))))),
      round_completion_(device) {
    if (model_.mtp.has_value()) {
        const std::uint64_t kv_element_bytes =
            allocation.plan().config.kv_cache == KvCacheStorage::Fp8E4M3Row256 ? 1ULL : sizeof(std::uint16_t);
        const auto kv_dt =
            allocation.plan().config.kv_cache == KvCacheStorage::Fp8E4M3Row256 ? DType::FP8_E4M3FN : DType::BF16;
        const std::uint64_t key_bytes =
            256ULL * 64ULL * 2ULL * allocation.plan().attention_physical_pages * kv_element_bytes;
        const std::uint64_t val_bytes = key_bytes;
        mtp_key_pages_   = std::make_unique<DeviceBuffer>(key_bytes);
        mtp_value_pages_ = std::make_unique<DeviceBuffer>(val_bytes);

        QsaAttentionCacheView mtp_cache{};
        mtp_cache.key_pages =
            Tensor(mtp_key_pages_->p, kv_dt,
                   {256, 64, 2, static_cast<std::int32_t>(allocation.plan().attention_physical_pages)});
        mtp_cache.value_pages =
            Tensor(mtp_value_pages_->p, kv_dt,
                   {256, 64, 2, static_cast<std::int32_t>(allocation.plan().attention_physical_pages)});
        mtp_cache.block_tables =
            allocation.state_view().qsa_attention_caches[0].block_tables;
        mtp_cache_ = mtp_cache;

        const std::size_t mtp_ws_bytes =
            flash_next_mtp_workspace_capacity_bytes(1);
        mtp_workspace_       = std::make_unique<WorkspaceArena>(mtp_ws_bytes);
        mtp_selected_blocks_ = std::make_unique<DeviceBuffer>(512 * sizeof(std::int32_t));
        mtp_selected_counts_ = std::make_unique<DeviceBuffer>(1 * sizeof(std::int32_t));
        const std::size_t draft_logits_rows =
            model_.proposal.has_value() ? static_cast<std::size_t>(model_.proposal->head.n)
                                        : 248'320ULL;
        mtp_draft_logits_    = std::make_unique<DeviceBuffer>(draft_logits_rows * sizeof(std::uint16_t));
        mtp_draft_tokens_    = std::make_unique<DeviceBuffer>(1 * sizeof(std::int32_t));
        mtp_input_embedding_ = std::make_unique<DeviceBuffer>(2'560 * sizeof(std::uint16_t));
        mtp_carried_hidden_  = std::make_unique<DeviceBuffer>(10'240 * sizeof(std::uint16_t));

        CUDA_CHECK(cudaMemsetAsync(mtp_key_pages_->p, 0, key_bytes, device_.stream));
        CUDA_CHECK(cudaMemsetAsync(mtp_value_pages_->p, 0, val_bytes, device_.stream));
        CUDA_CHECK(cudaMemsetAsync(mtp_selected_blocks_->p, 0, 512 * sizeof(std::int32_t), device_.stream));
        CUDA_CHECK(cudaMemsetAsync(mtp_selected_counts_->p, 0, 1 * sizeof(std::int32_t), device_.stream));
        CUDA_CHECK(cudaMemsetAsync(mtp_carried_hidden_->p, 0, 10'240 * sizeof(std::uint16_t), device_.stream));
        CUDA_CHECK(cudaStreamSynchronize(device_.stream));
    }
    instantiate_graphs();
}

void FlashNextTextExecutor::instantiate_graphs() {
    decode_graphs_.buckets = flash_next_decode_graph_buckets(alloc_.plan().maximum_blocks);
    if (!use_cuda_graph_ || model_.token_embedding.payload == nullptr) { return; }
    if (decode_graphs_.buckets.count == 0) { return; }

    const std::uint32_t max_concurrency = alloc_.plan().config.max_concurrency;
    const auto small_blocks = static_cast<std::int32_t>(decode_graphs_.buckets.blocks[0]);
    decode_graphs_.profiles.clear();
    decode_graphs_.topologies.clear();
    decode_graphs_.profiles.reserve(max_concurrency);
    decode_graphs_.topologies.reserve(max_concurrency);

    std::memset(alloc_.host_ingress(), 0, sizeof(FlashNextDecodeIngress));
    std::memset(const_cast<void*>(ple_pipeline_.fixed_host_buffer()), 0,
                max_concurrency * 2'560 * sizeof(std::uint16_t));

    pending_custom_embeddings_.clear();
    for (std::uint32_t B = 1; B <= max_concurrency; ++B) {
        // Startup captures the smallest bucket only. Larger buckets are lazy.
        execute_round_body(B, small_blocks, nullptr);
        device_.synchronize();

        if (!install_captured_graph(B, 0, small_blocks)) {
            throw std::runtime_error(
                "FlashNextTextExecutor: failed to capture decode graph for B=" + std::to_string(B) +
                " bucket=0 blocks=" + std::to_string(small_blocks));
        }
        device_.synchronize();
    }
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

bool FlashNextTextExecutor::decode_graph_pinned_eager(std::uint32_t batch_size,
                                                      std::uint32_t bucket_index) const noexcept {
    if (batch_size < 1 || batch_size > 8 || bucket_index >= kFlashNextDecodeGraphMaxBuckets) {
        return false;
    }
    return graph_pinned_eager_[batch_size - 1][bucket_index];
}

DecodeGraphTopology* FlashNextTextExecutor::find_topology(std::uint32_t batch_size,
                                                          std::uint32_t bucket_index) noexcept {
    for (auto& topology : decode_graphs_.topologies) {
        if (topology.batch_size == batch_size && topology.bucket_index == bucket_index) {
            return &topology;
        }
    }
    return nullptr;
}

const DecodeGraphTopology*
FlashNextTextExecutor::find_topology(std::uint32_t batch_size,
                                     std::uint32_t bucket_index) const noexcept {
    for (const auto& topology : decode_graphs_.topologies) {
        if (topology.batch_size == batch_size && topology.bucket_index == bucket_index) {
            return &topology;
        }
    }
    return nullptr;
}

bool FlashNextTextExecutor::install_captured_graph(std::uint32_t batch_size,
                                                   std::uint32_t bucket_index,
                                                   std::int32_t bucket_blocks) {
    DecodeGraphProfile profile;
    profile.batch_size             = batch_size;
    profile.bucket_index           = bucket_index;
    profile.bucket_blocks          = static_cast<std::uint32_t>(bucket_blocks);
    profile.min_execution_frontier = 0;
    profile.max_execution_frontier = static_cast<std::uint32_t>(bucket_blocks);
    profile.topology_class = flash_next_decode_graph_topology_class(batch_size, bucket_index);

    profile.definition.capture(device_.stream, [this, batch_size, bucket_blocks] {
        execute_round_body(batch_size, bucket_blocks, nullptr);
    });
    if (!profile.definition.ready()) { return false; }

    DecodeGraphTopology topology;
    topology.topology_class = profile.topology_class;
    topology.batch_size     = batch_size;
    topology.bucket_index   = bucket_index;
    topology.executable.instantiate(profile.definition);
    decode_graphs_.profiles.push_back(std::move(profile));
    topology.installed_profile = decode_graphs_.profiles.size() - 1;
    topology.executable.upload(device_.stream);
    decode_graphs_.topologies.push_back(std::move(topology));
    return true;
}

FlashNextTextExecutor::LazyCaptureOutcome
FlashNextTextExecutor::try_lazy_capture(std::uint32_t batch_size, std::uint32_t bucket_index,
                                        std::int32_t bucket_blocks) {
    const std::uint32_t bslot = batch_size - 1;
    if (graph_pinned_eager_[bslot][bucket_index]) { return LazyCaptureOutcome::NeedEager; }

    // Dummy warmup+capture mutates persistent decode state. Snapshot/restore so the
    // production round can launch the new graph against the live KV/ingress.
    const std::size_t persistent_bytes = alloc_.persistent_bytes();
    if (!lazy_capture_scratch_ || lazy_capture_scratch_->bytes < persistent_bytes) {
        lazy_capture_scratch_ = std::make_unique<DeviceBuffer>(persistent_bytes);
    }
    const std::size_t ple_bytes =
        static_cast<std::size_t>(alloc_.plan().config.max_concurrency) * 2'560 *
        sizeof(std::uint16_t);
    std::vector<std::byte> saved_ple(ple_bytes);
    std::memcpy(saved_ple.data(), ple_pipeline_.fixed_host_buffer(), ple_bytes);
    const FlashNextDecodeIngress saved_ing = *alloc_.host_ingress();
    CUDA_CHECK(cudaMemcpyAsync(lazy_capture_scratch_->p, alloc_.persistent_base(), persistent_bytes,
                               cudaMemcpyDeviceToDevice, device_.stream));

    bool installed = false;
    const auto t0  = std::chrono::steady_clock::now();
    try {
        std::memset(alloc_.host_ingress(), 0, sizeof(FlashNextDecodeIngress));
        std::memset(const_cast<void*>(ple_pipeline_.fixed_host_buffer()), 0, ple_bytes);
        execute_round_body(batch_size, bucket_blocks, nullptr);
        device_.synchronize();
        if (!install_captured_graph(batch_size, bucket_index, bucket_blocks)) {
            throw std::runtime_error("capture produced an empty graph definition");
        }
        const auto t1 = std::chrono::steady_clock::now();
        last_lazy_capture_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
        graph_capture_failures_[bslot][bucket_index] = 0;
        installed = true;
    } catch (const std::exception& ex) {
        graph_capture_failures_[bslot][bucket_index] =
            static_cast<std::uint8_t>(graph_capture_failures_[bslot][bucket_index] + 1);
        std::fprintf(stderr,
                     "FlashNextTextExecutor: CUDA graph capture failed for B=%u bucket=%u "
                     "blocks=%d (attempt %u): %s\n",
                     batch_size, bucket_index, bucket_blocks,
                     static_cast<unsigned>(graph_capture_failures_[bslot][bucket_index]),
                     ex.what());
        if (graph_capture_failures_[bslot][bucket_index] >= 2) {
            graph_pinned_eager_[bslot][bucket_index] = true;
            std::fprintf(stderr,
                         "FlashNextTextExecutor: pinning B=%u bucket=%u to permanent eager decode\n",
                         batch_size, bucket_index);
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(alloc_.persistent_base(), lazy_capture_scratch_->p, persistent_bytes,
                               cudaMemcpyDeviceToDevice, device_.stream));
    *alloc_.host_ingress() = saved_ing;
    std::memcpy(const_cast<void*>(ple_pipeline_.fixed_host_buffer()), saved_ple.data(), ple_bytes);
    return installed ? LazyCaptureOutcome::Installed : LazyCaptureOutcome::NeedEager;
}

void FlashNextTextExecutor::execute_round_body(std::uint32_t batch_size,
                                              std::int32_t active_blocks,
                                              const FlashNextDecodeStateSink* sink) {
    if (active_blocks < 0 ||
        static_cast<std::uint32_t>(active_blocks) > alloc_.plan().maximum_blocks) {
        throw std::invalid_argument(
            "FlashNextTextExecutor: active_blocks must be in [0, maximum_blocks]");
    }
    alloc_.workspace().reset();

    // 1. Single captured H2D memcpy of pinned ingress
    CUDA_CHECK(cudaMemcpyAsync(alloc_.device_ingress_ptr(), alloc_.host_ingress(),
                               sizeof(FlashNextDecodeIngress), cudaMemcpyHostToDevice,
                               device_.stream));

    // 2. Fixed-address H2D memcpy of PLE gathered rows
    Tensor gathered_ple(alloc_.round_tensors().gathered_ple_embedding.data, DType::BF16,
                        {2'560, static_cast<std::int32_t>(batch_size)});
    CUDA_CHECK(cudaMemcpyAsync(gathered_ple.data, ple_pipeline_.fixed_host_buffer(),
                               batch_size * 2'560 * sizeof(std::uint16_t),
                               cudaMemcpyHostToDevice, device_.stream));

    // 3. Token embedding
    Tensor token_ids =
        alloc_.round_tensors().token_ids.slice(0, 0, static_cast<std::int32_t>(batch_size));
    Tensor embedding =
        alloc_.workspace().alloc(DType::BF16, {2'560, static_cast<std::int32_t>(batch_size)}, 256);
    ops::embedding(token_ids, model_.token_embedding, embedding, device_.stream);
    for (std::uint32_t i = 0; i < batch_size && i < pending_custom_embeddings_.size(); ++i) {
        const Tensor* custom = pending_custom_embeddings_[i];
        if (custom == nullptr) { continue; }
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(embedding.data) +
                                       static_cast<std::size_t>(i) * 2'560 * sizeof(std::uint16_t),
                                   custom->data, 2'560 * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToDevice, device_.stream));
    }

    // 4. Target decode core
    Tensor token_indices =
        alloc_.round_tensors().token_indices.slice(0, 0, static_cast<std::int32_t>(batch_size));
    Tensor mrope_positions(alloc_.round_tensors().mrope_positions.data, DType::I32,
                           {static_cast<std::int32_t>(batch_size), 3});
    Tensor table_rows =
        alloc_.round_tensors().table_rows.slice(0, 0, static_cast<std::int32_t>(batch_size));
    Tensor source_slots =
        alloc_.round_tensors().source_slots.slice(0, 0, static_cast<std::int32_t>(batch_size));
    Tensor destination_slots =
        alloc_.round_tensors().destination_slots.slice(0, 0, static_cast<std::int32_t>(batch_size));
    Tensor final_hidden =
        alloc_.round_tensors().final_hidden.slice(1, 0, static_cast<std::int32_t>(batch_size));
    Tensor hyper_hidden =
        alloc_.round_tensors().hyper_hidden.slice(1, 0, static_cast<std::int32_t>(batch_size));
    Tensor logits =
        alloc_.round_tensors().logits.slice(1, 0, static_cast<std::int32_t>(batch_size));

    const auto max_blocks = static_cast<std::int32_t>(alloc_.plan().maximum_blocks);
    flash_next_text_decode_core(model_, embedding, token_indices, mrope_positions, table_rows,
                                source_slots, destination_slots, gathered_ple, max_blocks,
                                active_blocks, alloc_.state_view(), alloc_.workspace(),
                                final_hidden, logits, device_.stream, sink, &hyper_hidden);

    // 5. Sampler
    Tensor sampled_tokens =
        alloc_.round_tensors().sampled_tokens.slice(0, 0, static_cast<std::int32_t>(batch_size));
    constexpr std::int32_t kSemanticTokenDomain = 248'077;
    ops::sample(logits, sampled_tokens, kSemanticTokenDomain,
                alloc_.device_sampling_configs(), token_indices,
                ops::kSamplePurposeDecode, sampling_workspace_, device_.stream);

    // 6. Single captured D2H memcpy of pinned egress
    CUDA_CHECK(cudaMemcpyAsync(alloc_.host_egress(), alloc_.device_egress_ptr(),
                               sizeof(FlashNextDecodeEgress), cudaMemcpyDeviceToHost,
                               device_.stream));
}

PendingRound FlashNextTextExecutor::execute_round(std::span<const LaneStepRequest> requests,
                                                 const FlashNextDecodeStateSink* sink) {
    for (const auto& req : requests) {
        if (req.handle.owner() != this) {
            throw std::invalid_argument(
                "FlashNextTextExecutor: cross-executor or invalid owner handle");
        }
        if (req.custom_embedding != nullptr &&
            (req.custom_embedding->dtype != DType::BF16 || req.custom_embedding->ne[0] != 2'560 ||
             !req.custom_embedding->is_contiguous())) {
            throw std::invalid_argument(
                "FlashNextTextExecutor: custom embedding must be a contiguous BF16 [2560] column");
        }
    }

    const auto batch_size = static_cast<std::uint32_t>(requests.size());
    if (batch_size == 0 || batch_size > alloc_.plan().config.max_concurrency) {
        throw std::invalid_argument("FlashNextTextExecutor: invalid round batch size");
    }
    if (round_in_flight_) {
        throw std::logic_error(
            "FlashNextTextExecutor: cannot start round while previous round is in flight");
    }

    auto prepared = ledger_.begin_round(requests, ple_metadata_);
    const std::uint32_t bucket_index =
        flash_next_decode_graph_select_bucket(decode_graphs_.buckets, prepared.max_active_blocks);
    const auto bucket_blocks = static_cast<std::int32_t>(decode_graphs_.buckets.blocks[bucket_index]);
    return finish_prepared_round(requests, std::move(prepared), sink, false, bucket_blocks);
}

PendingRound FlashNextTextExecutor::execute_round_eager(std::span<const LaneStepRequest> requests,
                                                        std::int32_t active_blocks,
                                                        const FlashNextDecodeStateSink* sink) {
    for (const auto& req : requests) {
        if (req.handle.owner() != this) {
            throw std::invalid_argument(
                "FlashNextTextExecutor: cross-executor or invalid owner handle");
        }
        if (req.custom_embedding != nullptr &&
            (req.custom_embedding->dtype != DType::BF16 || req.custom_embedding->ne[0] != 2'560 ||
             !req.custom_embedding->is_contiguous())) {
            throw std::invalid_argument(
                "FlashNextTextExecutor: custom embedding must be a contiguous BF16 [2560] column");
        }
    }

    const auto batch_size = static_cast<std::uint32_t>(requests.size());
    if (batch_size == 0 || batch_size > alloc_.plan().config.max_concurrency) {
        throw std::invalid_argument("FlashNextTextExecutor: invalid round batch size");
    }
    if (round_in_flight_) {
        throw std::logic_error(
            "FlashNextTextExecutor: cannot start round while previous round is in flight");
    }
    if (active_blocks < 0 ||
        static_cast<std::uint32_t>(active_blocks) > alloc_.plan().maximum_blocks) {
        throw std::invalid_argument("FlashNextTextExecutor: active_blocks outside maximum_blocks");
    }

    auto prepared = ledger_.begin_round(requests, ple_metadata_);
    return finish_prepared_round(requests, std::move(prepared), sink, true, active_blocks);
}

PendingRound FlashNextTextExecutor::finish_prepared_round(
    std::span<const LaneStepRequest> requests, FlashNextLaneLedger::PreparedRound prepared,
    const FlashNextDecodeStateSink* sink, bool force_eager, std::int32_t active_blocks) {
    const auto batch_size = static_cast<std::uint32_t>(requests.size());
    try {
        pending_is_prefill_chunk_ = false;

        // 1. Sync dirty tables to device (out of captured graph)
        ledger_.sync_tables_if_dirty(alloc_, device_.stream);

        // 2. Synchronous host gather into fixed pinned host buffer before launch
        ple_pipeline_.gather_pinned(prepared.ple_indices);

        // 3. Populate pinned host ingress
        auto* host_ing = alloc_.host_ingress();
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            const auto lane            = requests[i].handle.lane_index();
            host_ing->token_ids[i]     = requests[i].token_id;
            host_ing->token_indices[i] = requests[i].token_index;
            for (std::uint32_t d = 0; d < 3; ++d) {
                host_ing->mrope_positions[d * batch_size + i] = requests[i].mrope_positions[d];
            }
            host_ing->table_rows[i]        = static_cast<std::int32_t>(lane);
            host_ing->source_slots[i]      = alloc_.current_source_slot(lane);
            host_ing->destination_slots[i] = alloc_.current_destination_slot(lane);
            host_ing->sampling[i]          = requests[i].sampling;
        }

        // 4. Launch either CUDA graph replay or eager decode body. A state sink needs the eager
        //    body: replay cannot emit per-stage states, and silently ignoring the sink would
        //    break the teacher-forced dump harness. Rounds carrying custom embeddings (the
        //    reference tool's per-token vision path) also run eagerly: their columns are copied
        //    after the token embedding, which a captured graph cannot do.
        pending_custom_embeddings_.assign(batch_size, nullptr);
        bool has_custom = false;
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            pending_custom_embeddings_[i] = requests[i].custom_embedding;
            has_custom = has_custom || requests[i].custom_embedding != nullptr;
        }

        const std::uint32_t bucket_index = flash_next_decode_graph_select_bucket(
            decode_graphs_.buckets, active_blocks);
        bool ran = false;
        if (!force_eager && use_cuda_graph_ && sink == nullptr && !has_custom &&
            model_.token_embedding.payload != nullptr && decode_graphs_.buckets.count > 0) {
            if (auto* topology = find_topology(batch_size, bucket_index);
                topology != nullptr && topology->executable.ready()) {
                topology->executable.launch(device_.stream);
                ran = true;
            } else if (!graph_pinned_eager_[batch_size - 1][bucket_index]) {
                const auto outcome = try_lazy_capture(batch_size, bucket_index, active_blocks);
                if (outcome == LazyCaptureOutcome::Installed) {
                    auto* topology = find_topology(batch_size, bucket_index);
                    if (topology == nullptr || !topology->executable.ready()) {
                        throw std::logic_error(
                            "FlashNextTextExecutor: lazy capture installed without a ready topology");
                    }
                    topology->executable.launch(device_.stream);
                    ran = true;
                }
            }
        }
        if (!ran) { execute_round_body(batch_size, active_blocks, sink); }

        // 5. Complete round on event wait
        round_completion_.record(device_.stream);
        {
            const auto wait_started = std::chrono::steady_clock::now();
            round_completion_.synchronize();
            round_device_wait_ns_ += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wait_started)
                    .count());
        }

        // 6. Copy sampled tokens from pinned host egress
        std::array<std::int32_t, 8> sampled_tokens{};
        const auto* host_egr = alloc_.host_egress();
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            sampled_tokens[i] = host_egr->sampled_tokens[i];
        }

        Tensor final_hidden =
            alloc_.round_tensors().final_hidden.slice(1, 0, static_cast<std::int32_t>(batch_size));
        Tensor hyper_hidden =
            alloc_.round_tensors().hyper_hidden.slice(1, 0, static_cast<std::int32_t>(batch_size));
        Tensor logits =
            alloc_.round_tensors().logits.slice(1, 0, static_cast<std::int32_t>(batch_size));

        round_in_flight_ = true;
        return PendingRound(this, prepared.transaction_id, batch_size, logits, final_hidden,
                            hyper_hidden, std::span(sampled_tokens.data(), batch_size));
    } catch (...) {
        ledger_.rollback_prepared_round(prepared.transaction_id);
        round_in_flight_ = false;
        throw;
    }
}

PendingRound FlashNextTextExecutor::execute_prefill_chunk(
    LaneHandle handle, std::span<const std::int32_t> token_ids,
    std::span<const std::array<std::int32_t, 3>> positions, std::int32_t first_token_index,
    const FlashNextDecodeStateSink* sink, const Tensor* visual_embeddings,
    std::span<const std::int32_t> chunk_local_scatter_indices) {
    if (handle.owner() != this) {
        throw std::invalid_argument(
            "FlashNextTextExecutor: cross-executor or invalid owner handle");
    }
    if (token_ids.empty() || token_ids.size() != positions.size()) {
        throw std::invalid_argument(
            "FlashNextTextExecutor: token_ids and positions must be non-empty and matching size");
    }
    if (round_in_flight_) {
        throw std::logic_error(
            "FlashNextTextExecutor: cannot start prefill chunk while previous round is in flight");
    }

    const auto num_tokens                  = static_cast<std::uint32_t>(token_ids.size());
    const std::uint32_t lane               = handle.lane_index();
    const std::int32_t initial_active_slot = alloc_.current_source_slot(lane);
    const std::int32_t initial_standby_slot =
        alloc_.current_destination_slot(lane);

    auto prepared =
        ledger_.begin_prefill_chunk(handle, token_ids, first_token_index, ple_metadata_);

    try {
        pending_is_prefill_chunk_             = true;
        pending_prefill_lane_                 = lane;
        pending_prefill_initial_active_slot_  = initial_active_slot;
        pending_prefill_initial_standby_slot_ = initial_standby_slot;

        alloc_.workspace().reset();

        // Sync table updates if dirty
        ledger_.sync_tables_if_dirty(alloc_, device_.stream);

        FlashNextStageLedger::instance().begin_chunk(device_.stream, static_cast<std::int32_t>(num_tokens));

        // 1. Gather all 16*T PLE rows for the chunk at once
        PleTokenHistory temp_history = ledger_.lane_history(handle);
        std::vector<std::array<std::int64_t, 16>> chunk_ple_indices(num_tokens);
        for (std::uint32_t t = 0; t < num_tokens; ++t) {
            chunk_ple_indices[t] = ple_indices(ple_metadata_, temp_history, token_ids[t]);
            temp_history.commit(token_ids[t]);
        }
        auto ticket = ple_pipeline_.prepare(std::span(chunk_ple_indices));
        // Staging comes from the workspace through the same layout the capacity estimate uses.
        FlashNextPrefillChunkStaging staging = allocate_flash_next_prefill_chunk_staging(
            alloc_.workspace(), static_cast<std::int32_t>(num_tokens));
        Tensor& gathered_ple        = staging.gathered_ple;
        Tensor& dev_token_ids       = staging.token_ids;
        Tensor& dev_token_indices   = staging.token_indices;
        Tensor& dev_mrope_positions = staging.mrope_positions;
        Tensor& embedding           = staging.embedding;
        ple_pipeline_.enqueue_copy(std::move(ticket), gathered_ple);

        // 2. Upload chunk input metadata to the staging tensors
        std::vector<std::int32_t> host_indices(num_tokens);
        std::vector<std::int32_t> host_flat_positions(3 * num_tokens);
        for (std::uint32_t t = 0; t < num_tokens; ++t) {
            host_indices[t]                         = first_token_index + static_cast<std::int32_t>(t);
            host_flat_positions[0 * num_tokens + t] = positions[t][0];
            host_flat_positions[1 * num_tokens + t] = positions[t][1];
            host_flat_positions[2 * num_tokens + t] = positions[t][2];
        }

        // Payload H2D of chunk token ids/indices/positions; not a host decision round-trip.
        CUDA_CHECK(cudaMemcpyAsync(dev_token_ids.data, token_ids.data(),
                                   num_tokens * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   device_.stream));
        CUDA_CHECK(cudaMemcpyAsync(dev_token_indices.data, host_indices.data(),
                                   num_tokens * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   device_.stream));
        CUDA_CHECK(cudaMemcpyAsync(dev_mrope_positions.data, host_flat_positions.data(),
                                   3 * num_tokens * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   device_.stream));

        // 3. Compute embedding [2560, T]
        ops::embedding(dev_token_ids, model_.token_embedding, embedding, device_.stream);

        // 3b. Scatter visual embeddings if provided
        if (visual_embeddings != nullptr && !chunk_local_scatter_indices.empty()) {
            const auto count = static_cast<std::int32_t>(chunk_local_scatter_indices.size());
            CUDA_CHECK(cudaMemcpyAsync(staging.visual_indices.data,
                                       chunk_local_scatter_indices.data(),
                                       count * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       device_.stream));
            Tensor indices_slice = staging.visual_indices.slice(0, 0, count);
            ops::scatter(*visual_embeddings, indices_slice, embedding, device_.stream);
        }
        stage_ledger_record(device_.stream, FlashNextStageId::Preamble_EmbeddingStaging);

        // 4. Output tensors
        Tensor logits(alloc_.round_tensors().logits.data, DType::BF16, {248'320, 1});
        Tensor final_hidden(alloc_.round_tensors().final_hidden.data, DType::BF16, {2'560, 1});

        // 5. Execute T-wide prefill core
        // Diagnostic hook: NINFER_FLASH_NEXT_TRACE_STAGES=<dir> dumps every stage of every prefill
        // chunk (raw device bytes, chunk<k>_<stage>.bin) through the core's state sink. This is
        // how the sparse-attention shared-memory race was localised in a served process.
        static const char* stage_dir = std::getenv("NINFER_FLASH_NEXT_TRACE_STAGES");
        static int chunk_counter     = 0;
        FlashNextDecodeStateSink stage_sink;
        const FlashNextDecodeStateSink* effective_sink = sink;
        if (stage_dir != nullptr && stage_dir[0] != '\0' && sink == nullptr) {
            const int chunk_id  = chunk_counter++;
            stage_sink.on_state = [chunk_id](std::string_view name, const Tensor& t) {
                std::size_t count = 1;
                for (int d = 0; d < 4; ++d) { if (t.ne[d] > 0) { count *= static_cast<std::size_t>(t.ne[d]); } }
                const std::size_t elem = (t.dtype == DType::FP32 || t.dtype == DType::I32) ? 4 : 2;
                std::vector<unsigned char> host(count * elem);
                cudaMemcpy(host.data(), t.data, host.size(), cudaMemcpyDeviceToHost);
                const std::string path = std::string(stage_dir) + "/chunk" + std::to_string(chunk_id) +
                                         "_" + std::string(name) + ".bin";
                std::FILE* f = std::fopen(path.c_str(), "wb");
                if (f != nullptr) { std::fwrite(host.data(), 1, host.size(), f); std::fclose(f); }
            };
            effective_sink = &stage_sink;
        }
        flash_next_text_prefill_chunk(
            model_, embedding, dev_token_indices, dev_mrope_positions,
            static_cast<std::int32_t>(lane), initial_active_slot, initial_standby_slot,
            gathered_ple, static_cast<std::int32_t>(alloc_.plan().maximum_blocks),
            first_token_index, alloc_.state_view(), alloc_.workspace(), final_hidden, logits,
            device_.stream, effective_sink, alloc_.plan().config.use_qsa_prefill_mma);

        round_in_flight_ = true;
        Tensor hyper_hidden(alloc_.round_tensors().hyper_hidden.data, DType::BF16, {10'240, 1});
        return PendingRound(this, prepared.transaction_id, 1, logits, final_hidden, hyper_hidden);
    } catch (...) {
        pending_is_prefill_chunk_ = false;
        round_in_flight_          = false;
        alloc_.restore_lane_slots(lane, initial_active_slot, initial_standby_slot, device_.stream);
        ledger_.rollback_prepared_prefill_chunk(prepared.transaction_id);
        throw;
    }
}

void FlashNextTextExecutor::commit_transaction(std::uint64_t tx_id,
                                               std::span<const LaneCommitDecision> decisions) {
    round_in_flight_ = false;
    if (pending_is_prefill_chunk_) {
        if (!decisions.empty() && decisions[0].accept) {
            alloc_.commit_row_slot(pending_prefill_lane_, device_.stream);
        } else {
            alloc_.restore_lane_slots(pending_prefill_lane_, pending_prefill_initial_active_slot_,
                                      pending_prefill_initial_standby_slot_, device_.stream);
        }
        pending_is_prefill_chunk_ = false;
    }
    ledger_.commit_round(tx_id, decisions, alloc_, device_.stream);
}

void FlashNextTextExecutor::commit_speculative_transaction(
    std::uint64_t tx_id, std::uint32_t lane_index,
    std::span<const std::int32_t> accepted_tokens) {
    (void)lane_index;
    round_in_flight_          = false;
    pending_is_prefill_chunk_ = false;
    ledger_.commit_speculative_round(tx_id, accepted_tokens, alloc_, device_.stream);
}

void FlashNextTextExecutor::abort_transaction(std::uint64_t tx_id) noexcept {
    round_in_flight_ = false;
    if (pending_is_prefill_chunk_) {
        alloc_.restore_lane_slots(pending_prefill_lane_, pending_prefill_initial_active_slot_,
                                  pending_prefill_initial_standby_slot_, device_.stream);
        pending_is_prefill_chunk_ = false;
    }
    ledger_.abort_round(tx_id);
}

PendingRound FlashNextTextExecutor::execute_speculative_verify_round(
    LaneHandle handle, std::int32_t anchor_token_id, std::span<const std::int32_t> draft_tokens,
    std::int32_t first_token_index, std::array<std::int32_t, 3> first_mrope_position,
    const ops::SamplingConfig& sampling) {
    if (handle.owner() != this) {
        throw std::invalid_argument("FlashNextTextExecutor: cross-executor or invalid owner handle");
    }
    if (round_in_flight_) {
        throw std::logic_error(
            "FlashNextTextExecutor: cannot start speculative round while previous round is in flight");
    }

    const auto num_tokens    = static_cast<std::uint32_t>(1U + draft_tokens.size());
    const std::uint32_t lane = handle.lane_index();

    auto prepared =
        ledger_.begin_speculative_round(handle, anchor_token_id, draft_tokens, first_token_index,
                                        ple_metadata_);

    try {
        pending_is_prefill_chunk_ = false;
        ledger_.sync_tables_if_dirty(alloc_, device_.stream);
        ple_pipeline_.gather_pinned(prepared.ple_indices);

        auto* host_ing = alloc_.host_ingress();
        for (std::uint32_t i = 0; i < num_tokens; ++i) {
            host_ing->token_ids[i]     = (i == 0) ? anchor_token_id : draft_tokens[i - 1];
            host_ing->token_indices[i] = first_token_index + static_cast<std::int32_t>(i);
            for (std::uint32_t d = 0; d < 3; ++d) {
                host_ing->mrope_positions[d * num_tokens + i] =
                    first_mrope_position[d] + static_cast<std::int32_t>(i);
            }
            host_ing->table_rows[i]        = static_cast<std::int32_t>(lane);
            host_ing->source_slots[i]      = alloc_.lane_ring_slot(lane, i);
            host_ing->destination_slots[i] = alloc_.lane_ring_slot(lane, i + 1U);
            host_ing->sampling[i]          = sampling;
        }

        alloc_.workspace().reset();

        CUDA_CHECK(cudaMemcpyAsync(alloc_.device_ingress_ptr(), alloc_.host_ingress(),
                                   sizeof(FlashNextDecodeIngress), cudaMemcpyHostToDevice,
                                   device_.stream));

        Tensor gathered_ple(alloc_.round_tensors().gathered_ple_embedding.data, DType::BF16,
                            {2'560, static_cast<std::int32_t>(num_tokens)});
        CUDA_CHECK(cudaMemcpyAsync(gathered_ple.data, ple_pipeline_.fixed_host_buffer(),
                                   num_tokens * 2'560 * sizeof(std::uint16_t),
                                   cudaMemcpyHostToDevice, device_.stream));

        Tensor token_ids =
            alloc_.round_tensors().token_ids.slice(0, 0, static_cast<std::int32_t>(num_tokens));
        Tensor embedding =
            alloc_.workspace().alloc(DType::BF16, {2'560, static_cast<std::int32_t>(num_tokens)}, 256);
        ops::embedding(token_ids, model_.token_embedding, embedding, device_.stream);

        Tensor token_indices =
            alloc_.round_tensors().token_indices.slice(0, 0, static_cast<std::int32_t>(num_tokens));
        Tensor mrope_positions(alloc_.round_tensors().mrope_positions.data, DType::I32,
                               {static_cast<std::int32_t>(num_tokens), 3});
        Tensor table_rows =
            alloc_.round_tensors().table_rows.slice(0, 0, static_cast<std::int32_t>(num_tokens));
        Tensor source_slots =
            alloc_.round_tensors().source_slots.slice(0, 0, static_cast<std::int32_t>(num_tokens));
        Tensor destination_slots =
            alloc_.round_tensors().destination_slots.slice(0, 0, static_cast<std::int32_t>(num_tokens));
        Tensor final_hidden =
            alloc_.round_tensors().final_hidden.slice(1, 0, static_cast<std::int32_t>(num_tokens));
        Tensor hyper_hidden =
            alloc_.round_tensors().hyper_hidden.slice(1, 0, static_cast<std::int32_t>(num_tokens));
        Tensor logits =
            alloc_.round_tensors().logits.slice(1, 0, static_cast<std::int32_t>(num_tokens));

        const auto max_blocks   = static_cast<std::int32_t>(alloc_.plan().maximum_blocks);
        const auto bucket_index = flash_next_decode_graph_select_bucket(decode_graphs_.buckets,
                                                                        prepared.max_active_blocks);
        const auto bucket_blocks = static_cast<std::int32_t>(decode_graphs_.buckets.blocks[bucket_index]);

        flash_next_text_decode_core(model_, embedding, token_indices, mrope_positions, table_rows,
                                    source_slots, destination_slots, gathered_ple, max_blocks,
                                    bucket_blocks, alloc_.state_view(), alloc_.workspace(),
                                    final_hidden, logits, device_.stream, nullptr, &hyper_hidden,
                                    /*aliased_recurrent_scan=*/true);

        Tensor sampled_tokens =
            alloc_.round_tensors().sampled_tokens.slice(0, 0, static_cast<std::int32_t>(num_tokens));
        constexpr std::int32_t kSemanticTokenDomain = 248'077;
        ops::sample(logits, sampled_tokens, kSemanticTokenDomain,
                    alloc_.device_sampling_configs(), token_indices,
                    ops::kSamplePurposeDecode, sampling_workspace_, device_.stream);

        CUDA_CHECK(cudaMemcpyAsync(alloc_.host_egress(), alloc_.device_egress_ptr(),
                                   sizeof(FlashNextDecodeEgress), cudaMemcpyDeviceToHost,
                                   device_.stream));

        round_completion_.record(device_.stream);
        {
            const auto wait_started = std::chrono::steady_clock::now();
            round_completion_.synchronize();
            round_device_wait_ns_ += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wait_started)
                    .count());
        }

        std::array<std::int32_t, 8> sampled_tokens_host{};
        const auto* host_egr = alloc_.host_egress();
        for (std::uint32_t i = 0; i < num_tokens; ++i) {
            sampled_tokens_host[i] = host_egr->sampled_tokens[i];
        }

        round_in_flight_ = true;
        return PendingRound(this, prepared.transaction_id, num_tokens, logits, final_hidden,
                            hyper_hidden, std::span(sampled_tokens_host.data(), num_tokens));
    } catch (...) {
        ledger_.rollback_prepared_round(prepared.transaction_id);
        round_in_flight_ = false;
        throw;
    }
}

void FlashNextTextExecutor::draft_mtp_tokens(LaneHandle handle, std::int32_t token_id,
                                             std::int32_t token_index,
                                             std::array<std::int32_t, 3> mrope_pos,
                                             const Tensor& backbone_hidden,
                                             std::uint32_t draft_count,
                                             std::span<std::int32_t> out_draft_tokens) {
    if (!model_.mtp.has_value()) {
        throw std::logic_error("FlashNextTextExecutor: cannot draft MTP tokens without MTP model view");
    }
    if (draft_count == 0) { return; }
    if (out_draft_tokens.size() < draft_count) {
        throw std::invalid_argument("FlashNextTextExecutor: out_draft_tokens too small");
    }

    const std::uint32_t lane = handle.lane_index();
    const auto max_blocks    = static_cast<std::int32_t>(alloc_.plan().maximum_blocks);
    const auto active_blocks = std::min(max_blocks, (token_index + 1) / 4);

    Tensor selected_blocks(mtp_selected_blocks_->p, DType::I32, {512, 1});
    Tensor selected_counts(mtp_selected_counts_->p, DType::I32, {1});
    const std::int32_t draft_logits_rows =
        model_.proposal.has_value() ? model_.proposal->head.n : 248'320;
    Tensor draft_logits(mtp_draft_logits_->p, DType::BF16, {draft_logits_rows, 1});
    Tensor draft_tokens_tensor(mtp_draft_tokens_->p, DType::I32, {1});
    Tensor carried_hidden(mtp_carried_hidden_->p, DType::BF16, {10'240, 1});

    for (std::uint32_t k = 0; k < draft_count; ++k) {
        mtp_workspace_->reset();

        const std::int32_t cur_token_index = token_index + static_cast<std::int32_t>(k);
        std::array<std::int32_t, 3> cur_mrope_pos = {
            mrope_pos[0] + static_cast<std::int32_t>(k),
            mrope_pos[1] + static_cast<std::int32_t>(k),
            mrope_pos[2] + static_cast<std::int32_t>(k)
        };

        auto* host_ing = alloc_.host_ingress();
        host_ing->token_ids[0]     = token_id;
        host_ing->token_indices[0] = cur_token_index;
        for (std::uint32_t d = 0; d < 3; ++d) {
            host_ing->mrope_positions[d] = cur_mrope_pos[d];
        }
        host_ing->table_rows[0] = static_cast<std::int32_t>(lane);
        host_ing->sampling[0]   = ops::SamplingConfig{};

        CUDA_CHECK(cudaMemcpyAsync(alloc_.device_ingress_ptr(), host_ing,
                                   sizeof(FlashNextDecodeIngress), cudaMemcpyHostToDevice,
                                   device_.stream));

        Tensor dev_token_indices = alloc_.round_tensors().token_indices.slice(0, 0, 1);
        Tensor dev_mrope_positions(alloc_.round_tensors().mrope_positions.data, DType::I32, {1, 3});
        Tensor dev_table_rows = alloc_.round_tensors().table_rows.slice(0, 0, 1);

        Tensor cur_token_id_tensor = alloc_.round_tensors().token_ids.slice(0, 0, 1);
        Tensor cur_embedding(mtp_input_embedding_->p, DType::BF16, {2'560, 1});
        ops::embedding(cur_token_id_tensor, model_.token_embedding, cur_embedding, device_.stream);

        const Tensor& step_backbone = (k == 0) ? backbone_hidden : carried_hidden;
        flash_next_mtp_step(
            model_, cur_embedding, step_backbone, dev_token_indices, dev_mrope_positions,
            dev_table_rows, selected_blocks, selected_counts, *mtp_cache_,
            *mtp_workspace_, draft_logits, draft_tokens_tensor, device_.stream, nullptr,
            &carried_hidden);

        std::int32_t sampled_tok = 0;
        CUDA_CHECK(cudaMemcpyAsync(&sampled_tok, draft_tokens_tensor.data, sizeof(std::int32_t),
                                   cudaMemcpyDeviceToHost, device_.stream));
        device_.synchronize();

        out_draft_tokens[k] = sampled_tok;
        token_id            = sampled_tok;
    }
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
