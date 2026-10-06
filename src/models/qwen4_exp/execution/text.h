#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/tensor.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/frontend.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen4_exp/execution/ple.h"
#include "models/qwen4_exp/execution/workspace.h"
#include "models/qwen4_exp/program/round_buffers.h"
#include "models/qwen4_exp/state/decoder_state.h"
#include "models/qwen4_exp/state/state_image.h"
#include "ninfer/ops/qsa_indexer.h"
#include "ninfer/ops/sampling.h"
#include "runtime/contract/timing.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace ninfer::models::qwen4_exp::execution {

using qwen3_5::execution::VisionContext;
using qwen3_5::execution::VisionPrefillSession;
using qwen3_5::execution::VisionChunk;

// Device readout of the first generated token's logits (ops::candidate_logprobs), enqueued at
// the sampling site.
struct FirstTokenReadout {
    std::optional<Tensor> candidate_ids;  // I32 [N]
    std::optional<Tensor> allowed;        // I32 [mask words], the position's structured-output mask
    Tensor sampled_out;                   // FP32 [2, 1]
    std::optional<Tensor> candidates_out; // FP32 [2, N, 1]
    void* host        = nullptr;          // pinned destination, sampled_out then candidates_out
    std::size_t bytes = 0;
};

// Device readout of the next-token distribution at chosen prompt positions, run on each prefill
// chunk that contains some of them: the four-stream rows are gathered, mixed, projected through
// the output head a tile at a time, and each column resolved by ops::candidate_logprobs into its
// own block of (2 + 2 * candidates) floats at `readout` / `host`.
struct PromptReadout {
    // Reasoning feature readout: the final mixed BF16 row at this absolute prompt position is
    // copied to `feature_host` (pinned, hidden-size values) by the chunk that computes it.
    std::optional<std::uint32_t> feature_position;
    std::uint16_t* feature_host = nullptr;
    std::span<const std::uint32_t> positions; // ascending absolute prompt positions
    const std::int32_t* next_ids = nullptr;   // device I32 [positions.size()]: token p+1 of each
    std::optional<Tensor> candidate_ids;      // I32 [candidates]
    std::size_t candidates = 0;
    float* readout         = nullptr; // device, positions.size() blocks
    float* host            = nullptr; // pinned mirror of the same blocks

    [[nodiscard]] std::size_t block_floats() const noexcept { return 2U + 2U * candidates; }
};

// Enqueues the device readout of one logit column for the token in `sampled` (I32 [1]).
void enqueue_first_token_readout(const FirstTokenReadout& readout, const Tensor& logits,
                                 const Tensor& sampled, std::int32_t token_domain,
                                 cudaStream_t stream);

struct PrefillChunkResult {
    std::uint32_t processed_tokens = 0;
    bool finalized                 = false;
    runtime::ExecutionTiming timing;
};

// One exact-B ordinary decode step's device inputs. rope_positions is planar I32 [B,3]; the PLE
// rows are the B x heads gathered rows in ops::ple_ngram_decode operand order.
struct OrdinaryDecodeInputs {
    Tensor ids;
    Tensor cache_positions;
    Tensor rope_positions;
    Tensor kv_table_rows;
    Tensor state_source_slots;
    Tensor state_destination_slots;
    Tensor ple_codes;
    Tensor ple_scales;
    // Bounds every row's complete indexer blocks floor((position + 1) / 4).
    ops::QsaIndexerSelectEnvelope indexer;
    // Verification packs W consecutive tokens per row. kv_table_rows expands to W*B entries;
    // sequence_rows and state selectors retain B entries. W=1 is ordinary decode.
    std::int32_t width = 1;
    Tensor sequence_rows;
    Tensor valid_columns;
    const GdnReplayRecords* replay = nullptr;
};

// The fixed Qwen4Exp Text schedule: four-stream hyper-connections around 48 blocks (QSA attention
// or Gated DeltaNet, then the NVFP4-bank MoE), the per-layer n-gram embedding injected before
// block `ple.layer`, and the final mixer before the independent output head.
class TextContext {
public:
    TextContext(DeviceContext& ctx, const Parameters& parameters, WorkspaceArena& work,
                PleGather& ple, qwen4_exp::PagedKVCacheView kv, StateImageDevicePool& state,
                qwen4_exp::RoundState& io, Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                std::uint32_t text_kv_base, const qwen4_exp::PagedKVCache* batch_kv = nullptr);

    TextContext(const TextContext&)            = delete;
    TextContext& operator=(const TextContext&) = delete;

    void set_sampling(const ops::SamplingConfig* config) noexcept { sampling_config_ = config; }

    void set_rope_delta(std::int32_t delta) noexcept { rope_delta_ = delta; }

    void set_state_slots(std::int32_t source_slot, std::int32_t destination_slot);

    // Pinned host destination for the final prompt position's full logits, or null.
    void set_first_token_logit_capture(std::uint16_t* host) noexcept {
        first_token_logits_host_ = host;
    }

    void set_first_token_readout(const FirstTokenReadout* readout) noexcept {
        first_token_readout_ = readout;
    }

    void set_prompt_readout(const PromptReadout* readout) noexcept { prompt_readout_ = readout; }

    void set_prefill_split_frontier(std::int64_t position) noexcept {
        prefill_split_frontier_ = position;
    }

    // Four-stream hidden [stream_width] of the split frontier's last column, when a chunk ends
    // there.
    void set_rewrite_checkpoint_hidden_output(Tensor* output) noexcept {
        rewrite_checkpoint_hidden_output_ = output;
    }

    // Executes at most one prefill_chunk-wide slice of [begin, begin + nominal_length) of
    // `full_ids` (the sequence ledger from position 0, which also supplies the n-gram history).
    // The four-stream hidden of the slice's last column is left in io.prefill_tail. With
    // `finalize_at_end` the last prompt column is mixed, projected and sampled into io.token.
    [[nodiscard]] PrefillChunkResult
    prefill_chunk(std::span<const int> full_ids, std::uint32_t begin, std::uint32_t nominal_length,
                  bool finalize_at_end, const PreparedPromptData* prompt = nullptr,
                  VisionPrefillSession* vision = nullptr);

    // One token for each of B independent sequences; writes the four-stream hidden [stream,B]
    // and the logits [vocab,B]. Graph capturable for a fixed B and indexer envelope.
    void ordinary_decode_batch(const OrdinaryDecodeInputs& inputs, Tensor& hidden, Tensor& logits);
    void verify_batch(const OrdinaryDecodeInputs& inputs, Tensor& hidden, Tensor& logits);

    // Final mixer and output head of four-stream hidden columns [stream,N] into logits [vocab,N].
    void project_streams(const Tensor& streams, Tensor& logits);

    void set_mtp_cache(qwen4_exp::PagedKVCacheView cache,
                       const qwen4_exp::PagedKVCache* batch_cache = nullptr) noexcept {
        mtp_kv_       = cache;
        mtp_batch_kv_ = batch_cache;
    }

    // One MTP layer over next-token embeddings and the preceding target/recursive MTP streams.
    // Teacher extension writes only the attention/indexer keys used by subsequent proposals.
    void mtp_forward(const Tensor& embedding, const Tensor& previous_streams,
                     const Tensor& positions, const Tensor& rope_positions, Tensor& streams,
                     Tensor* logits, const OrdinaryDecodeInputs* batch = nullptr,
                     bool teacher = false);

private:
    void project(const Tensor& x, const LinearParameters& parameters, Tensor& out);
    void decode_batch(const OrdinaryDecodeInputs& inputs, Tensor& hidden, Tensor& logits);
    void run_layers(workspace::LayerRoots& roots, const Tensor& ple_codes, const Tensor& ple_scales,
                    bool prefill, std::int32_t tokens);

    void inject_ple(workspace::LayerRoots& roots, const Tensor& codes, const Tensor& scales,
                    bool prefill, std::int32_t tokens);
    void attention(const AttentionParameters& p, std::uint32_t layer, workspace::LayerRoots& roots,
                   bool prefill, std::int32_t tokens);
    void gdn(const GdnParameters& p, std::uint32_t layer, workspace::LayerRoots& roots,
             bool prefill, std::int32_t tokens);
    void run_prompt_readout(const Tensor& streams, std::int64_t begin, std::int32_t length);

    DeviceContext& ctx_;
    const Parameters& parameters_;
    const TextConfig& config_;
    WorkspaceArena& work_;
    PleGather& ple_;
    qwen4_exp::PagedKVCacheView kv_;
    const qwen4_exp::PagedKVCache* batch_kv_ = nullptr;
    qwen4_exp::PagedKVCacheView mtp_kv_;
    const qwen4_exp::PagedKVCache* mtp_batch_kv_ = nullptr;
    bool mtp_active_                             = false;
    bool mtp_teacher_                            = false;
    StateImageDevicePool& state_;
    qwen4_exp::RoundState& io_;
    Tensor& prefill_hidden_;
    std::uint32_t prefill_chunk_ = 0;
    std::uint32_t text_kv_base_  = 0;
    std::int32_t rope_delta_     = 0;

    // Bindings of the call in flight.
    const Tensor* positions_                        = nullptr;
    const Tensor* rope_positions_                   = nullptr;
    const OrdinaryDecodeInputs* decode_             = nullptr;
    ops::QsaIndexerSelectEnvelope indexer_envelope_ = {};
    std::int32_t state_source_slot_                 = 0;
    std::int32_t state_destination_slot_            = 0;
    std::int64_t prefill_split_frontier_            = -1;
    Tensor* rewrite_checkpoint_hidden_output_       = nullptr;
    const ops::SamplingConfig* sampling_config_     = nullptr;
    std::uint16_t* first_token_logits_host_         = nullptr;
    const FirstTokenReadout* first_token_readout_   = nullptr;
    const PromptReadout* prompt_readout_            = nullptr;
};

} // namespace ninfer::models::qwen4_exp::execution
