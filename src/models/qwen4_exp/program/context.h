#pragma once
#include "models/qwen4_exp/program/internal.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "models/qwen4_exp/execution/ple.h"
#include "models/qwen4_exp/execution/text.h"
#include "models/qwen4_exp/frontend.h"
#include "models/qwen4_exp/state/decoder_state.h"
#include "models/qwen4_exp/state/state_image.h"
#include "ninfer/ops/qsa_indexer.h"
#include "ninfer/ops/sampling.h"

#include <cstdint>
#include <optional>
#include <span>

namespace ninfer::models::qwen4_exp::execution {

struct ExecutionCore {
    DeviceContext& device;
    const execution::Parameters& parameters;
    WorkspaceArena& work;
    StateImageDevicePool& state;
    PleGather& ple;
    qwen4_exp::RoundState& io;
    Tensor& prefill_hidden;
    std::uint32_t prefill_chunk;
};

struct PrefillContext {
    ExecutionCore execution;
    qwen4_exp::PagedKVCacheView text_kv;
    const qwen4_exp::PagedKVCache& text_cache;
    std::uint32_t text_kv_base;
    const ops::SamplingConfig* sampling;
    Tensor* rewrite_checkpoint_hidden;
    std::int32_t state_source_slot      = 0;
    std::int32_t state_destination_slot = 0;
    // Host full-column copy of the first generated token's logits, or null.
    std::uint16_t* first_token_logits_host       = nullptr;
    const FirstTokenReadout* first_token_readout = nullptr;
    const PromptReadout* prompt_readout          = nullptr;
};

struct OrdinaryBatchContext {
    ExecutionCore execution;
    const qwen4_exp::PagedKVCache& text_cache;
    qwen4_exp::OrdinaryDecodeState& frame;
    const qwen4_exp::OrdinaryDecodeIngress& host_ingress;
    qwen4_exp::OrdinaryDecodeEgress& host_egress;
    Tensor& continuation_hidden_store;
};

[[nodiscard]] PrefillChunkResult prefill_text_chunk(PrefillContext& state,
                                                    std::span<const TokenId> ids,
                                                    std::uint32_t nominal_length,
                                                    std::optional<std::uint32_t> split_frontier,
                                                    bool finalize_at_end);

// Samples the next token from a continuation's four-stream hidden (an exact prefix hit has no
// prompt suffix to run): final mixer, output head and the sampler at `absolute_position`.
void sample_from_hidden(PrefillContext& state, const Tensor& streams,
                        std::int32_t absolute_position, std::int32_t purpose);

// Executes one exact-B ordinary decode traversal. All request rows enter through the stable
// ordinary ingress, share one model schedule, publish their continuation hidden by selector, and
// leave through one compact egress transfer.
void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::QsaIndexerSelectEnvelope envelope,
                                   DecodeGraphDefinition& definition);
void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           ops::QsaIndexerSelectEnvelope envelope,
                           DecodeGraphExecutable* executable);

} // namespace ninfer::models::qwen4_exp::execution
