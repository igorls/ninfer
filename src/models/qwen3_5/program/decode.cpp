#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/graph_execution.h"
#include "core/nvtx.h"
#include "core/device.h"
#include "ninfer/ops/candidate_logprobs.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scatter.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
namespace {

auto ordinary_batch_body(OrdinaryBatchContext& state, std::int32_t batch_size,
                         ops::CausalAttentionExecutionEnvelope envelope) {
    return [&state, batch_size, envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency)) {
            throw std::logic_error("ordinary decode batch state is incomplete");
        }

        qwen3_5::OrdinaryDecodeState& ordinary = state.frame;
        CUDA_CHECK(cudaMemcpyAsync(ordinary.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_5::OrdinaryDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                         {}, state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache);

        Tensor tokens             = ordinary.tokens.slice(0, 0, batch_size);
        Tensor cache_positions    = ordinary.cache_positions.slice(0, 0, batch_size);
        Tensor rope_positions     = ordinary.rope_positions.slice(0, 0, batch_size);
        Tensor kv_rows            = ordinary.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor state_sources      = ordinary.state_source_slots.slice(0, 0, batch_size);
        Tensor state_destinations = ordinary.state_destination_slots.slice(0, 0, batch_size);
        Tensor hidden             = ordinary.hidden.slice(1, 0, batch_size);
        Tensor logits             = ordinary.logits.slice(1, 0, batch_size);
        Tensor sampled            = ordinary.sampled_tokens.slice(0, 0, batch_size);

        card.ordinary_decode_batch(tokens, cache_positions, rope_positions, kv_rows, state_sources,
                                   state_destinations, envelope, hidden, logits);
        ops::scatter(hidden, state_destinations, state.continuation_hidden_store,
                     state.execution.device.stream);
        ops::sample(logits, sampled,
                    dimension(state.execution.parameters.model.resources().public_token_count),
                    ordinary.sampling, cache_positions, ops::kSamplePurposeDecode,
                    state.execution.work, state.execution.device.stream);
        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, ordinary.egress.data,
                                   sizeof(qwen3_5::OrdinaryDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

} // namespace

void capture_ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   DecodeGraphDefinition& definition) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    capture_graph(state, definition, body);
}

void ordinary_decode_batch(OrdinaryBatchContext& state, std::int32_t batch_size,
                           ops::CausalAttentionExecutionEnvelope envelope,
                           DecodeGraphExecutable* executable) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    run_prepared(state, executable, body);
}

} // namespace ninfer::models::qwen3_5::execution

namespace ninfer::models::qwen3_5::detail {

namespace {

DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t batch_size,
                                         std::uint32_t frontier, const char* label);

DecodeGraphTopology& select_graph_topology(DecodeGraphFamily& family, std::uint32_t topology_class,
                                           const char* label);

DecodeGraphExecutable& install_graph_profile(DecodeGraphFamily& family, DecodeGraphProfile& profile,
                                             const char* label);

DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t batch_size,
                                         std::uint32_t frontier, const char* label) {
    const auto it = std::find_if(
        family.profiles.begin(), family.profiles.end(), [&](const DecodeGraphProfile& profile) {
            return profile.batch_size == batch_size && profile.min_execution_frontier <= frontier &&
                   frontier <= profile.max_execution_frontier;
        });
    if (it == family.profiles.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage is incomplete");
    }
    return *it;
}

DecodeGraphTopology& select_graph_topology(DecodeGraphFamily& family, std::uint32_t topology_class,
                                           const char* label) {
    const auto it = std::find_if(family.topologies.begin(), family.topologies.end(),
                                 [topology_class](const DecodeGraphTopology& topology) {
                                     return topology.topology_class == topology_class;
                                 });
    if (it == family.topologies.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph topology is unavailable");
    }
    return *it;
}

DecodeGraphExecutable& install_graph_profile(DecodeGraphFamily& family, DecodeGraphProfile& profile,
                                             const char* label) {
    DecodeGraphTopology& topology   = select_graph_topology(family, profile.topology_class, label);
    const std::size_t profile_index = static_cast<std::size_t>(&profile - family.profiles.data());
    if (topology.installed_profile != profile_index) {
        topology.executable.update(profile.definition);
        topology.installed_profile = profile_index;
    }
    return topology.executable;
}

} // namespace

void ProgramImpl::install_sampling(SequenceState& sequence, RequestControl& request,
                                   const ops::SamplingConfig& config,
                                   std::span<const TokenId> prompt) {
    Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(sequence.lane), 1)
                        .view({dimension(parameters.model.resources().public_token_count)});
    request.sampling_host = config;
    if (request.output_constraint) {
        request.sampling_host.allowed_tokens =
            upload_constraint_mask(sequence.lane, 0, request.output_constraint->next_mask());
        request.sampling_host.allowed_tokens_column_stride = constraint_masks.ne[0];
    }
    request.logprobs_device_readout = request.logprobs.enabled && request.logprobs.top == 0;
    request.logprob_readout_columns = 0;
    if (request.logprobs_device_readout) {
        if (!logprob_readout_host) {
            logprob_readout_host.emplace(kLogprobReadoutFloats * max_concurrency * sizeof(float));
        }
        if (!request.logprobs.candidates.empty()) {
            Tensor ids =
                logprob_candidate_ids.slice(1, static_cast<std::int32_t>(sequence.lane), 1);
            CUDA_CHECK(cudaMemcpyAsync(ids.data, request.logprobs.candidates.data(),
                                       request.logprobs.candidates.size() * sizeof(TokenId),
                                       cudaMemcpyHostToDevice, device.stream));
        }
    }
    request.speculative_stats = SpeculativeStats{
        .backend               = speculative_backend,
        .enabled               = speculative_backend != SpeculativeBackend::None,
        .draft_window          = draft_window,
        .accepted_per_position = std::vector<std::uint64_t>(draft_window, 0),
    };
    const bool penalties = request.sampling_host.presence_penalty != 0.0F ||
                           request.sampling_host.frequency_penalty != 0.0F ||
                           request.sampling_host.repetition_penalty != 1.0F;
    if (penalties) { CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), device.stream)); }
    request.sampling_host.token_counts =
        penalties ? static_cast<std::int32_t*>(counts.data) : nullptr;
    request.sampling_host.prompt_presence = nullptr;
    if (config.repetition_penalty != 1.0F) {
        // Membership over the complete prepared prompt, including reused tokens; media
        // placeholders outside the sampled domain never match.
        Tensor bits       = prompt_presence.slice(1, static_cast<std::int32_t>(sequence.lane), 1);
        const auto domain = static_cast<TokenId>(counts.ne[0]);
        request.prompt_presence_host.assign(static_cast<std::size_t>(bits.ne[0]), 0);
        for (const TokenId token : prompt) {
            if (token < 0 || token >= domain) { continue; }
            auto& word = request.prompt_presence_host[static_cast<std::size_t>(token) / 32U];
            word       = static_cast<std::int32_t>(static_cast<std::uint32_t>(word) |
                                                   (1U << (static_cast<std::uint32_t>(token) % 32U)));
        }
        CUDA_CHECK(cudaMemcpyAsync(bits.data, request.prompt_presence_host.data(), bits.bytes(),
                                   cudaMemcpyHostToDevice, device.stream));
        request.sampling_host.prompt_presence = static_cast<const std::int32_t*>(bits.data);
    }
    Tensor config_lane = sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1);
    CUDA_CHECK(cudaMemcpyAsync(config_lane.data, &request.sampling_host,
                               sizeof(request.sampling_host), cudaMemcpyHostToDevice,
                               device.stream));
}

const std::int32_t* ProgramImpl::upload_constraint_mask(std::uint32_t lane, std::uint32_t column,
                                                        std::span<const std::int32_t> mask) {
    if (lane >= static_cast<std::uint32_t>(constraint_masks.ne[2]) ||
        column >= static_cast<std::uint32_t>(constraint_masks.ne[1])) {
        throw std::logic_error("structured output mask column is out of range");
    }
    Tensor row = constraint_masks.slice(2, static_cast<std::int32_t>(lane), 1)
                     .slice(1, static_cast<std::int32_t>(column), 1);
    if (mask.size_bytes() != row.bytes()) {
        throw std::logic_error("structured output mask shape mismatch");
    }
    CUDA_CHECK(cudaMemcpyAsync(row.data, mask.data(), mask.size_bytes(), cudaMemcpyHostToDevice,
                               device.stream));
    return static_cast<const std::int32_t*>(row.data);
}

std::uint16_t* ProgramImpl::token_logits_capture(const RequestControl& request) {
    if (!request.logprobs.enabled || request.logprobs_device_readout) { return nullptr; }
    if (!token_logits_host) {
        token_logits_host.emplace(
            static_cast<std::size_t>(parameters.model.resources().public_token_count) *
            sizeof(std::uint16_t));
    }
    return static_cast<std::uint16_t*>(token_logits_host->data());
}

void ProgramImpl::record_round_logprobs(RequestControl& request, const Tensor* column,
                                        TokenId token) {
    request.round_logprobs.clear();
    std::uint16_t* const host = token_logits_capture(request);
    if (host == nullptr) { return; }
    const std::int32_t domain = dimension(parameters.model.resources().public_token_count);
    const auto rows           = static_cast<std::size_t>(domain);
    if (column != nullptr) {
        if (column->dtype != DType::BF16 || column->ne[0] < domain || column->data == nullptr) {
            throw std::logic_error("token logprobs: target logit column has an invalid shape");
        }
        CUDA_CHECK(cudaMemcpyAsync(host, column->data, rows * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToHost, device.stream));
        device.synchronize();
    }
    const std::span<const std::int32_t> allowed = request.output_constraint
                                                      ? request.output_constraint->current_mask()
                                                      : std::span<const std::int32_t>{};
    request.round_logprobs.push_back(runtime::compute_token_logprobs(
        std::span<const std::uint16_t>(host, rows), allowed, token, request.logprobs));
}

std::span<const TokenLogprobs> ProgramImpl::round_token_logprobs(std::uint32_t lane) const {
    if (lane >= max_concurrency) { throw std::out_of_range("token logprobs lane is out of range"); }
    return requests[lane].round_logprobs;
}

void ProgramImpl::enqueue_round_logprobs(const SequenceState& sequence, RequestControl& request,
                                         const Tensor& logits, const Tensor& sampled,
                                         std::uint32_t columns) {
    request.logprob_readout_columns = 0;
    if (!request.logprobs_device_readout || columns == 0) { return; }
    if (columns > kLogprobReadoutColumns) {
        throw std::logic_error("token logprobs: round exceeds the readout capacity");
    }
    const std::int32_t domain = dimension(parameters.model.resources().public_token_count);
    const auto lane           = static_cast<std::int32_t>(sequence.lane);
    const auto count          = static_cast<std::int32_t>(request.logprobs.candidates.size());
    const auto cols           = static_cast<std::int32_t>(columns);
    const runtime::TokenLogprobReadoutLayout layout{
        .columns = columns, .candidates = request.logprobs.candidates.size()};
    Tensor lane_readout = logprob_readout.slice(1, lane, 1);
    Tensor sampled_out(lane_readout.data, DType::FP32, {cols, 2});
    std::optional<Tensor> candidate_ids;
    std::optional<Tensor> candidates_out;
    if (count > 0) {
        candidate_ids.emplace(logprob_candidate_ids.slice(1, lane, 1).data, DType::I32,
                              std::initializer_list<std::int32_t>{count});
        candidates_out.emplace(static_cast<float*>(lane_readout.data) + 2 * cols, DType::FP32,
                               std::initializer_list<std::int32_t>{cols, count, 2});
    }
    std::optional<Tensor> allowed;
    if (request.sampling_host.allowed_tokens != nullptr) {
        allowed.emplace(const_cast<std::int32_t*>(request.sampling_host.allowed_tokens), DType::I32,
                        std::initializer_list<std::int32_t>{(domain + 31) / 32});
    }
    ops::candidate_logprobs(logits, domain, sampled, candidate_ids ? &*candidate_ids : nullptr,
                            allowed ? &*allowed : nullptr, sampled_out,
                            candidates_out ? &*candidates_out : nullptr, device.stream);
    auto* host = static_cast<float*>(logprob_readout_host->data()) +
                 static_cast<std::size_t>(lane) * kLogprobReadoutFloats;
    CUDA_CHECK(cudaMemcpyAsync(host, lane_readout.data, layout.floats() * sizeof(float),
                               cudaMemcpyDeviceToHost, device.stream));
    request.logprob_readout_columns = columns;
}

void ProgramImpl::collect_round_logprobs(const SequenceState& sequence, RequestControl& request,
                                         std::span<const TokenId> tokens) {
    request.round_logprobs.clear();
    if (!request.logprobs_device_readout) { return; }
    if (tokens.size() > request.logprob_readout_columns) {
        throw std::logic_error("token logprobs: round licensed more tokens than were read out");
    }
    const runtime::TokenLogprobReadoutLayout layout{.columns = request.logprob_readout_columns,
                                                    .candidates =
                                                        request.logprobs.candidates.size()};
    const auto* host = static_cast<const float*>(logprob_readout_host->data()) +
                       static_cast<std::size_t>(sequence.lane) * kLogprobReadoutFloats;
    const std::span<const float> readout(host, layout.floats());
    for (std::size_t column = 0; column < tokens.size(); ++column) {
        request.round_logprobs.push_back(runtime::assemble_token_logprobs(
            tokens[column], readout, layout, column, request.logprobs.candidates));
    }
    request.logprob_readout_columns = 0;
}

execution::FirstTokenReadout ProgramImpl::first_token_readout(const SequenceState& sequence,
                                                              RequestControl& request) {
    execution::FirstTokenReadout out;
    request.logprob_readout_columns = 0;
    if (!request.logprobs_device_readout) { return out; }
    const std::int32_t domain = dimension(parameters.model.resources().public_token_count);
    const auto lane           = static_cast<std::int32_t>(sequence.lane);
    const auto count          = static_cast<std::int32_t>(request.logprobs.candidates.size());
    Tensor lane_readout       = logprob_readout.slice(1, lane, 1);
    out.sampled_out           = Tensor(lane_readout.data, DType::FP32, {1, 2});
    if (count > 0) {
        out.candidate_ids.emplace(logprob_candidate_ids.slice(1, lane, 1).data, DType::I32,
                                  std::initializer_list<std::int32_t>{count});
        out.candidates_out.emplace(static_cast<float*>(lane_readout.data) + 2, DType::FP32,
                                   std::initializer_list<std::int32_t>{1, count, 2});
    }
    if (request.sampling_host.allowed_tokens != nullptr) {
        out.allowed.emplace(const_cast<std::int32_t*>(request.sampling_host.allowed_tokens),
                            DType::I32, std::initializer_list<std::int32_t>{(domain + 31) / 32});
    }
    const runtime::TokenLogprobReadoutLayout layout{
        .columns = 1, .candidates = request.logprobs.candidates.size()};
    out.host = static_cast<float*>(logprob_readout_host->data()) +
               static_cast<std::size_t>(lane) * kLogprobReadoutFloats;
    out.bytes                       = layout.floats() * sizeof(float);
    request.logprob_readout_columns = 1;
    return out;
}

void ProgramImpl::install_prompt_readout(const SequenceState& sequence, RequestControl& request,
                                         std::span<const TokenId> prompt) {
    request.prompt_logprobs.clear();
    request.prompt_readout = execution::PromptReadout{};
    const auto& positions  = request.logprobs.prompt_positions;
    if (positions.empty()) { return; }
    if (!request.logprobs_device_readout || positions.size() > kMaximumPromptReadouts ||
        static_cast<std::size_t>(positions.back()) + 1U >= prompt.size()) {
        throw std::logic_error("prompt position logprobs were not validated at submission");
    }
    if (!logprob_prompt_readout_host) {
        logprob_prompt_readout_host.emplace(kLogprobPromptReadoutFloats * max_concurrency *
                                            sizeof(float));
    }
    const auto lane            = static_cast<std::int32_t>(sequence.lane);
    std::vector<TokenId>& next = request.prompt_readout_next_ids;
    next.resize(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i) { next[i] = prompt[positions[i] + 1U]; }
    Tensor ids = logprob_prompt_next_ids.slice(1, lane, 1);
    CUDA_CHECK(cudaMemcpyAsync(ids.data, next.data(), next.size() * sizeof(TokenId),
                               cudaMemcpyHostToDevice, device.stream));
    execution::PromptReadout& readout = request.prompt_readout;
    readout.positions                 = positions;
    readout.next_ids                  = static_cast<const std::int32_t*>(ids.data);
    readout.candidates                = request.logprobs.candidates.size();
    if (readout.candidates != 0) {
        readout.candidate_ids.emplace(
            logprob_candidate_ids.slice(1, lane, 1).data, DType::I32,
            std::initializer_list<std::int32_t>{static_cast<std::int32_t>(readout.candidates)});
    }
    readout.readout = static_cast<float*>(logprob_prompt_readout.slice(1, lane, 1).data);
    readout.host    = static_cast<float*>(logprob_prompt_readout_host->data()) +
                   static_cast<std::size_t>(lane) * kLogprobPromptReadoutFloats;
}

void ProgramImpl::collect_prompt_readout(RequestControl& request) {
    request.prompt_logprobs.clear();
    const execution::PromptReadout& readout = request.prompt_readout;
    if (readout.positions.empty()) { return; }
    const runtime::TokenLogprobReadoutLayout layout{.columns = 1, .candidates = readout.candidates};
    for (std::size_t i = 0; i < readout.positions.size(); ++i) {
        request.prompt_logprobs.push_back(runtime::assemble_token_logprobs(
            request.prompt_readout_next_ids[i],
            std::span<const float>(readout.host + i * layout.floats(), layout.floats()), layout, 0,
            request.logprobs.candidates));
    }
}

std::span<const TokenLogprobs> ProgramImpl::prompt_token_logprobs(std::uint32_t lane) const {
    if (lane >= max_concurrency) {
        throw std::out_of_range("prompt logprobs lane is out of range");
    }
    return requests[lane].prompt_logprobs;
}

void ProgramImpl::copy_tail(SequenceState& sequence, const Tensor& source) {
    if (source.dtype != DType::BF16 ||
        source.ne[0] != dimension(parameters.model.config().text.hidden_size) ||
        source.ne[1] != 1) {
        throw std::logic_error("target tail hidden has an invalid shape");
    }
    CUDA_CHECK(cudaMemcpyAsync(sequence.tail_hidden.data, source.data, sequence.tail_hidden.bytes(),
                               cudaMemcpyDeviceToDevice, device.stream));
    sequence.tail_hidden_valid = true;
}

void ProgramImpl::copy_round_token() {
    CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.token.data, sizeof(TokenId), cudaMemcpyDeviceToHost,
                               device.stream));
}

void ProgramImpl::mark_workspace_usage(std::size_t phase_bytes) noexcept {
    workspace_logical_peak_bytes = std::max(workspace_logical_peak_bytes, phase_bytes);
}

void ProgramImpl::enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                                std::span<const std::uint32_t> starts,
                                                std::span<const std::uint32_t> counts) {
    if (!is_masked_draft_backend(speculative_backend) || !dflash || !io.dflash_decode ||
        lanes.empty() || lanes.size() > max_concurrency || starts.size() != lanes.size() ||
        counts.size() != lanes.size()) {
        throw std::logic_error("DFlash context append has invalid membership");
    }

    std::uint32_t minimum_count = draft_window + 1U;
    std::uint32_t maximum_count = 0;
    *dflash_host_ingress        = {};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || counts[row] == 0 || counts[row] > draft_window + 1U ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("DFlash context append contains an invalid row");
        }
        SequenceState& sequence   = active_sequence(lane);
        const std::uint32_t start = starts[row];
        const std::uint64_t end64 = static_cast<std::uint64_t>(start) + counts[row];
        const std::uint32_t end   = static_cast<std::uint32_t>(end64);
        if (!sequence.kv || text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            (backend_kv_cache() && (!sequence.kv->backend ||
                                    backend_kv_addresses->bound_row(*sequence.kv->backend) < 0)) ||
            end64 > capacity) {
            throw std::logic_error("DFlash context append is outside retained target storage");
        }
        dflash_host_ingress->context_frontiers[row] =
            checked_i32(start, "DFlash append context frontier");
        dflash_host_ingress->execution_frontiers[row] =
            checked_i32(end, "DFlash append target frontier");
        dflash_host_ingress->dflash_kv_table_rows[row] =
            sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend) : 0;
        dflash_host_ingress->active_lanes[row]            = static_cast<std::int32_t>(lane);
        const StateImageSelectors selectors               = state_selectors(sequence);
        dflash_host_ingress->state_source_slots[row]      = selectors.source;
        dflash_host_ingress->state_destination_slots[row] = selectors.destination;
        // Context append writes only the draft caches. Target execution owns Main KV
        // coverage, including any uncommitted suffix that survives until the round is settled.
        // DFlash2 has only fixed cyclic state; DFlash also grows its Full backend KV here.
        if (sequence.kv->backend) {
            backend_kv_addresses->ensure_mapped_to_tokens(*sequence.kv->backend, end,
                                                          device.stream);
        }
        minimum_count = std::min(minimum_count, counts[row]);
        maximum_count = std::max(maximum_count, counts[row]);
    }

    qwen3_5::DFlashDecodeState& frame = *io.dflash_decode;
    CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, dflash_host_ingress,
                               sizeof(qwen3_5::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                               device.stream));
    const auto batch                = static_cast<std::int32_t>(lanes.size());
    Tensor active_lane_tensor       = frame.active_lanes.slice(0, 0, batch);
    Tensor state_destination_tensor = frame.state_destination_slots.slice(0, 0, batch);
    Tensor device_starts            = frame.context_frontiers.slice(0, 0, batch);
    Tensor device_ends              = frame.execution_frontiers.slice(0, 0, batch);
    Tensor table_rows               = frame.dflash_kv_table_rows.slice(0, 0, batch);
    Tensor positions                = frame.append_positions.slice(1, 0, batch);
    Tensor device_counts            = frame.append_counts.slice(0, 0, batch);

    work.reset();
    Tensor features =
        work.alloc(DType::BF16, {dimension(parameters.draft->feature_projection.weight.k),
                                 static_cast<std::int32_t>(draft_window + 1U), batch});
    ops::prepare_ragged_prefix(dflash->pending_features, active_lane_tensor, device_starts,
                               device_ends, features, positions, device_counts, device.stream);

    execution::DFlashAppendContext state{{device, parameters, work, state_images->linear(),
                                          replay_records ? &*replay_records : nullptr, io,
                                          prefill_hidden, prefill_chunk, proposal_head},
                                         *dflash};
    mark_workspace_usage(workspace_plan.dflash_context);
    execution::dflash_append_context(state, features, positions, device_counts,
                                     state_destination_tensor, table_rows,
                                     {minimum_count, maximum_count});
}

void ProgramImpl::validate_licensed_tokens(std::span<const TokenId> tokens) const {
    for (const TokenId token : tokens) {
        if (token < 0 || token >= dimension(parameters.model.resources().public_token_count)) {
            throw std::runtime_error("target returned a token outside the public token domain");
        }
    }
}

runtime::BatchedGeneratedRound
ProgramImpl::decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                                   std::span<const runtime::RoundBudget> budgets,
                                   runtime::ExecutionTiming* failed_timing) {
    nvtx::ScopedRange round_range(nvtx::Name::DecodeOrdinaryRound, nvtx::Category::Decode,
                                  static_cast<std::uint64_t>(lanes.size()));
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (speculative_backend != SpeculativeBackend::None) {
        throw std::logic_error("ordinary batch execution requires the ordinary backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("ordinary batch membership is invalid");
    }

    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("ordinary batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = active_sequence(lane);
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier) {
            throw std::logic_error("ordinary batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    const auto start = Clock::now();
    try {
        std::optional<nvtx::ScopedRange> submit_range;
        submit_range.emplace(nvtx::Name::DecodeOrdinarySubmit, nvtx::Category::Decode,
                             static_cast<std::uint64_t>(lanes.size()));
        DecodeGraphExecutable* executable = nullptr;
        ops::CausalAttentionExecutionEnvelope envelope{maximum_frontier + 1, maximum_frontier + 1};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(ordinary_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "ordinary batch");
            executable = &install_graph_profile(ordinary_graphs, profile, "ordinary batch");
            envelope   = {profile.min_execution_frontier + 1, profile.max_execution_frontier + 1};
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence            = active_sequence(lanes[row]);
            const RequestControl& request      = requests[lanes[row]];
            const std::uint32_t frontier       = sequence.execution_frontier;
            ordinary_host_ingress->tokens[row] = sequence.ledger.back();
            ordinary_host_ingress->cache_positions[row] =
                checked_i32(frontier, "ordinary batch position");
            ordinary_host_ingress->rope_positions[row] =
                checked_i32(frontier, "ordinary batch RoPE position") + sequence.rope_delta;
            ordinary_host_ingress->text_kv_table_rows[row] =
                text_kv_addresses->bound_row(sequence.kv->text);
            const StateImageSelectors selectors                 = state_selectors(sequence);
            ordinary_host_ingress->state_source_slots[row]      = selectors.source;
            ordinary_host_ingress->state_destination_slots[row] = selectors.destination;
            if (request.output_constraint) {
                upload_constraint_mask(sequence.lane, 0, request.output_constraint->next_mask());
            }
            ordinary_host_ingress->sampling[row]                = request.sampling_host;
            ensure_sequence_kv_mapped(sequence, frontier + 1, 0);
        }

        execution::OrdinaryBatchContext schedule_state{
            {device, parameters, work, state_images->linear(),
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head},
            decoder->text_kv,
            *io.ordinary,
            *ordinary_host_ingress,
            *ordinary_host_egress,
            state_images->continuation_hidden_store()};

        mark_workspace_usage(workspace_plan.ordinary_round);
        execution::ordinary_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                         envelope, executable);
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            RequestControl& request = requests[lanes[row]];
            if (!request.logprobs_device_readout) { continue; }
            const auto column = static_cast<std::int32_t>(row);
            enqueue_round_logprobs(active_sequence(lanes[row]), request,
                                   io.ordinary->logits.slice(1, column, 1),
                                   io.ordinary->sampled_tokens.slice(0, column, 1), 1);
        }
        submit_range.reset();
        timing.begin_wait();
        {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeOrdinaryWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        }
        timing.end_wait();

        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence    = active_sequence(lanes[row]);
            RequestControl& request    = requests[lanes[row]];
            const std::uint32_t base_E = sequence.execution_frontier;
            const std::uint32_t base_S = sequence.ledger_frontier;
            const TokenId token        = ordinary_host_egress->sampled_tokens[row];
            validate_licensed_tokens(std::span<const TokenId>(&token, 1));
            if (request.logprobs_device_readout) {
                collect_round_logprobs(sequence, request, std::span<const TokenId>(&token, 1));
            } else if (request.logprobs.enabled) {
                const Tensor column =
                    io.ordinary->logits.slice(1, static_cast<std::int32_t>(row), 1);
                record_round_logprobs(request, &column, token);
            }
            sequence.text_kv_valid = base_E + 1;
            commit_sequence_kv(sequence, sequence.text_kv_valid, 0);
            sequence.tail_hidden_valid = true;
            sequence.ledger.push_back(token);
            sequence.prefix_identity.append_generated(1, sequence.rope_delta);
            sequence.prefix_digests.append_generated(std::span<const TokenId>(&token, 1),
                                                     sequence.rope_delta);
            request.pending   = PendingCandidate{.kind          = PendingKind::Ordinary,
                                                 .base_E        = base_E,
                                                 .base_S        = base_S,
                                                 .prompt_tokens = 0,
                                                 .produced      = 1};
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens =
                std::span<const TokenId>(ordinary_host_egress->sampled_tokens.data(), lanes.size()),
            .timing = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeOrdinaryWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        clear_execution_failure_lanes(lanes);
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImpl::decode_mtp_batch(std::span<const std::uint32_t> lanes,
                              std::span<const runtime::RoundBudget> budgets,
                              runtime::ExecutionTiming* failed_timing) {
    nvtx::ScopedRange round_range(nvtx::Name::DecodeMtpRound, nvtx::Category::Mtp,
                                  static_cast<std::uint64_t>(lanes.size()));
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (speculative_backend != SpeculativeBackend::Mtp || !io.mtp_decode ||
        decoder->mtp_cache() == nullptr) {
        throw std::logic_error("MTP batch execution requires the MTP backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("MTP batch membership is invalid");
    }

    const std::uint32_t width      = draft_window + 1;
    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("MTP batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = active_sequence(lane);
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv || !sequence.kv->backend ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            backend_kv_addresses->bound_row(*sequence.kv->backend) < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.mtp_kv_valid != sequence.execution_frontier ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier ||
            sequence.mtp_draft_count > draft_window) {
            throw std::logic_error("MTP batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    const auto started = Clock::now();
    try {
        std::optional<nvtx::ScopedRange> submit_range;
        submit_range.emplace(nvtx::Name::DecodeMtpSubmit, nvtx::Category::Mtp,
                             static_cast<std::uint64_t>(lanes.size()));
        DecodeGraphExecutable* executable = nullptr;
        execution::MtpCausalAttentionEnvelopes envelopes =
            mtp_causal_attention_envelopes(maximum_frontier, draft_window, capacity);
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(mtp_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "MTP batch");
            executable = &install_graph_profile(mtp_graphs, profile, "MTP batch");
            envelopes = mtp_causal_attention_envelopes(profile.max_execution_frontier, draft_window,
                                                       capacity);
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = active_sequence(lanes[row]);
            const RequestControl& request     = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1
                                                    : 0;
            // A host-readout logprob request reads one full column per round, so it never drafts.
            std::uint32_t extent =
                request.logprobs.enabled && !request.logprobs_device_readout
                    ? 0U
                    : std::min({sequence.mtp_draft_count, draft_window, max_by_budget,
                                capacity - sequence.execution_frontier - 1});
            if (request.output_constraint) {
                // Column 0 is the committed grammar state. Column j+1 gets the state after
                // drafts[0..j]; a draft stops at its first token the grammar rejects or that
                // completes it, so every verified draft is admissible.
                upload_constraint_mask(sequence.lane, 0, request.output_constraint->next_mask());
                if (extent > 0) {
                    runtime::OutputConstraintState preview = request.output_constraint->fork();
                    std::uint32_t admissible               = 0;
                    while (admissible < extent &&
                           preview.try_accept(sequence.mtp_drafts[admissible]) &&
                           !preview.terminated()) {
                        ++admissible;
                        upload_constraint_mask(sequence.lane, admissible, preview.next_mask());
                    }
                    extent = admissible;
                }
            }
            mtp_host_ingress->anchors[row]        = sequence.ledger.back();
            mtp_host_ingress->base_frontiers[row] = checked_i32(frontier, "MTP batch frontier");
            mtp_host_ingress->remaining_budgets[row] =
                checked_i32(budgets[row].generated_tokens_remaining, "MTP batch remaining budget");
            mtp_host_ingress->current_extents[row]      = static_cast<std::int32_t>(extent);
            mtp_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1);
            for (std::uint32_t j = 0; j < draft_window; ++j) {
                mtp_host_ingress->current_drafts[row * draft_window + j] =
                    j < extent ? sequence.mtp_drafts[j] : sequence.ledger.back();
            }
            for (std::uint32_t j = 0; j < width; ++j) {
                const std::uint32_t position = frontier + std::min(j, extent);
                mtp_host_ingress->target_rope_positions[row * width + j] =
                    checked_i32(position, "MTP batch RoPE position") + sequence.rope_delta;
            }
            mtp_host_ingress->text_kv_table_rows[row] =
                text_kv_addresses->bound_row(sequence.kv->text);
            mtp_host_ingress->mtp_kv_table_rows[row] =
                backend_kv_addresses->bound_row(*sequence.kv->backend);
            const StateImageSelectors selectors            = state_selectors(sequence);
            mtp_host_ingress->state_source_slots[row]      = selectors.source;
            mtp_host_ingress->state_destination_slots[row] = selectors.destination;
            mtp_host_ingress->rope_deltas[row]             = sequence.rope_delta;
            mtp_host_ingress->sampling[row]                = request.sampling_host;
            ensure_sequence_kv_mapped(sequence, frontier + extent + 1,
                                      std::min(capacity, frontier + extent + draft_window));
        }

        execution::MtpBatchContext schedule_state{{device, parameters, work, state_images->linear(),
                                                   replay_records ? &*replay_records : nullptr, io,
                                                   prefill_hidden, prefill_chunk, proposal_head},
                                                  decoder->text_kv,
                                                  *decoder->mtp_cache(),
                                                  *io.mtp_decode,
                                                  *mtp_host_ingress,
                                                  *mtp_host_egress,
                                                  state_images->continuation_hidden_store()};

        mark_workspace_usage(workspace_plan.mtp_round);
        execution::mtp_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                    draft_window, envelopes, executable);
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            RequestControl& request = requests[lanes[row]];
            if (!request.logprobs_device_readout) { continue; }
            const auto column  = static_cast<std::int32_t>(row);
            const auto columns = mtp_host_ingress->target_valid_columns[row];
            enqueue_round_logprobs(
                active_sequence(lanes[row]), request,
                io.mtp_decode->target_logits.slice(2, column, 1).slice(1, 0, columns),
                io.mtp_decode->licensed_tokens.slice(1, column, 1).slice(0, 0, columns),
                static_cast<std::uint32_t>(columns));
        }
        submit_range.reset();
        timing.begin_wait();
        {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeMtpWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        }
        timing.end_wait();

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = active_sequence(lanes[row]);
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = mtp_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = mtp_host_egress->accepted_drafts[row];
            const std::int32_t next_i     = mtp_host_egress->next_extents[row];
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || next_i < 0 ||
                next_i > static_cast<std::int32_t>(draft_window) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("MTP batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(mtp_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            if (request.logprobs_device_readout) {
                collect_round_logprobs(sequence, request, row_tokens);
            } else if (request.logprobs.enabled) {
                if (count_i != 1) {
                    throw std::logic_error("token logprobs round licensed a draft");
                }
                const Tensor column =
                    io.mtp_decode->target_logits.slice(2, static_cast<std::int32_t>(row), 1)
                        .slice(1, 0, 1);
                record_round_logprobs(request, &column, row_tokens[0]);
            }
            const std::uint32_t pcur =
                static_cast<std::uint32_t>(mtp_host_ingress->current_extents[row]);
            if (pcur == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += pcur;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            request.pending = PendingCandidate{
                .kind          = PendingKind::Speculative,
                .base_E        = base_E,
                .base_S        = base_S,
                .prompt_tokens = 0,
                .produced      = static_cast<std::uint32_t>(count_i),
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(mtp_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(mtp_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width,
            .timing     = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeMtpWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        clear_execution_failure_lanes(lanes);
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImpl::decode_dflash_batch(std::span<const std::uint32_t> lanes,
                                 std::span<const runtime::RoundBudget> budgets,
                                 runtime::ExecutionTiming* failed_timing) {
    nvtx::ScopedRange round_range(nvtx::Name::DecodeDFlashRound, nvtx::Category::DFlash,
                                  static_cast<std::uint64_t>(lanes.size()));
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (!is_masked_draft_backend(speculative_backend) || !io.dflash_decode || !dflash) {
        throw std::logic_error("DFlash batch execution requires the DFlash backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("DFlash batch membership is invalid");
    }

    const std::uint32_t width           = draft_window + 1U;
    std::uint32_t maximum_frontier      = 0;
    std::uint32_t maximum_target_tokens = 1;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("DFlash batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = active_sequence(lane);
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            text_kv_addresses->bound_row(sequence.kv->text) < 0 ||
            (backend_kv_cache() && (!sequence.kv->backend ||
                                    backend_kv_addresses->bound_row(*sequence.kv->backend) < 0)) ||
            sequence.execution_frontier >= capacity ||
            sequence.text_kv_valid != sequence.execution_frontier ||
            sequence.dflash_context_frontier > sequence.execution_frontier ||
            sequence.execution_frontier - sequence.dflash_context_frontier > width ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.prefix_digests.size() != sequence.ledger_frontier) {
            throw std::logic_error("DFlash batch row is not decode-ready");
        }
        const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                ? budgets[row].generated_tokens_remaining - 1U
                                                : 0U;
        // DFlash proposes on the device inside the round, so its drafts cannot be walked
        // through the grammar ahead of verification; constrained rows verify no drafts.
        const std::uint32_t extent = requests[lanes[row]].output_constraint
                                         ? 0U
                                         : std::min({draft_window, max_by_budget,
                                                     capacity - sequence.execution_frontier - 1U});
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
        maximum_target_tokens =
            std::max(maximum_target_tokens, sequence.execution_frontier + extent + 1U);
    }

    const auto started = Clock::now();
    try {
        std::optional<nvtx::ScopedRange> submit_range;
        submit_range.emplace(nvtx::Name::DecodeDFlashSubmit, nvtx::Category::DFlash,
                             static_cast<std::uint64_t>(lanes.size()));
        DecodeGraphExecutable* executable    = nullptr;
        execution::DFlashEnvelopes envelopes = dflash_envelopes(0, maximum_frontier, draft_window);
        ops::CausalAttentionExecutionEnvelope target_envelope{1, maximum_target_tokens};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(dflash_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "DFlash batch");
            executable      = &install_graph_profile(dflash_graphs, profile, "DFlash batch");
            envelopes       = dflash_envelopes(profile.min_execution_frontier,
                                               profile.max_execution_frontier, draft_window);
            target_envelope = {
                1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                       capacity, static_cast<std::uint64_t>(profile.max_execution_frontier) +
                                     draft_window + 1ULL))};
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence           = active_sequence(lanes[row]);
            const RequestControl& request     = requests[lanes[row]];
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1U
                                                    : 0U;
            const std::uint32_t extent =
                request.output_constraint ||
                        (request.logprobs.enabled && !request.logprobs_device_readout)
                    ? 0U
                    : std::min({draft_window, max_by_budget, capacity - frontier - 1U});
            dflash_host_ingress->anchors[row] = sequence.ledger.back();
            dflash_host_ingress->execution_frontiers[row] =
                checked_i32(frontier, "DFlash batch frontier");
            dflash_host_ingress->context_frontiers[row] =
                checked_i32(sequence.dflash_context_frontier, "DFlash context frontier");
            dflash_host_ingress->proposal_valid_columns[row] = static_cast<std::int32_t>(width);
            dflash_host_ingress->proposal_extents[row]       = static_cast<std::int32_t>(extent);
            dflash_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1U);
            for (std::uint32_t column = 0; column < width; ++column) {
                const std::uint32_t position = frontier + std::min(column, extent);
                dflash_host_ingress->target_rope_positions[row * width + column] =
                    checked_i32(position, "DFlash target RoPE position") + sequence.rope_delta;
            }
            dflash_host_ingress->text_kv_table_rows[row] =
                text_kv_addresses->bound_row(sequence.kv->text);
            dflash_host_ingress->dflash_kv_table_rows[row] =
                sequence.kv->backend ? backend_kv_addresses->bound_row(*sequence.kv->backend) : 0;
            dflash_host_ingress->active_lanes[row]       = static_cast<std::int32_t>(sequence.lane);
            const StateImageSelectors selectors          = state_selectors(sequence);
            dflash_host_ingress->state_source_slots[row] = selectors.source;
            dflash_host_ingress->state_destination_slots[row] = selectors.destination;
            if (request.output_constraint) {
                upload_constraint_mask(sequence.lane, 0, request.output_constraint->next_mask());
            }
            dflash_host_ingress->sampling[row]                = request.sampling_host;
            ensure_sequence_kv_mapped(sequence, frontier + extent + 1U,
                                      backend_kv_cache() ? frontier : 0U);
        }

        execution::DFlashBatchContext schedule_state{
            {device, parameters, work, state_images->linear(),
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head},
            decoder->text_kv,
            *dflash,
            *io.dflash_decode,
            *dflash_host_ingress,
            *dflash_host_egress,
            state_images->continuation_hidden_store()};

        mark_workspace_usage(workspace_plan.dflash_round);
        execution::dflash_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                       draft_window, envelopes, target_envelope, executable);
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            RequestControl& request = requests[lanes[row]];
            if (!request.logprobs_device_readout) { continue; }
            const auto column  = static_cast<std::int32_t>(row);
            const auto columns = dflash_host_ingress->target_valid_columns[row];
            enqueue_round_logprobs(
                active_sequence(lanes[row]), request,
                io.dflash_decode->target_logits.slice(2, column, 1).slice(1, 0, columns),
                io.dflash_decode->licensed_tokens.slice(1, column, 1).slice(0, 0, columns),
                static_cast<std::uint32_t>(columns));
        }
        submit_range.reset();
        timing.begin_wait();
        {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeDFlashWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        }
        timing.end_wait();

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence       = active_sequence(lanes[row]);
            RequestControl& request       = requests[lanes[row]];
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
            const std::int32_t count_i    = dflash_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = dflash_host_egress->accepted_drafts[row];
            const std::uint32_t extent =
                static_cast<std::uint32_t>(dflash_host_ingress->proposal_extents[row]);
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || accepted_i > static_cast<std::int32_t>(extent) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("DFlash batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(dflash_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            if (request.logprobs_device_readout) {
                collect_round_logprobs(sequence, request, row_tokens);
            } else if (request.logprobs.enabled) {
                if (count_i != 1) {
                    throw std::logic_error("token logprobs round licensed a draft");
                }
                const Tensor column =
                    io.dflash_decode->target_logits.slice(2, static_cast<std::int32_t>(row), 1)
                        .slice(1, 0, 1);
                record_round_logprobs(request, &column, row_tokens[0]);
            }
            if (extent == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += extent;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
            }
            sequence.dflash_context_frontier = base_E;
            request.pending                  = PendingCandidate{
                                 .kind          = PendingKind::Speculative,
                                 .base_E        = base_E,
                                 .base_S        = base_S,
                                 .prompt_tokens = 0,
                                 .produced      = static_cast<std::uint32_t>(count_i),
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(dflash_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(dflash_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width,
            .timing     = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeDFlashWait, nvtx::Category::Control,
                                         static_cast<std::uint64_t>(lanes.size()));
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        clear_execution_failure_lanes(lanes);
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImpl::decode_raw(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets,
                        runtime::ExecutionTiming* failed_timing) {
    if (speculative_backend == SpeculativeBackend::None) {
        return decode_ordinary_batch(lanes, budgets, failed_timing);
    }
    if (speculative_backend == SpeculativeBackend::Mtp) {
        return decode_mtp_batch(lanes, budgets, failed_timing);
    }
    return decode_dflash_batch(lanes, budgets, failed_timing);
}

runtime::ExecutionTiming ProgramImpl::resolve_non_speculative_pending(
    SequenceState& sequence, RequestControl& request, std::uint32_t accepted_tokens, bool terminal,
    std::optional<std::uint32_t> prefix_execution_split_after,
    runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed_timing);
    if (request.lifecycle != Lifecycle::Pending) {
        throw std::logic_error("pending resolution requires a pending generated round");
    }
    if ((request.pending.kind != PendingKind::Begin &&
         request.pending.kind != PendingKind::Ordinary) ||
        request.pending.produced != 1 || accepted_tokens != 1) {
        throw std::logic_error("non-speculative pending round must commit its single token");
    }

    const std::uint32_t base_ledger_frontier = request.pending.kind == PendingKind::Begin
                                                   ? request.pending.prompt_tokens
                                                   : request.pending.base_S;
    commit_generated_prefix_identity(
        sequence, base_ledger_frontier,
        std::span<const TokenId>(sequence.ledger).subspan(base_ledger_frontier, accepted_tokens),
        prefix_execution_split_after);

    switch (request.pending.kind) {
    case PendingKind::Begin:
        sequence.execution_frontier = request.pending.prompt_tokens;
        sequence.ledger_frontier    = request.pending.prompt_tokens + 1;
        break;
    case PendingKind::Ordinary:
        advance_rebuild_work(sequence, request.pending.base_E + request.pending.produced,
                             prefill_chunk);
        sequence.execution_frontier = request.pending.base_E + request.pending.produced;
        sequence.ledger_frontier    = request.pending.base_S + request.pending.produced;
        break;
    case PendingKind::Speculative:
    case PendingKind::None:
        throw std::logic_error("non-speculative pending round has an invalid kind");
    }
    if (sequence.ledger_frontier != sequence.execution_frontier + 1 ||
        sequence.ledger.size() != sequence.ledger_frontier ||
        sequence.prefix_identity.size() != sequence.ledger_frontier ||
        sequence.prefix_digests.size() != sequence.ledger_frontier) {
        throw std::logic_error("resolved round did not establish a valid frontier");
    }
    // Begin publishes a sampled token but does not execute it through the target. An exact-hit
    // Fork therefore still names an immutable read source and an unwritten destination here; the
    // first state-mutating decode commit closes it. A suffix prefill already closed its Fork at
    // the committed prefill frontier.
    if (request.pending.kind == PendingKind::Begin && terminal && sequence.state.fork_pending) {
        const StateImageSelectors selectors = state_selectors(sequence);
        timing.resume_submit();
        state_images->copy_slot(selectors.source, selectors.destination, device.stream);
        timing.begin_wait();
        device.synchronize();
        timing.end_wait();
        settle_state_fork(sequence);
    } else if (request.pending.kind == PendingKind::Ordinary) {
        settle_state_fork(sequence);
    }
    trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
    if (terminal) { sequence.mtp_draft_count = 0; }
    if (request.output_constraint) {
        request.output_constraint->accept(
            std::span<const TokenId>(sequence.ledger)
                .subspan(sequence.ledger.size() - accepted_tokens, accepted_tokens));
    }
    request.lifecycle = terminal ? Lifecycle::Finishable : Lifecycle::Active;
    request.pending   = {};
    return timing.finish();
}


} // namespace ninfer::models::qwen3_5::detail
