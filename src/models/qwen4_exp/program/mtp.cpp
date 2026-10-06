#include "models/qwen4_exp/program/program_impl.h"
#include "models/qwen4_exp/program/context_work.h"
#include "models/qwen4_exp/program/context.h"
#include "models/qwen4_exp/program/planning/graph_profiles.h"
#include "core/device.h"
#include "core/nvtx.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/speculative_round.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <optional>
#include <stdexcept>

namespace ninfer::models::qwen4_exp::detail {
namespace {
void copy(const Tensor& destination, const Tensor& source, cudaStream_t stream) {
    if (destination.bytes() != source.bytes()) {
        throw std::logic_error("MTP copy shape mismatch");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(),
                               cudaMemcpyDeviceToDevice, stream));
}

execution::OrdinaryDecodeInputs mtp_verify_inputs(MtpFrame& frame, std::int32_t batch,
                                                  ops::QsaIndexerSelectEnvelope envelope) {
    const auto width   = frame.width;
    const auto columns = width * batch;
    const auto input   = [&](std::size_t offset, std::int32_t count) {
        return frame.i32(offset, count);
    };
    return {.ids                = input(offsetof(MtpIngress, ids), columns),
            .cache_positions    = input(offsetof(MtpIngress, positions), columns),
            .rope_positions     = input(offsetof(MtpIngress, rope), columns * 3).view({columns, 3}),
            .kv_table_rows      = input(offsetof(MtpIngress, rows), columns),
            .state_source_slots = input(offsetof(MtpIngress, sources), batch),
            .state_destination_slots = input(offsetof(MtpIngress, snapshots), batch),
            .ple_codes =
                Tensor(static_cast<std::byte*>(frame.ingress.data) + offsetof(MtpIngress, codes),
                       DType::U8,
                       {static_cast<std::int32_t>(kPleCodeRowBytes),
                        columns * static_cast<std::int32_t>(kPleHeads)}),
            .ple_scales =
                Tensor(static_cast<std::byte*>(frame.ingress.data) + offsetof(MtpIngress, scales),
                       DType::FP16,
                       {static_cast<std::int32_t>(kPleScaleRowWords),
                        columns * static_cast<std::int32_t>(kPleHeads)}),
            .indexer       = envelope,
            .width         = width,
            .sequence_rows = input(offsetof(MtpIngress, sequence_rows), batch),
            .valid_columns = input(offsetof(MtpIngress, valid), batch),
            .replay        = &frame.records};
}

DecodeGraphExecutable& mtp_graph_for(DecodeGraphFamily& family, std::uint32_t batch,
                                     std::uint32_t frontier) {
    for (std::size_t i = 0; i < family.profiles.size(); ++i) {
        auto& profile = family.profiles[i];
        if (profile.batch_size != batch || frontier < profile.min_execution_frontier ||
            frontier > profile.max_execution_frontier) {
            continue;
        }
        for (auto& topology : family.topologies) {
            if (topology.topology_class != profile.topology_class) { continue; }
            if (topology.installed_profile != i) {
                topology.executable.update(profile.definition);
                topology.installed_profile = i;
            }
            return topology.executable;
        }
    }
    throw std::logic_error("MTP CUDA Graph coverage is incomplete");
}

} // namespace

void ProgramImpl::mtp_proposal_body(std::int32_t batch, ops::QsaIndexerSelectEnvelope envelope) {
    auto frame       = mtp_frame->batch(batch);
    const auto input = [&](std::size_t offset, std::int32_t count) {
        return frame.i32(offset, count);
    };
    execution::TextContext card(device, parameters, work, *ple_gather, {}, *state_images, io,
                                prefill_hidden, prefill_chunk, 0, &decoder->text_kv);
    card.set_mtp_cache({}, decoder->mtp_cache());
    work.reset();
    auto embedded = frame.embedding.slice(1, 0, batch);
    ops::embedding(frame.proposal_ids, parameters.mtp->token_embedding, embedded, device.stream);
    execution::OrdinaryDecodeInputs proposal{
        .ids                     = frame.proposal_ids,
        .cache_positions         = frame.proposal_positions,
        .rope_positions          = frame.proposal_rope,
        .kv_table_rows           = input(offsetof(MtpIngress, mtp_sequence_rows), batch),
        .state_source_slots      = frame.proposal_sources,
        .state_destination_slots = input(offsetof(MtpIngress, snapshots), batch),
        .indexer                 = envelope};
    card.mtp_forward(embedded, frame.ar_hidden, frame.proposal_positions, frame.proposal_rope,
                     frame.next_hidden, &frame.proposal_logits, &proposal);
    ops::argmax(frame.proposal_logits, frame.proposal_ids,
                dimension(parameters.model.resources().public_token_count), device.stream);
    copy(frame.ar_hidden, frame.next_hidden, device.stream);
    copy(frame.proposal_sources, proposal.state_destination_slots, device.stream);
}

void ProgramImpl::mtp_verify_body(std::int32_t batch, ops::QsaIndexerSelectEnvelope envelope) {
    auto frame  = mtp_frame->batch(batch);
    auto inputs = mtp_verify_inputs(frame, batch, envelope);
    auto hidden = frame.hidden.view({frame.hidden.ne[0], frame.width * batch});
    auto logits = frame.logits.view({frame.logits.ne[0], frame.width * batch});
    execution::TextContext card(device, parameters, work, *ple_gather, {}, *state_images, io,
                                prefill_hidden, prefill_chunk, 0, &decoder->text_kv);
    card.verify_batch(inputs, hidden, logits);
}

runtime::BatchedGeneratedRound
ProgramImpl::decode_mtp_batch(std::span<const std::uint32_t> lanes,
                              std::span<const runtime::RoundBudget> budgets,
                              runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (!mtp_frame || !mtp_fold || !parameters.mtp || lanes.empty() ||
        lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::logic_error("MTP batch has no prepared frame or invalid membership");
    }
    const auto started    = Clock::now();
    nvtx::ScopedRange round_range(nvtx::Name::DecodeMtpRound, nvtx::Category::Mtp, lanes.size());
    const auto batch      = static_cast<std::int32_t>(lanes.size());
    const auto width      = static_cast<std::int32_t>(draft_window + 1);
    const auto columns    = width * batch;
    auto frame            = mtp_frame->batch(batch);
    auto& host            = *mtp_ingress;
    *mtp_egress           = {};
    host                  = {};
    std::uint32_t maximum = 0;
    for (std::int32_t row = 0; row < batch; ++row) {
        const auto lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + row, lane) != lanes.begin() + row) {
            throw std::invalid_argument("MTP duplicate or invalid lane");
        }
        auto& sequence      = active_sequence(lane);
        auto& request       = requests[lane];
        const auto frontier = sequence.execution_frontier;
        if (request.lifecycle != Lifecycle::Active || !sequence.kv || !sequence.kv->backend ||
            !sequence.tail_hidden_valid || frontier == 0 || frontier >= capacity ||
            sequence.ledger_frontier != frontier + 1 || sequence.ledger.size() != frontier + 1 ||
            sequence.text_kv_valid != frontier || sequence.mtp_kv_valid != frontier - 1 ||
            budgets[row].generated_tokens_remaining == 0) {
            throw std::logic_error("MTP sequence is not at a committed decode frontier");
        }
        const auto extent =
            request.logprobs.enabled && !request.logprobs_device_readout
                ? 0U
                : std::min({draft_window, budgets[row].generated_tokens_remaining - 1,
                            capacity - frontier - 1});
        host.extents[row]           = static_cast<std::int32_t>(extent);
        host.valid[row]             = static_cast<std::int32_t>(extent + 1);
        host.frontiers[row]         = static_cast<std::int32_t>(frontier);
        host.anchors[row]           = sequence.ledger.back();
        host.sequence_rows[row]     = text_kv_addresses->bound_row(sequence.kv->text);
        host.mtp_sequence_rows[row] = backend_kv_addresses->bound_row(*sequence.kv->backend);
        const auto selectors        = state_selectors(sequence);
        host.sources[row]           = selectors.source;
        host.destinations[row]      = selectors.destination;
        host.snapshots[row]         = state_images->slot_count() + row * width;
        host.sampling[row]          = request.sampling_host;
        ensure_sequence_kv_mapped(sequence, frontier + extent + 1, frontier + extent);
        maximum = std::max(maximum, frontier + extent);
        for (std::int32_t step = 0; step < width - 1; ++step) {
            const auto position =
                static_cast<std::int32_t>(frontier) + std::min(step, host.extents[row]) - 1;
            host.proposal_positions[step * batch + row] = position;
            for (std::int32_t axis = 0; axis < 3; ++axis) {
                host.proposal_rope[step * 3 * batch + axis * batch + row] =
                    position + sequence.rope_delta;
            }
        }
        copy(frame.ar_hidden.slice(1, row, 1),
             state_images->continuation_hidden_slot(selectors.source)
                 .view({frame.ar_hidden.ne[0], 1}),
             device.stream);
    }
    ops::QsaIndexerSelectEnvelope envelope{
        indexer_envelope_blocks(maximum, parameters.model.config().text.indexer.compress_ratio)};
    DecodeGraphExecutable* proposal_graph = nullptr;
    DecodeGraphExecutable* verify_graph   = nullptr;
    if (use_cuda_graph) {
        proposal_graph = &mtp_graph_for(mtp_proposal_graphs, batch, maximum);
        verify_graph   = &mtp_graph_for(mtp_verify_graphs, batch, maximum);
    }
    const auto input = [&](std::size_t offset, std::int32_t count) {
        return frame.i32(offset, count);
    };
    CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &host, sizeof(host), cudaMemcpyHostToDevice,
                               device.stream));
    copy(frame.proposal_ids, input(offsetof(MtpIngress, anchors), batch), device.stream);
    copy(frame.proposal_sources, input(offsetof(MtpIngress, sources), batch), device.stream);
    execution::TextContext card(device, parameters, work, *ple_gather, {}, *state_images, io,
                                prefill_hidden, prefill_chunk, 0, &decoder->text_kv);
    card.set_mtp_cache({}, decoder->mtp_cache());
    mark_workspace_usage(workspace_plan.mtp_round);
    // Each recursive proposal writes a private forming-block slot. Canonical MTP state remains
    // untouched until teacher extension has rebuilt every accepted target prefix.
    std::optional<nvtx::ScopedRange> phase_range;
    phase_range.emplace(nvtx::Name::DecodeMtpDraft, nvtx::Category::Mtp,
                        static_cast<std::uint64_t>(width - 1));
    for (std::int32_t step = 0; step < width - 1; ++step) {
        copy(frame.proposal_positions,
             input(offsetof(MtpIngress, proposal_positions), (width - 1) * batch)
                 .slice(0, step * batch, batch),
             device.stream);
        copy(frame.proposal_rope,
             input(offsetof(MtpIngress, proposal_rope), 3 * (width - 1) * batch)
                 .slice(0, step * 3 * batch, 3 * batch)
                 .view({batch, 3}),
             device.stream);
        if (step == 0) {
            for (std::int32_t row = 0; row < batch; ++row) {
                const auto saved = state_images->continuation_positions_slot(host.sources[row]);
                for (std::int32_t axis = 0; axis < 3; ++axis) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        static_cast<std::int32_t*>(frame.proposal_rope.data) + axis * batch + row,
                        static_cast<const std::int32_t*>(saved.data) + axis, sizeof(std::int32_t),
                        cudaMemcpyDeviceToDevice, device.stream));
                }
            }
        }
        if (proposal_graph != nullptr) {
            proposal_graph->launch(device.stream);
        } else {
            mtp_proposal_body(batch, envelope);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::int32_t*>(frame.drafts.data) + step,
                                     (width - 1) * sizeof(std::int32_t), frame.proposal_ids.data,
                                     sizeof(std::int32_t), sizeof(std::int32_t), batch,
                                     cudaMemcpyDeviceToDevice, device.stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(mtp_egress->drafts.data(), frame.drafts.data, frame.drafts.bytes(),
                               cudaMemcpyDeviceToHost, device.stream));
    timing.begin_wait();
    device.synchronize();
    timing.end_wait();
    phase_range.reset();
    timing.resume_submit();
    // PLE's large stored table is host mapped. Gather exactly the rows addressed by the proposed
    // ledger, and preview grammar transitions before verification mutates any canonical state.
    for (std::int32_t row = 0; row < batch; ++row) {
        const auto lane    = lanes[row];
        auto& sequence     = active_sequence(lane);
        auto& request      = requests[lane];
        const auto* drafts = mtp_egress->drafts.data() + row * (width - 1);
        if (request.output_constraint) {
            upload_constraint_mask(lane, 0, request.output_constraint->next_mask());
            auto preview            = request.output_constraint->fork();
            std::int32_t admissible = 0;
            for (; admissible < host.extents[row]; ++admissible) {
                if (!preview.try_accept(drafts[admissible]) || preview.terminated()) { break; }
                upload_constraint_mask(lane, admissible + 1, preview.next_mask());
            }
            host.extents[row] = admissible;
            host.valid[row]   = admissible + 1;
        }
        materialization_ledger_.assign(sequence.ledger.begin(), sequence.ledger.end());
        materialization_ledger_.insert(materialization_ledger_.end(), drafts, drafts + width - 1);
        for (std::int32_t w = 0; w < width; ++w) {
            const auto column      = row * width + w;
            const auto local       = std::min(w, host.extents[row]);
            host.ids[column]       = materialization_ledger_[sequence.execution_frontier + local];
            host.positions[column] = host.frontiers[row] + local;
            host.teacher_positions[column] = host.positions[column] - 1;
            host.rows[column]              = host.sequence_rows[row];
            host.mtp_rows[column]          = host.mtp_sequence_rows[row];
            for (std::int32_t axis = 0; axis < 3; ++axis) {
                host.rope[axis * columns + column] = host.positions[column] + sequence.rope_delta;
            }
            ple_gather->gather_position(
                materialization_ledger_, host.positions[column],
                {reinterpret_cast<std::byte*>(host.codes.data()) +
                     column * kPleHeads * kPleCodeRowBytes,
                 reinterpret_cast<std::byte*>(host.scales.data() +
                                              column * kPleHeads * kPleScaleRowWords)});
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &host, sizeof(host), cudaMemcpyHostToDevice,
                               device.stream));
    phase_range.emplace(nvtx::Name::DecodeMtpTarget, nvtx::Category::Mtp,
                        static_cast<std::uint64_t>(columns));
    auto verify = mtp_verify_inputs(frame, batch, envelope);
    auto logits = frame.logits.view({frame.logits.ne[0], columns});
    if (verify_graph != nullptr) {
        verify_graph->launch(device.stream);
    } else {
        mtp_verify_body(batch, envelope);
    }
    // Teacher MTP keys at position p use target streams at p and token embedding at p+1.
    // The first preceding stream/position comes from the saved continuation, including MRoPE.
    for (std::int32_t row = 0; row < batch; ++row) {
        auto previous = frame.previous.slice(2, row, 1).view({frame.previous.ne[0], width});
        copy(previous.slice(1, 0, 1),
             state_images->continuation_hidden_slot(host.sources[row]).view({previous.ne[0], 1}),
             device.stream);
        copy(previous.slice(1, 1, width - 1),
             frame.hidden.slice(2, row, 1).view({previous.ne[0], width}).slice(1, 0, width - 1),
             device.stream);
        const auto saved = state_images->continuation_positions_slot(host.sources[row]);
        for (std::int32_t axis = 0; axis < 3; ++axis) {
            auto* dst =
                static_cast<std::int32_t*>(frame.teacher_rope.data) + axis * columns + row * width;
            CUDA_CHECK(cudaMemcpyAsync(dst, static_cast<const std::int32_t*>(saved.data) + axis,
                                       sizeof(std::int32_t), cudaMemcpyDeviceToDevice,
                                       device.stream));
            CUDA_CHECK(cudaMemcpyAsync(
                dst + 1,
                static_cast<const std::int32_t*>(verify.rope_positions.data) + axis * columns +
                    row * width,
                (width - 1) * sizeof(std::int32_t), cudaMemcpyDeviceToDevice, device.stream));
        }
    }
    copy(frame.teacher_positions,
         input(offsetof(MtpIngress, teacher_positions), columns).view({width, batch}),
         device.stream);
    work.reset();
    ops::embedding(verify.ids, parameters.mtp->token_embedding, frame.embedding, device.stream);
    auto teacher_inputs          = verify;
    teacher_inputs.kv_table_rows = input(offsetof(MtpIngress, mtp_rows), columns);
    teacher_inputs.sequence_rows = input(offsetof(MtpIngress, mtp_sequence_rows), batch);
    auto previous                = frame.previous.view({frame.previous.ne[0], columns});
    auto teacher_out             = frame.teacher_output.view({frame.teacher_output.ne[0], columns});
    {
        nvtx::ScopedRange teacher_range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(columns));
        card.mtp_forward(frame.embedding, previous, frame.teacher_positions.view({columns}),
                         frame.teacher_rope, teacher_out, nullptr, &teacher_inputs, true);
    }
    auto argmax = frame.argmax.view({columns});
    ops::argmax(logits, argmax, dimension(parameters.model.resources().public_token_count),
                device.stream);
    auto lengths = input(offsetof(MtpIngress, frontiers), batch);
    auto anchors = input(offsetof(MtpIngress, anchors), batch);
    work.reset();
    ops::speculative_accept_greedy_drafts(
        frame.argmax, frame.logits, frame.drafts, input(offsetof(MtpIngress, extents), batch),
        lengths, anchors, frame.licensed, frame.counts, frame.accepted,
        dimension(parameters.model.resources().public_token_count), frame.sampling(), work,
        device.stream);
    for (std::int32_t row = 0; row < batch; ++row) {
        auto& request = requests[lanes[row]];
        if (request.logprobs_device_readout) {
            enqueue_round_logprobs(active_sequence(lanes[row]), request,
                                   frame.logits.slice(2, row, 1), frame.licensed.slice(1, row, 1),
                                   width);
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(mtp_egress->licensed.data(), frame.licensed.data,
                               frame.licensed.bytes(), cudaMemcpyDeviceToHost, device.stream));
    CUDA_CHECK(cudaMemcpyAsync(mtp_egress->counts.data(), frame.counts.data, frame.counts.bytes(),
                               cudaMemcpyDeviceToHost, device.stream));
    CUDA_CHECK(cudaMemcpyAsync(mtp_egress->accepted.data(), frame.accepted.data,
                               frame.accepted.bytes(), cudaMemcpyDeviceToHost, device.stream));
    timing.begin_wait();
    device.synchronize();
    timing.end_wait();
    phase_range.reset();
    for (std::int32_t row = 0; row < batch; ++row) {
        auto& sequence      = active_sequence(lanes[row]);
        auto& request       = requests[lanes[row]];
        const auto count    = mtp_egress->counts[row];
        const auto accepted = mtp_egress->accepted[row];
        if (count < 1 || count > host.valid[row] || accepted + 1 != count || accepted < 0) {
            throw std::runtime_error("MTP acceptance returned invalid metadata");
        }
        const std::span<const TokenId> tokens(mtp_egress->licensed.data() + row * width, count);
        validate_licensed_tokens(tokens);
        if (request.logprobs_device_readout) {
            collect_round_logprobs(sequence, request, tokens);
        } else if (request.logprobs.enabled) {
            auto column = frame.logits.slice(2, row, 1).slice(1, 0, 1);
            record_round_logprobs(request, &column, tokens[0]);
        }
        auto& stats = request.speculative_stats;
        if (host.extents[row] == 0) {
            ++stats.fallback_steps;
        } else {
            ++stats.rounds;
            stats.drafted_tokens += host.extents[row];
            stats.accepted_tokens += accepted;
            for (std::int32_t i = 0; i < accepted; ++i) { ++stats.accepted_per_position[i]; }
        }
        request.pending   = {.kind     = PendingKind::Speculative,
                             .base_E   = sequence.execution_frontier,
                             .base_S   = sequence.ledger_frontier,
                             .produced = static_cast<std::uint32_t>(count)};
        request.lifecycle = Lifecycle::Pending;
        request.timings.decode_seconds +=
            std::chrono::duration<double>(Clock::now() - started).count();
    }
    return {.tokens     = {mtp_egress->licensed.data(), lanes.size() * width},
            .row_counts = {mtp_egress->counts.data(), lanes.size()},
            .row_stride = static_cast<std::uint32_t>(width),
            .timing     = timing.finish()};
}

runtime::ExecutionTiming ProgramImpl::resolve_mtp_pending(
    std::span<const std::uint32_t> lanes, std::span<const std::uint32_t> accepted_tokens,
    std::span<const std::uint8_t> terminal, std::span<const std::uint8_t> cancelled,
    std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
    runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    const auto width = static_cast<std::int32_t>(draft_window + 1);
    const auto batch = static_cast<std::int32_t>(lanes.size());
    auto frame       = mtp_frame->batch(batch);
    std::array<ops::GdnReplayFoldRow, kMaximumConcurrency> rows{};
    for (std::int32_t row = 0; row < batch; ++row) {
        auto& sequence       = active_sequence(lanes[row]);
        const auto& request  = requests[lanes[row]];
        const auto selectors = state_selectors(sequence);
        if (request.lifecycle != Lifecycle::Pending ||
            request.pending.kind != PendingKind::Speculative ||
            sequence.execution_frontier != request.pending.base_E ||
            sequence.ledger.size() != request.pending.base_S ||
            selectors.source != mtp_ingress->sources[row] ||
            selectors.destination != mtp_ingress->destinations[row]) {
            throw std::logic_error("MTP pending state changed before commit");
        }
        const auto count = static_cast<std::int32_t>(accepted_tokens[row]);
        rows[row]        = {selectors.source, selectors.destination, cancelled[row] ? 0 : count};
        if (cancelled[row]) { continue; }
        if (count < 1 || count > mtp_egress->counts[row]) {
            throw std::logic_error("MTP invalid committed prefix");
        }
        state_images->commit_token_mixer(mtp_ingress->snapshots[row] + count - 1,
                                         selectors.destination, device.stream);
        copy(state_images->continuation_hidden_slot(selectors.destination),
             frame.hidden.slice(2, row, 1).slice(1, count - 1, 1).view({frame.hidden.ne[0]}),
             device.stream);
        const auto saved = state_images->continuation_positions_slot(selectors.destination);
        const auto rope  = frame.i32(offsetof(MtpIngress, rope), width * batch * 3);
        for (std::int32_t axis = 0; axis < 3; ++axis) {
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(saved.data) + axis,
                                       static_cast<const std::int32_t*>(rope.data) +
                                           axis * width * batch + row * width + count - 1,
                                       sizeof(std::int32_t), cudaMemcpyDeviceToDevice,
                                       device.stream));
        }
    }
    mtp_fold->execute({rows.data(), lanes.size()}, device.stream);
    timing.begin_wait();
    device.synchronize();
    timing.end_wait();
    for (std::int32_t row = 0; row < batch; ++row) {
        auto& sequence = active_sequence(lanes[row]);
        auto& request  = requests[lanes[row]];
        if (cancelled[row]) {
            if (!clear_lane_strict(sequence, request)) {
                throw std::logic_error("MTP cancelled lane cannot be released");
            }
            continue;
        }
        const auto count   = accepted_tokens[row];
        const auto pending = request.pending;
        const std::span<const TokenId> tokens(mtp_egress->licensed.data() + row * width, count);
        settle_state_fork(sequence);
        sequence.ledger.insert(sequence.ledger.end(), tokens.begin(), tokens.end());
        if (request.output_constraint) { request.output_constraint->accept(tokens); }
        commit_generated_prefix_identity(sequence, pending.base_S, tokens,
                                         prefix_execution_splits[row]);
        advance_rebuild_work(sequence, pending.base_E + count, prefill_chunk);
        sequence.execution_frontier = pending.base_E + count;
        sequence.ledger_frontier    = pending.base_S + count;
        sequence.text_kv_valid      = sequence.execution_frontier;
        sequence.mtp_kv_valid       = sequence.execution_frontier - 1;
        sequence.tail_hidden_valid  = true;
        commit_sequence_kv(sequence, sequence.text_kv_valid, sequence.mtp_kv_valid);
        trim_sequence_kv(sequence, sequence.text_kv_valid, sequence.mtp_kv_valid);
        request.pending   = {};
        request.lifecycle = terminal[row] ? Lifecycle::Finishable : Lifecycle::Active;
    }
    work.reset();
    return timing.finish();
}
} // namespace ninfer::models::qwen4_exp::detail
