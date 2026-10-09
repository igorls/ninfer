#include "models/qwen4_exp/program/program_impl.h"
#include "models/qwen4_exp/program/context_work.h"
#include "models/qwen4_exp/program/context.h"
#include "core/device.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/scatter.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen4_exp::execution {

PrefillChunkResult prefill_text_chunk(PrefillContext& state, std::span<const TokenId> ids,
                                      std::uint32_t nominal_length,
                                      std::optional<std::uint32_t> split_frontier,
                                      bool finalize_at_end) {
    TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                     state.execution.ple, state.text_kv, state.execution.state, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, &state.text_cache);
    card.set_sampling(state.sampling);
    card.set_rope_delta(state.rope_delta);
    card.set_mtp_cache(state.mtp_kv);
    card.set_state_slots(state.state_source_slot, state.state_destination_slot);
    card.set_prefill_gpu_timer(state.prefill_gpu_timer);
    card.set_rewrite_checkpoint_hidden_output(state.rewrite_checkpoint_hidden);
    card.set_prefill_split_frontier(split_frontier ? static_cast<std::int64_t>(*split_frontier)
                                                   : -1);
    card.set_first_token_logit_capture(state.first_token_logits_host);
    card.set_first_token_readout(state.first_token_readout);
    card.set_prompt_readout(state.prompt_readout);
    const std::span<const int> prompt(ids.data(), ids.size());
    return card.prefill_chunk(prompt, state.text_kv_base, nominal_length, finalize_at_end,
                              state.prompt, state.vision);
}

void sample_from_hidden(PrefillContext& state, const Tensor& streams,
                        std::int32_t absolute_position, std::int32_t purpose) {
    const auto& config = state.execution.parameters.model.config().text;
    if (streams.dtype != DType::BF16 || streams.ne[0] != dimension(config.stream_width()) ||
        streams.ne[1] != 1 || streams.ne[2] != 1 || streams.ne[3] != 1 || streams.data == nullptr) {
        throw std::invalid_argument("sample_from_hidden requires BF16 [stream_width,1]");
    }
    auto& io = state.execution.io;
    state.execution.work.reset();
    TextContext card(state.execution.device, state.execution.parameters, state.execution.work,
                     state.execution.ple, state.text_kv, state.execution.state, io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, &state.text_cache);
    Tensor logits = io.logits.slice(1, 0, 1);
    card.project_streams(streams, logits);
    cudaStream_t stream = state.execution.device.stream;
    CUDA_CHECK(cudaMemcpyAsync(io.pos.data, &absolute_position, sizeof(absolute_position),
                               cudaMemcpyHostToDevice, stream));
    const std::int32_t domain =
        dimension(state.execution.parameters.model.resources().public_token_count);
    ops::sample(logits, io.token, domain, state.sampling, io.pos, purpose, state.execution.work,
                stream);
    // A zero-suffix reuse samples its first token here instead of in a prefill chunk.
    if (purpose == ops::kSamplePurposePrefill) {
        if (state.first_token_logits_host != nullptr) {
            CUDA_CHECK(cudaMemcpyAsync(state.first_token_logits_host, logits.data,
                                       static_cast<std::size_t>(domain) * sizeof(std::uint16_t),
                                       cudaMemcpyDeviceToHost, stream));
        }
        if (state.first_token_readout != nullptr) {
            enqueue_first_token_readout(*state.first_token_readout, logits, io.token, domain,
                                        stream);
        }
    }
    state.execution.work.reset();
}

} // namespace ninfer::models::qwen4_exp::execution

namespace ninfer::models::qwen4_exp::detail {

PrefillProgress ProgramImpl::advance_prefill(SequenceHandle handle,
                                             runtime::ExecutionTiming* timing,
                                             runtime::TokenMaskProvider* masks) {
    if (!valid_sequence(handle)) { throw std::logic_error("prefill has a stale lane"); }
    const auto lane = ContractAccess::lane(handle).value;
    require_unit(lane, ExecutionUnitKind::Prefill);
    auto& state = sequences[lane];
    prepare_capture_boundary(lane);
    const auto permit = *requests[lane].permit;
    ensure_sequence_kv_mapped(state, permit.main_frontier, permit.backend_frontier);

    grammar_dead_positions[0] = 0;
    if (requests[lane].prefill && permit.main_frontier >= requests[lane].prefill->prompt_tokens &&
        masks && masks->constrained(0)) {
        auto& config = static_cast<qwen4_exp::PrefillRoundHost*>(round_host->data())->sampling;
        config       = requests[lane].sampling_host;
        config.mask  = fill_grammar_mask(masks, 0, {});
        requests[lane].sampling_host.mask = config.mask;
        Tensor config_lane = sampling_config.slice(1, static_cast<std::int32_t>(lane), 1);
        CUDA_CHECK(cudaMemcpyAsync(config_lane.data, &config, sizeof(config),
                                   cudaMemcpyHostToDevice, device.stream));
    }
    auto progress = wrap_prefill(lane, advance_prefill_raw(lane, timing));
    if (!progress.complete) { settle_unit(lane); }
    return progress;
}

runtime::PrefillStepResult
ProgramImpl::advance_prefill_raw(std::uint32_t lane, runtime::ExecutionTiming* failed_timing) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    return advance_prefill(active_sequence(lane), requests[lane], failed_timing);
}

runtime::ExecutionTiming ProgramImpl::resolve_prefill_raw(std::uint32_t lane, bool terminal,
                                                          runtime::ExecutionTiming* failed_timing) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    if (requests[lane].pending.kind != PendingKind::Begin) {
        throw std::logic_error("prefill resolution requires a pending prefill token");
    }
    return resolve_non_speculative_pending(active_sequence(lane), requests[lane], 1, terminal,
                                           std::nullopt, failed_timing);
}

runtime::ExecutionTiming ProgramImpl::resolve_pending_raw(
    std::span<const std::uint32_t> lanes, std::span<const std::uint32_t> accepted_tokens,
    std::span<const std::uint8_t> terminal, std::span<const std::uint8_t> discarded,
    std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
    runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed_timing);
    if (lanes.empty() || lanes.size() > max_concurrency || accepted_tokens.size() != lanes.size() ||
        terminal.size() != lanes.size() || discarded.size() != lanes.size() ||
        prefix_execution_splits.size() != lanes.size()) {
        throw std::invalid_argument("pending batch resolution has inconsistent membership");
    }

    if (lanes.size() == 1 && lanes.front() < max_concurrency &&
        requests[lanes.front()].pending.kind == PendingKind::Begin) {
        const std::uint32_t lane = lanes.front();
        if (requests[lane].lifecycle != Lifecycle::Pending) {
            throw std::logic_error("prefill pending token no longer matches Program state");
        }
        if (discarded.front()) {
            if (accepted_tokens.front() != 0 || !terminal.front()) {
                throw std::logic_error("discarded prefill pending decision is invalid");
            }
            clear_lane(active_sequence(lane), requests[lane]);
        } else {
            timing.pause();
            timing.include(resolve_non_speculative_pending(
                active_sequence(lane), requests[lane], accepted_tokens.front(),
                terminal.front() != 0, prefix_execution_splits.front(), failed_timing));
            timing.resume_post();
        }
        return timing.finish();
    }

    if (speculative_backend == SpeculativeBackend::None) {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
                requests[lane].pending.kind != PendingKind::Ordinary) {
                throw std::logic_error("ordinary pending batch no longer matches Program state");
            }
            if (discarded[row]) {
                clear_lane(active_sequence(lane), requests[lane]);
            } else {
                timing.pause();
                timing.include(resolve_non_speculative_pending(
                    active_sequence(lane), requests[lane], accepted_tokens[row], terminal[row] != 0,
                    prefix_execution_splits[row], failed_timing));
                timing.resume_post();
            }
        }
        return timing.finish();
    }

    return resolve_mtp_pending(lanes, accepted_tokens, terminal, discarded, prefix_execution_splits,
                               failed_timing);
}

runtime::PrefillStepResult ProgramImpl::advance_prefill(SequenceState& sequence,
                                                        RequestControl& request,
                                                        runtime::ExecutionTiming* failed_timing) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill) {
        throw std::logic_error("staged prefill step requires an active concurrent request");
    }

    RequestControl::Prefill& staged = *request.prefill;
    if (request.capture_pending) {
        throw std::logic_error("prefill cannot advance while a capture is pending");
    }
    const runtime::BeginSummary summary{.prompt_tokens        = staged.prompt_tokens,
                                        .reused_prompt_tokens = staged.base,
                                        .prefix_reuse_path    = staged.reuse};
    std::uint32_t processed_prompt_tokens = 0;
    const auto started                    = Clock::now();
    try {
        StateImageSelectors selectors = state_selectors(sequence);
        Tensor rewrite_capture_hidden;
        Tensor* rewrite_capture_hidden_ptr = nullptr;
        if (request.next_capture < request.capture_groups.size()) {
            rewrite_capture_hidden = state_images->continuation_hidden_slot(selectors.destination);
            rewrite_capture_hidden_ptr = &rewrite_capture_hidden;
        }
        execution::PrefillContext schedule_state{
            .execution  = {device, parameters, work, *state_images, *ple_gather, io, prefill_hidden,
                           prefill_chunk},
            .text_kv    = text_kv_view(sequence),
            .text_cache = decoder->text_kv,
            .text_kv_base = staged.cursor,
            .sampling     = static_cast<const ops::SamplingConfig*>(
                sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1).data),
            .rewrite_checkpoint_hidden = rewrite_capture_hidden_ptr,
            .state_source_slot         = selectors.source,
            .state_destination_slot    = selectors.destination,
            .first_token_logits_host   = token_logits_capture(request),
            .prompt                    = &staged.prompt,
            .vision                    = staged.vision.get(),
            .mtp_kv                    = mtp_kv_view(sequence),
            .rope_delta                = sequence.rope_delta,
        };
        schedule_state.prefill_gpu_timer           = &prefill_gpu_timer_;
        const execution::FirstTokenReadout readout = first_token_readout(sequence, request);
        if (request.logprobs_device_readout) { schedule_state.first_token_readout = &readout; }
        if (!request.prompt_readout.positions.empty() || request.prompt_readout.feature_position) {
            schedule_state.prompt_readout = &request.prompt_readout;
        }

        if (staged.cursor < staged.prompt_tokens) {
            const std::uint32_t nominal =
                std::min(prefill_chunk, staged.prompt_tokens - staged.cursor);
            mark_workspace_usage(workspace_plan.text_prefill);
            if (staged.vision) { mark_workspace_usage(workspace_plan.vision->capacity_bytes); }
            std::uint32_t remaining = nominal;
            bool finalized          = false;
            while (remaining != 0) {
                schedule_state.text_kv_base           = staged.cursor;
                selectors                             = state_selectors(sequence);
                schedule_state.state_source_slot      = selectors.source;
                schedule_state.state_destination_slot = selectors.destination;
                if (request.next_capture < request.capture_groups.size()) {
                    rewrite_capture_hidden =
                        state_images->continuation_hidden_slot(selectors.destination);
                    schedule_state.rewrite_checkpoint_hidden = &rewrite_capture_hidden;
                } else {
                    schedule_state.rewrite_checkpoint_hidden = nullptr;
                }

                const bool final_candidate = staged.cursor + remaining == staged.prompt_tokens;
                const std::optional<std::uint32_t> capture_frontier =
                    request.next_capture < request.capture_groups.size()
                        ? std::optional<std::uint32_t>(
                              request.capture_groups[request.next_capture].frontier)
                        : std::nullopt;
                std::optional<std::uint32_t> split_frontier = capture_frontier;
                // A publishing request splits at its rewrite execution frontiers so a later turn
                // resumed from its typed rewrite checkpoint shares its GDN decomposition. A
                // read-only request is never a resume source and runs its suffix unsplit.
                if (request.publish_continuation) {
                    const auto rewrite_split = std::upper_bound(
                        staged.prompt.identity.rewrite_execution_frontiers.begin(),
                        staged.prompt.identity.rewrite_execution_frontiers.end(), staged.cursor);
                    if (rewrite_split != staged.prompt.identity.rewrite_execution_frontiers.end() &&
                        (!split_frontier || *rewrite_split < *split_frontier)) {
                        split_frontier = *rewrite_split;
                    }
                }
                // The reasoning feature row ends a sub-block, so it is computed exactly as for a
                // prompt that stops there, whatever suffix the chosen action renders after it.
                if (request.reasoning_feature_position) {
                    const std::uint32_t frontier = *request.reasoning_feature_position + 1U;
                    if (frontier > staged.cursor &&
                        (!split_frontier || frontier < *split_frontier)) {
                        split_frontier = frontier;
                    }
                }
                execution::PrefillChunkResult result;
                timing.pause();
                result = execution::prefill_text_chunk(
                    schedule_state, std::span<const TokenId>(staged.prompt.token_ids), remaining,
                    split_frontier, final_candidate);
                timing.include(result.timing);
                timing.resume_post();
                if (result.processed_tokens == 0 || result.processed_tokens > remaining) {
                    throw std::logic_error("ordinary prefill chunk made invalid progress");
                }
                staged.cursor += result.processed_tokens;
                processed_prompt_tokens += result.processed_tokens;
                remaining -= result.processed_tokens;
                sequence.text_kv_valid = staged.cursor;
                if (speculative_backend == SpeculativeBackend::Mtp) {
                    sequence.mtp_kv_valid = staged.cursor - 1;
                }
                commit_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));

                // Prompt transitions are canonical immediately. If this was the first write after
                // an immutable source, close the Fork before potentially freezing a new rewrite.
                settle_state_fork(sequence);
                // TextContext publishes the full hyper-connection stream tail to the
                // destination StateImage before the fork is settled.
                sequence.tail_hidden_valid = true;
                const bool reached_capture = capture_frontier && staged.cursor == *capture_frontier;
                if (reached_capture) {
                    if (result.finalized) {
                        // The prompt-frontier state becomes publishable only after the generated
                        // Begin token is committed. commit() marks this capture group ready.
                    } else {
                        staged.elapsed_seconds +=
                            std::chrono::duration<double>(Clock::now() - started).count();

                        request.capture_pending = true;
                        return runtime::PrefillStepResult{
                            .summary                 = summary,
                            .processed_prompt_tokens = processed_prompt_tokens,
                            .timing                  = timing.finish(),
                        };
                    }
                }

                finalized = result.finalized;
                if (finalized || remaining == 0) { break; }
            }

            if (!finalized) {
                if (staged.cursor == staged.prompt_tokens) {
                    throw std::logic_error("staged prefill reached the prompt without sampling");
                }
                staged.elapsed_seconds +=
                    std::chrono::duration<double>(Clock::now() - started).count();
                return runtime::PrefillStepResult{
                    .summary                 = summary,
                    .processed_prompt_tokens = processed_prompt_tokens,
                    .timing                  = timing.finish(),
                };
            }
            if (staged.cursor != staged.prompt_tokens) {
                throw std::logic_error("staged prefill sampled before the prompt frontier");
            }
            timing.resume_submit();
            copy_tail(sequence, io.prefill_tail);
        } else {
            mark_workspace_usage(workspace_plan.ordinary_round);
            if (!sequence.tail_hidden_valid) {
                throw std::logic_error("zero-suffix reuse has no target tail hidden");
            }
            execution::sample_from_hidden(schedule_state, sequence.tail_hidden,
                                          checked_i32(staged.prompt_tokens, "sample position"),
                                          ops::kSamplePurposePrefill);
        }

        copy_round_token();
        timing.begin_wait();
        device.synchronize();
        timing.end_wait();
        staged.elapsed_seconds += std::chrono::duration<double>(Clock::now() - started).count();
        const double vision_seconds = staged.retired_vision_seconds +
                                      (staged.vision ? staged.vision->elapsed_seconds() : 0.0);
        const std::uint32_t prompt_tokens = staged.prompt_tokens;

        validate_licensed_tokens(std::span<const TokenId>(host_tokens, 1));
        if (request.logprobs_device_readout) {
            collect_round_logprobs(sequence, request, std::span<const TokenId>(host_tokens, 1));
        } else {
            record_round_logprobs(request, nullptr, host_tokens[0]);
        }
        collect_prompt_readout(request);
        if (sequence.ledger.size() != prompt_tokens) {
            throw std::logic_error("candidate token ledger does not match prompt length");
        }
        sequence.ledger.push_back(host_tokens[0]);
        sequence.prefix_identity.append_generated(1, sequence.rope_delta);
        sequence.prefix_digests.append_generated(std::span<const TokenId>(host_tokens, 1),
                                                 sequence.rope_delta);
        sequence.text_kv_valid     = prompt_tokens;
        sequence.tail_hidden_valid = true;
        request.timings.vision_seconds += vision_seconds;
        request.timings.prefill_seconds = std::max(0.0, staged.elapsed_seconds - vision_seconds);
        if (staged.vision) { staged.vision->retire_handoff(); }
        staged.prompt.release_all_media_payloads();

        const bool prompt_frontier_capture =
            request.next_capture < request.capture_groups.size() &&
            request.capture_groups[request.next_capture].frontier == prompt_tokens;
        if (!prompt_frontier_capture) { request.prefill.reset(); }
        request.pending   = PendingCandidate{.kind          = PendingKind::Begin,
                                             .base_E        = 0,
                                             .base_S        = 0,
                                             .prompt_tokens = prompt_tokens,
                                             .produced      = 1};
        request.lifecycle = Lifecycle::Pending;
        return runtime::PrefillStepResult{
            .summary = summary,
            .round   = runtime::GeneratedRound{.tokens = std::span<const TokenId>(host_tokens, 1)},
            .processed_prompt_tokens = processed_prompt_tokens,
            .complete                = true,
            .timing                  = timing.finish(),
        };
    } catch (...) {
        timing.begin_wait();
        try {
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        const std::uint32_t lane = sequence.lane;
        clear_execution_failure_lanes(std::span<const std::uint32_t>(&lane, 1));
        throw;
    }
}


} // namespace ninfer::models::qwen4_exp::detail
