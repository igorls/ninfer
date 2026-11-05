#include "models/qwen4_exp/program/internal.h"
#include "core/transfer.h"
#include "ninfer/ops/target_logprobs.h"
#include "ninfer/ops/argmax.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

void RankProgramImpl::score_next() {
    auto& f          = *compute;
    const auto& work = std::get<ScoreWork>(f.request.work.operation);
    if (f.score_at == work.targets.size()) {
        if (!compute_done.ready()) return;
        ScoreResult result;
        result.processed = work.row.ids.size();
        result.scores    = std::move(f.scores);
        if (f.timer_started)
            f.gpu.add_prefill(
                static_cast<std::uint64_t>(double(prefill_timer.elapsed_ms()) * 1.0e6 + 0.5));
        if (auto ns = collect_vision_timing(binding(work.row.member.residency),
                                            work.row.main_frontier + result.processed))
            f.gpu.add_vision(*ns);
        gpu_timer.observe_result(f.gpu);
        events.emplace_back(
            runtime::ResultReady<ExecutionTypes>{f.request.ticket, Result(std::move(result))});
        f.phase = ComputePhase::PendingCommit;
        if (resources.compute_actions->occupied()) resources.compute_actions->finish();
        return;
    }
    const auto n = std::min<std::size_t>(1024, work.targets.size() - f.score_at);
    const auto slots = work.readout.reference_count + 1;
    auto scope   = resources.workspace.scope();
    auto buffers = execution::workspace::score(resources.workspace,
                                               parameters.profile->local.vocabulary, n, slots);
    auto* host   = static_cast<TokenId*>(resources.compute_aux().data);
    for (std::size_t column = 0; column < n; ++column) {
        const auto index = f.score_at + column;
        host[column * slots] = work.targets.at(index);
        std::copy_n(work.readout.reference_tokens.begin() + index * (slots - 1), slots - 1,
            host + column * slots + 1);
    }
    copy_bytes_async({buffers.targets.data, buffers.targets.bytes(), cuda.device()},
                     {host, buffers.targets.bytes(), Device::cpu(), true}, cuda);
    auto hidden = f.sample.slice(1, work.first_target_column + f.score_at, n);
    execution::project_head(hidden, parameters.text.head, buffers.logits, resources.workspace, cuda);
    if (parallel) {
        if (slots != 1 || work.readout.capture_hidden_rows)
            throw std::invalid_argument("distributed causal readouts support target scores only");
        auto stats    = resources.workspace.alloc(DType::FP32, {3, static_cast<int>(n)});
        auto gathered = resources.workspace.alloc(
            DType::FP32, {3, static_cast<int>(n), dimension(parallel->size())});
        ops::vocabulary_logprob_stats(
            buffers.logits, buffers.targets, parameters.profile->local.vocabulary_begin,
            parameters.profile->public_tokens, stats, resources.workspace, cuda);
        parallel->gather(stats, gathered, cuda);
        ops::target_logprobs_from_stats(gathered, buffers.logprobs, cuda);
    } else
        ops::target_logprobs(buffers.logits, buffers.targets, parameters.profile->public_tokens,
                             buffers.logprobs, cuda);
    copy_bytes_async(resources.egress().subspan(0, buffers.logprobs.bytes()),
                     {buffers.logprobs.data, buffers.logprobs.bytes(), cuda.device()}, cuda);
    if (work.readout.reference_count) {
        ops::argmax(buffers.logits, buffers.argmax, parameters.profile->public_tokens, cuda);
        ops::target_logprobs(buffers.logits, buffers.argmax, parameters.profile->public_tokens,
            buffers.argmax_logprobs, cuda);
        auto offset = buffers.logprobs.bytes();
        copy_bytes_async(resources.egress().subspan(offset, buffers.argmax.bytes()),
            {buffers.argmax.data, buffers.argmax.bytes(), cuda.device()}, cuda);
        offset += buffers.argmax.bytes();
        copy_bytes_async(resources.egress().subspan(offset, buffers.argmax_logprobs.bytes()),
            {buffers.argmax_logprobs.data, buffers.argmax_logprobs.bytes(), cuda.device()}, cuda);
    }
    if (work.readout.capture_hidden_rows) {
        f.scores.hidden_size = static_cast<std::uint32_t>(hidden.ne[0]);
        copy_bytes_async(resources.score_hidden().subspan(0, hidden.bytes()),
            {hidden.data, hidden.bytes(), cuda.device()}, cuda);
    }
    f.score_count = n;
    gpu_timer.record_result();
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::AwaitScore;
}

} // namespace ninfer::models::qwen4_exp::detail
