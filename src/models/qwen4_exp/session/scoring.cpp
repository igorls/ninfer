#include "models/qwen4_exp/session/internal.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {

std::optional<runtime::ExecutionTicket> Session::start_score(qwen::PreparedPrompt&& prompt,
                                                             std::uint32_t first_target,
                                                             const CausalScoreReadout& readout) {
    auto& p = *impl_;
    if (p.shape.purpose != EnginePurpose::CausalScoring)
        throw std::logic_error("not a causal scoring Session");
    if (p.compute || p.operation || p.scoring || p.release_ticket || !p.releases.empty()) return {};
    runtime::ResolvedExecutionOptions options;
    options.allow_prefix_reuse = false;
    auto base                  = plan_request(std::move(prompt), options);
    if (!first_target || first_target >= base.summary().prompt_tokens)
        throw std::invalid_argument("invalid scoring target range");
    if (base.plan_->prompt.has_media())
        throw std::invalid_argument("causal scoring accepts text tokens only");
    const auto count = base.summary().prompt_tokens - first_target;
    if (readout.reference_count > kMaximumCausalReferenceTokens ||
        readout.reference_tokens.size() != std::size_t(count) * readout.reference_count)
        throw std::invalid_argument("causal reference readout has an invalid shape");
    for (auto token : readout.reference_tokens)
        if (token < 0 || static_cast<std::uint32_t>(token) >= p.definition->resources().public_token_count)
            throw std::invalid_argument("causal reference token is outside the public vocabulary");
    auto scoring          = std::make_unique<detail::Scoring>();
    scoring->first_target = first_target;
    scoring->readout = readout;
    scoring->scores.target_logprobs.reserve(count);
    scoring->scores.reference_logprobs.reserve(std::size_t(count) * readout.reference_count);
    if (readout.reference_count) {
        scoring->scores.argmax_tokens.reserve(count);
        scoring->scores.argmax_logprobs.reserve(count);
    }
    if (readout.capture_hidden_rows) {
        scoring->scores.hidden_size = p.definition->config().text.hidden;
        scoring->scores.hidden_rows.reserve(std::size_t(count) * scoring->scores.hidden_size);
    }
    p.scoring = std::move(scoring);
    try {
        if (!start_binding(base, {0}, {})) {
            p.scoring.reset();
            return {};
        }
        p.scoring->ticket = *p.operation->ticket;
        return p.scoring->ticket;
    } catch (...) {
        p.scoring.reset();
        throw;
    }
}

} // namespace ninfer::models::qwen4_exp

namespace ninfer::models::qwen4_exp::detail {

bool SessionImpl::submit_score() {
    if (!scoring || !scoring->sequence || compute)
        throw std::logic_error("scoring continuation is unavailable");
    auto& s          = sequence(*scoring->sequence);
    const auto begin = s.continuation.metadata.main_frontier;
    const auto total = s.request->summary.prompt_tokens;
    if (begin + 1 >= total) {
        if (scoring->scores.target_logprobs.size() != total - scoring->first_target)
            throw std::logic_error("scoring result omitted targets");
        const auto lane = s.lane;
        clear(s);
        sequences[lane].reset();
        scoring->sequence.reset();
        scoring->finishing = true;
        submit_releases();
        return true;
    }
    const auto end   = window_end(s, begin, total - 1);
    auto reservation = executor->try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return false;
    compute  = std::make_unique<ComputeTransaction>(ComputeTransaction::Kind::Score, directory);
    auto& tx = *compute;
    tx.members.push_back(handle(s));
    try {
        const ExecutionUnit unit{handle(s), ExecutionUnitKind::Prefill, end - begin};
        if (!reserve_units({&unit, 1}))
            throw std::logic_error("scoring working set cannot reserve its execution chain");
        prepare_write(s, end, tx.context);
        ScoreWork work;
        work.row                 = work_row(s, end - begin);
        const auto first         = std::min(end, std::max(begin, scoring->first_target - 1));
        work.first_target_column = first - begin;
        work.targets.assign(s.identity.tokens.begin() + first + 1,
                            s.identity.tokens.begin() + end + 1);
        work.readout.reference_count = scoring->readout.reference_count;
        work.readout.capture_hidden_rows = scoring->readout.capture_hidden_rows;
        const auto offset = work.targets.empty() ? 0 :
            std::size_t(first + 1 - scoring->first_target) * work.readout.reference_count;
        const auto references = scoring->readout.reference_tokens.begin() + offset;
        work.readout.reference_tokens.assign(references,
            references + work.targets.size() * work.readout.reference_count);
        tx.rows.push_back(work.row);
        tx.context.prepare_actions();
        tx.ticket =
            executor->submit(std::move(*reservation), {actions(tx.context), std::move(work)});
        tx.context.mark_submitted();
        return true;
    } catch (const std::exception& error) {
        executor->fail(error.what());
        throw;
    }
}

void SessionImpl::finish_score(ScoreResult result) {
    auto& tx = *compute;
    if (!scoring || tx.kind != ComputeTransaction::Kind::Score || tx.rows.size() != 1 ||
        result.processed != tx.rows[0].ids.size())
        throw std::logic_error("invalid Qwen4Exp score result");
    auto& s          = sequence(tx.members[0]);
    const auto begin = tx.rows[0].main_frontier;
    const auto end   = begin + result.processed;
    const auto first = std::min(end, std::max(begin, scoring->first_target - 1));
    if (result.scores.target_logprobs.size() != end - first)
        throw std::logic_error("score result has the wrong target extent");
    if (result.scores.target_logprobs.size() >
        scoring->scores.target_logprobs.capacity() - scoring->scores.target_logprobs.size())
        throw std::logic_error("score result exceeds its prepared target capacity");
    tx.completion        = std::make_unique<context::ContextTransaction>(directory);
    const auto completed = prepare_commit(s, tx.rows[0], tx.rows[0].ids, false, *tx.completion);
    tx.completion->prepare_actions();
    Commit commit;
    commit.rows.push_back({member(s), tx.rows[0].states, result.processed});
    commit.actions = actions(*tx.completion);
    (void)executor->commit(tx.ticket, std::move(commit));
    tx.completion->mark_submitted();
    tx.completion->adopt();
    adopt_commit(s, completed, result.processed != 0, false);
    const auto append = [](auto& destination, const auto& source) {
        destination.insert(destination.end(), source.begin(), source.end());
    };
    append(scoring->scores.target_logprobs, result.scores.target_logprobs);
    append(scoring->scores.reference_logprobs, result.scores.reference_logprobs);
    append(scoring->scores.argmax_tokens, result.scores.argmax_tokens);
    append(scoring->scores.argmax_logprobs, result.scores.argmax_logprobs);
    append(scoring->scores.hidden_rows, result.scores.hidden_rows);
    tx.committed = true;
}

} // namespace ninfer::models::qwen4_exp::detail
