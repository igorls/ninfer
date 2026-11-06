#include "models/qwen4_exp/program/internal.h"

#include <algorithm>
#include <iterator>

namespace ninfer::models::qwen4_exp::detail {

void RankProgramImpl::enqueue_round_readout() {
    auto& f = *compute;
    if (!resources.readout) return;
    const auto& buffers = resources.plan.persistent.buffers;
    const auto width = static_cast<int>(f.verify ? f.verify->width : 1);
    const auto batch = static_cast<int>(f.rows.size());
    auto logits = packed_tensor(*buffers.logits,
        {dimension(parameters.profile->config.text.vocabulary), width, batch});
    auto tokens = packed_tensor(*buffers.sampled_ids, {width, batch});
    for (int b = 0; b < batch; ++b) {
        const auto& row = f.rows[b];
        if (!row.logprobs.enabled) continue;
        const auto count = static_cast<int>(f.verify ? f.verify->extents[b] + 1 : 1);
        auto columns = logits.slice(2, b, 1).slice(1, 0, count)
                           .view({logits.ne[0], count});
        auto sampled = tokens.slice(1, b, 1).slice(0, 0, count).view({count});
        std::optional<Tensor> allowed;
        if (row.constrained) {
            auto masks = device_tensor(*buffers.grammar_masks);
            allowed = masks.slice(2, b, 1).slice(1, 0, count).view({masks.ne[0], count});
        }
        resources.readout->enqueue_round(b, columns, sampled, allowed ? &*allowed : nullptr,
            parameters.profile->public_tokens, row.logprobs, cuda);
    }
}

void RankProgramImpl::enqueue_prompt_readout() {
    auto& f = *compute;
    f.prompt_indices.clear();
    f.feature_pending = false;
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    if (!forward || forward->kind != ForwardKind::Prefill || !forward->readout || !f.tokens)
        return;
    const auto& requested = *forward->readout;
    const auto begin = f.rows.front().main_frontier;
    const auto end = begin + f.tokens;
    const auto hidden = dimension(parameters.profile->config.text.hidden);
    if (requested.feature_position && *requested.feature_position >= begin &&
        *requested.feature_position < end) {
        resources.readout->enqueue_feature(
            f.sample.slice(1, static_cast<int>(*requested.feature_position - begin), 1)
                .view({hidden}), cuda);
        f.feature_pending = true;
    }
    for (std::size_t i = 0; i < requested.options.prompt_positions.size(); ++i) {
        const auto position = requested.options.prompt_positions[i];
        if (position < begin || position >= end) continue;
        auto scratch = resources.workspace.scope();
        auto sample = f.sample.slice(1, static_cast<int>(position - begin), 1).view({hidden, 1});
        auto logits = packed_tensor(*resources.plan.persistent.buffers.logits,
            {dimension(parameters.profile->config.text.vocabulary), 1});
        execution::project_head(sample, parameters.text.head, logits, resources.workspace, cuda);
        resources.readout->enqueue_prompt(i, logits, parameters.profile->public_tokens,
            requested.options, cuda);
        f.prompt_indices.push_back(i);
    }
}

void RankProgramImpl::collect_readout(std::size_t b, RowResult& output) {
    const auto& f = *compute;
    const auto& row = f.rows[b];
    if (!output.tokens.empty() && row.logprobs.enabled)
        output.logprobs = resources.readout->collect_round(static_cast<int>(b),
            f.verify ? f.verify->extents[b] + 1 : 1, output.tokens, row.logprobs);
    if (b != 0) return;
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    if (!forward || !forward->readout) return;
    const auto& requested = *forward->readout;
    for (const auto index : f.prompt_indices)
        output.prompt_logprobs.push_back(resources.readout->collect_prompt(index,
            requested.next_ids.at(index), requested.options));
    if (f.feature_pending) output.reasoning_features = resources.readout->collect_feature();
}

void RankProgramImpl::collect_trace() {
    auto& f = *compute;
    if (!route_trace || !f.tokens) return;
    std::vector<execution::RouteTraceColumn> columns;
    if (f.verify) {
        const auto steps = *std::max_element(f.verify->extents.begin(), f.verify->extents.end());
        for (std::uint32_t step = 0; step < steps; ++step) {
            columns.clear();
            for (std::size_t b = 0; b < f.rows.size(); ++b)
                if (step < f.verify->extents[b])
                    columns.push_back({f.rows[b].member.residency.lane.value,
                        f.rows[b].main_frontier - 1 + step, static_cast<std::uint32_t>(b)});
            route_trace->record_proposal(step, columns);
        }
        columns.clear();
    }
    for (std::size_t b = 0; b < f.rows.size(); ++b) {
        const auto count = f.verify ? f.verify->extents[b] + 1 : f.tokens;
        for (std::uint32_t j = 0; j < count; ++j)
            columns.push_back({f.rows[b].member.residency.lane.value,
                f.rows[b].main_frontier + j, static_cast<std::uint32_t>(b * f.tokens + j)});
    }
    route_trace->record_text(f.verify ? ExpertRouteKind::MtpVerify :
        f.sequence ? ExpertRouteKind::Prefill : ExpertRouteKind::Decode, columns);
    auto records = route_trace->take();
    f.result.routes.insert(f.result.routes.end(), std::make_move_iterator(records.begin()),
        std::make_move_iterator(records.end()));
}

} // namespace ninfer::models::qwen4_exp::detail
