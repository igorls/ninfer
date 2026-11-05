#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/linear.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {
Tensor prefix(const Tensor& tensor, std::initializer_list<std::int32_t> shape) {
    return Tensor(MemorySpan{tensor.data, tensor.bytes(), tensor.device}, tensor.dtype, shape);
}
}

void RankProgramImpl::enqueue_round_readout(ComputeOperation& operation) {
    if (!resources.readout) return;
    auto& frame = resources.frame;
    const auto width = operation.speculative ? static_cast<int>(draft_window + 1) : 1;
    const auto batch = static_cast<int>(operation.batch);
    auto logits = prefix(frame.logits, {frame.spec.output_rows, width, batch});
    auto tokens = prefix(frame.output_tokens, {width, batch});
    for (int row = 0; row < batch; ++row) {
        const auto& sampling = operation.sampling.at(row);
        if (!sampling.logprobs.enabled) continue;
        const int count = frame.host_ingress->extents[row] + 1;
        if (count < 1 || count > width)
            throw std::logic_error("readout extent exceeds the frozen decode row");
        auto columns = logits.slice(2, row, 1).slice(1, 0, count)
                           .view({frame.spec.output_rows, count});
        auto sampled = tokens.slice(1, row, 1).slice(0, 0, count).view({count});
        std::optional<Tensor> allowed;
        if (sampling.constrained)
            allowed = frame.masks.slice(2, row, 1).slice(1, 0, count)
                          .view({frame.masks.ne[0], count});
        resources.readout->enqueue_round(row, columns, sampled, allowed ? &*allowed : nullptr,
            static_cast<int>(parameters.profile->public_tokens), sampling.logprobs,
            device.execution_view());
    }
}

void RankProgramImpl::enqueue_prompt_readout(ComputeOperation& operation) {
    operation.prompt_indices.clear();
    operation.feature_pending = false;
    const auto* forward = std::get_if<ForwardWork>(&operation.request.work.operation);
    if (!forward || forward->purpose != ForwardPurpose::Prefill || !operation.width) return;
    const auto& window = forward->inputs.at(operation.forward_row);
    if (!window.readout) return;
    auto& readout = *resources.readout;
    const auto& options = window.readout->options;
    const auto begin = operation.rows.at(operation.forward_row).main_frontier;
    const auto end = begin + operation.width;
    auto& frame = resources.frame;
    auto hidden = frame.input_view(static_cast<int>(operation.width), 1).hidden;
    if (window.readout->feature_position && *window.readout->feature_position >= begin &&
        *window.readout->feature_position < end) {
        readout.enqueue_feature(hidden.slice(1,
            static_cast<int>(*window.readout->feature_position - begin), 1), device.execution_view());
        operation.feature_pending = true;
    }
    // Reuse the target logit column only after Main has produced its normalized hidden rows;
    // the final sampling projection is submitted later and reads the unchanged hidden frame.
    for (std::size_t index = 0; index < options.prompt_positions.size(); ++index) {
        const auto position = options.prompt_positions[index];
        if (position < begin || position >= end) continue;
        auto scope = resources.work.scope();
        auto input = hidden.slice(1, static_cast<int>(position - begin), 1)
                         .view({frame.spec.hidden, 1});
        auto logits = prefix(frame.logits, {frame.spec.output_rows, 1});
        execution::project(input, parameters.text.output_head, logits, resources.work,
            device.execution_view());
        readout.enqueue_prompt(index, logits, static_cast<int>(parameters.profile->public_tokens),
            options, device.execution_view());
        operation.prompt_indices.push_back(index);
    }
}

void RankProgramImpl::collect_prompt_readout(ComputeOperation& operation) {
    const auto& work = std::get<ForwardWork>(operation.request.work.operation);
    const auto& window = work.inputs.at(operation.forward_row);
    if (!window.readout) return;
    auto& output = std::get<ForwardResult>(operation.result->output);
    for (const auto index : operation.prompt_indices)
        output.prompt_logprobs.push_back(resources.readout->collect_prompt(index,
            window.readout->next_ids.at(index), window.readout->options));
    if (operation.feature_pending) {
        if (!output.reasoning_features.empty())
            throw std::logic_error("reasoning feature row was read twice");
        output.reasoning_features = resources.readout->collect_feature();
    }
}

} // namespace ninfer::models::qwen3_5::detail
