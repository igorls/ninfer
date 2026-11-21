#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/linear.h"

#include <algorithm>
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
    operation.feature_pending       = false;
    operation.prompt_hidden_pending = false;
    const auto* forward = std::get_if<ForwardWork>(&operation.request.work.operation);
    if (!forward ||
        (forward->purpose != ForwardPurpose::Prefill &&
         forward->purpose != ForwardPurpose::Replay) ||
        !operation.width)
        return;
    const auto& window = forward->inputs.at(operation.forward_row);
    if (!window.readout) return;
    auto& frame = resources.frame;
    auto hidden = frame.input_view(static_cast<int>(operation.width), 1).hidden;
    if (window.readout->capture_hidden_rows) {
        const auto& req_layers = window.readout->hidden_layers;
        const std::size_t slot_bytes =
            static_cast<std::size_t>(frame.spec.hidden) *
            static_cast<std::size_t>(frame.spec.prefill_chunk) * sizeof(std::uint16_t);
        auto* host_base = static_cast<std::uint8_t*>(frame.host_prompt_hidden.data);
        const std::size_t bytes = hidden.bytes();
        if (bytes > slot_bytes)
            throw std::logic_error("prompt hidden readout exceeds frame capacity");
        if (req_layers.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(host_base, hidden.data, bytes,
                                       cudaMemcpyDeviceToHost, device.stream()));
        } else {
            for (std::size_t k = 0; k < req_layers.size(); ++k) {
                if (req_layers[k] == kHiddenStateTapFinalNorm) {
                    CUDA_CHECK(cudaMemcpyAsync(host_base + k * slot_bytes, hidden.data, bytes,
                                               cudaMemcpyDeviceToHost, device.stream()));
                }
            }
        }
        operation.prompt_hidden_pending = true;
    }
    if (!resources.readout) return;
    auto& readout = *resources.readout;
    const auto& options = window.readout->options;
    const auto begin = operation.rows.at(operation.forward_row).main_frontier;
    const auto end = begin + operation.width;
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
    if (operation.prompt_hidden_pending) {
        const auto* base = static_cast<const std::uint16_t*>(resources.frame.host_prompt_hidden.data);
        const std::size_t H = static_cast<std::size_t>(resources.frame.spec.hidden);
        const std::size_t W = static_cast<std::size_t>(operation.width);
        const std::size_t K = std::max<std::size_t>(1, window.readout->hidden_layers.size());
        const std::size_t slot_elems =
            H * static_cast<std::size_t>(resources.frame.spec.prefill_chunk);
        output.prompt_hidden_size = static_cast<std::uint32_t>(H);
        if (K == 1) {
            output.prompt_hidden_rows.insert(output.prompt_hidden_rows.end(), base, base + W * H);
        } else {
            const std::size_t old_size = output.prompt_hidden_rows.size();
            output.prompt_hidden_rows.resize(old_size + W * K * H);
            std::uint16_t* dst = output.prompt_hidden_rows.data() + old_size;
            for (std::size_t t = 0; t < W; ++t) {
                for (std::size_t k = 0; k < K; ++k) {
                    const std::uint16_t* src = base + k * slot_elems + t * H;
                    std::copy_n(src, H, dst);
                    dst += H;
                }
            }
        }
    }
}

} // namespace ninfer::models::qwen3_5::detail
