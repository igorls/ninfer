#include "models/qwen4_exp/program/internal.h"
#include "core/transfer.h"
#include "ninfer/ops/candidate_logprobs.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

void ProgramImpl::enqueue_readout(Sequence& s, const Tensor& logits, const Tensor& sampled,
                                  const Tensor* allowed, bool prompt) {
    const auto& options = s.request->logprobs;
    s.readout_layout = {};
    if (!options.enabled) return;
    const auto columns = static_cast<std::int32_t>(sampled.numel());
    const auto count = static_cast<std::int32_t>(options.candidates.size());
    const auto top = static_cast<std::int32_t>(prompt ? 0 : options.top);
    const runtime::TokenLogprobReadoutLayout layout{
        std::size_t(columns), std::size_t(count), std::size_t(top)};
    const auto bytes = layout.floats() * sizeof(float);
    if (!s.readout_host || s.readout_host->size() < bytes) s.readout_host.emplace(bytes);
    if (!s.readout_input)
        s.readout_input.emplace((kMaximumLogprobCandidates + 32) * sizeof(TokenId));
    auto scope = resources.workspace.scope();
    auto buffers = execution::workspace::token_readout(resources.workspace, columns, count, top);
    auto* base = static_cast<float*>(buffers.values.data);
    Tensor values(base, cuda.device(), DType::FP32, {columns, 2});
    std::optional<Tensor> candidates, candidate_values, top_ids, top_values;
    if (count) {
        candidates = buffers.candidates;
        auto* input = static_cast<TokenId*>(s.readout_input->data());
        std::copy(options.candidates.begin(), options.candidates.end(), input);
        copy_bytes_async({candidates->data, std::size_t(count) * 4, cuda.device()},
                         s.readout_input->span().subspan(0, std::size_t(count) * 4), cuda);
        candidate_values.emplace(base + 2 * columns, cuda.device(), DType::FP32,
                                  std::initializer_list<std::int32_t>{columns, count, 2});
    }
    if (top) {
        top_ids.emplace(base + layout.top_id_offset(), cuda.device(), DType::I32,
                         std::initializer_list<std::int32_t>{columns, top});
        top_values.emplace(base + layout.top_value_offset(), cuda.device(), DType::FP32,
                            std::initializer_list<std::int32_t>{columns, top, 2});
    }
    ops::candidate_logprobs(logits, parameters.profile->public_tokens, sampled,
                            candidates ? &*candidates : nullptr, allowed, values,
                            candidate_values ? &*candidate_values : nullptr, cuda,
                            top_ids ? &*top_ids : nullptr, top_values ? &*top_values : nullptr);
    copy_bytes_async(s.readout_host->span().subspan(0, bytes),
                     {base, bytes, cuda.device()}, cuda);
    s.readout_layout = layout;
}

void ProgramImpl::collect_readout(Sequence& s, std::span<const TokenId> tokens, bool prompt) {
    auto& output = prompt ? s.prompt_logprobs : s.round_logprobs;
    if (!prompt) output.clear();
    if (!s.request->logprobs.enabled) return;
    const auto& layout = s.readout_layout;
    if (!s.readout_host || tokens.size() > layout.columns)
        throw std::logic_error("token readout is missing a licensed column");
    const std::span<const float> data(static_cast<const float*>(s.readout_host->data()),
                                      layout.floats());
    for (std::size_t i = 0; i < tokens.size(); ++i)
        output.push_back(runtime::assemble_token_logprobs(tokens[i], data, layout, i,
                                                          s.request->logprobs.candidates));
    s.readout_layout = {};
}

void ProgramImpl::read_prompt(Sequence& s, const Tensor& hidden, std::uint32_t begin) {
    const auto end = begin + static_cast<std::uint32_t>(hidden.ne[1]);
    const auto h = hidden.ne[0];
    if (const auto at = s.request->reasoning_feature_position; at && *at >= begin && *at < end) {
        PinnedHostBuffer host(std::size_t(h) * 2);
        auto row = hidden.slice(1, *at - begin, 1);
        copy_bytes_async(host.span(), {row.data, row.bytes(), cuda.device()}, cuda);
        ple.wait_compute();
        const auto* values = static_cast<const std::uint16_t*>(host.data());
        s.reasoning_features.resize(h);
        std::transform(values, values + h, s.reasoning_features.begin(), runtime::bf16_to_float);
    }
    const auto& positions = s.request->logprobs.prompt_positions;
    auto first = std::lower_bound(positions.begin(), positions.end(), begin);
    const auto last = std::lower_bound(first, positions.end(), end);
    while (first != last) {
        const auto n = static_cast<std::int32_t>(std::min<std::ptrdiff_t>(32, last - first));
        auto scope = resources.workspace.scope();
        auto gathered = resources.workspace.alloc(DType::BF16, {h, n});
        auto score = execution::workspace::score(resources.workspace,
                                                 parameters.model.config().text.vocabulary, n);
        if (!s.readout_input)
            s.readout_input.emplace((kMaximumLogprobCandidates + 32) * sizeof(TokenId));
        auto* input = static_cast<TokenId*>(s.readout_input->data()) + kMaximumLogprobCandidates;
        const std::span<TokenId> targets(input, n);
        for (int i = 0; i < n; ++i) {
            const auto row = hidden.slice(1, first[i] - begin, 1);
            const auto destination = gathered.slice(1, i, 1);
            copy_bytes_async({destination.data, destination.bytes(), cuda.device()},
                             {row.data, row.bytes(), cuda.device()}, cuda);
            targets[i] = s.identity.tokens[first[i] + 1];
        }
        copy_bytes_async({score.targets.data, std::size_t(n) * 4, cuda.device()},
                         {targets.data(), std::size_t(n) * 4, Device::cpu(), true}, cuda);
        execution::project_head(gathered, parameters.text.head, score.logits,
                                resources.workspace, cuda);
        enqueue_readout(s, score.logits, score.targets.view({n}), nullptr, true);
        ple.wait_compute();
        collect_readout(s, targets, true);
        first += n;
    }
}

} // namespace ninfer::models::qwen4_exp::detail

namespace ninfer::models::qwen4_exp {
std::span<const TokenLogprobs> Program::round_token_logprobs(runtime::LaneId lane) const {
    if (lane.value >= impl_->shape.max_concurrency || !impl_->sequences[lane.value]) return {};
    return impl_->sequences[lane.value]->round_logprobs;
}

std::span<const TokenLogprobs> Program::prompt_token_logprobs(runtime::LaneId lane) const {
    if (lane.value >= impl_->shape.max_concurrency || !impl_->sequences[lane.value]) return {};
    return impl_->sequences[lane.value]->prompt_logprobs;
}

std::span<const float> Program::reasoning_features(runtime::LaneId lane) const {
    if (lane.value >= impl_->shape.max_concurrency || !impl_->sequences[lane.value]) return {};
    return impl_->sequences[lane.value]->reasoning_features;
}
} // namespace ninfer::models::qwen4_exp
