#include "models/common/qwen/distribution_readout.h"
#include "ninfer/ops/candidate_logprobs.h"
#include "runtime/contract/token_logprobs.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen {
namespace {
constexpr auto kPromptFloats = 2U + 2U * kMaximumLogprobCandidates;
}

DistributionReadoutLayout plan_distribution_readout(LayoutBuilder& builder,
                                                     std::int32_t columns, std::int32_t rows,
                                                     std::int32_t hidden) {
    if (columns < 1 || columns > 16 || rows < 1 || rows > kMaximumConcurrency || hidden < 1)
        throw std::invalid_argument("distribution readout geometry is invalid");
    DistributionReadoutLayout out;
    out.columns = columns;
    out.rows = rows;
    out.hidden = hidden;
    out.round_floats = std::size_t(columns) *
        (2U + 2U * kMaximumLogprobCandidates + 3U * kMaximumTopLogprobs);
    out.candidate_ids = builder.add_tensor(DType::I32,
        {static_cast<std::int32_t>(kMaximumLogprobCandidates), rows}, 256, "readout candidates");
    out.round = builder.add_tensor(DType::FP32,
        {static_cast<std::int32_t>(out.round_floats), rows}, 256, "round distribution readout");
    out.prompt_ids = builder.add_tensor(DType::I32,
        {static_cast<std::int32_t>(kMaximumPromptReadouts)}, 256, "prompt readout target ids");
    out.prompt_offset = out.round_floats * rows * sizeof(float);
    out.feature_offset = out.prompt_offset + kMaximumPromptReadouts * kPromptFloats * sizeof(float);
    out.host_bytes = out.feature_offset + std::size_t(hidden) * sizeof(std::uint16_t);
    return out;
}

DistributionReadout::DistributionReadout(MemorySpan backing,
                                         const DistributionReadoutLayout& layout)
    : layout_(layout), candidates_(layout.candidate_ids.bind(backing)),
      round_(layout.round.bind(backing)), prompt_ids_(layout.prompt_ids.bind(backing)),
      host_(layout.host_bytes) {}

void DistributionReadout::install(std::int32_t row, const TokenLogprobOptions& options,
                                   CudaExecutionView cuda) {
    if (row < 0 || row >= layout_.rows || options.candidates.size() > kMaximumLogprobCandidates ||
        options.top > kMaximumTopLogprobs)
        throw std::invalid_argument("distribution readout options exceed frame capacity");
    if (!options.candidates.empty())
        CUDA_CHECK(cudaMemcpyAsync(candidates_.slice(1, row, 1).data, options.candidates.data(),
            options.candidates.size() * sizeof(TokenId), cudaMemcpyHostToDevice, cuda.stream()));
}

void DistributionReadout::enqueue(std::int32_t row, const Tensor& logits, const Tensor& sampled,
                                   const Tensor* allowed, std::int32_t domain,
                                   const TokenLogprobOptions& options, void* destination,
                                   CudaExecutionView cuda) {
    const auto cols = sampled.ne[0];
    if (row < 0 || row >= layout_.rows || cols < 1 || cols > layout_.columns)
        throw std::invalid_argument("distribution readout exceeds frame columns");
    const auto count = static_cast<std::int32_t>(options.candidates.size());
    const auto top = static_cast<std::int32_t>(options.top);
    const runtime::TokenLogprobReadoutLayout layout{std::size_t(cols), options.candidates.size(), std::size_t(top)};
    auto* base = static_cast<float*>(round_.slice(1, row, 1).data);
    Tensor sampled_out(base, cuda.device(), DType::FP32, {cols, 2});
    std::optional<Tensor> ids, candidates_out, top_ids, top_out;
    if (count) {
        ids.emplace(candidates_.slice(1, row, 1).data, cuda.device(), DType::I32,
                     std::initializer_list<std::int32_t>{count});
        candidates_out.emplace(base + 2 * cols, cuda.device(), DType::FP32,
                                std::initializer_list<std::int32_t>{cols, count, 2});
    }
    if (top) {
        top_ids.emplace(base + layout.top_id_offset(), cuda.device(), DType::I32,
                         std::initializer_list<std::int32_t>{cols, top});
        top_out.emplace(base + layout.top_value_offset(), cuda.device(), DType::FP32,
                         std::initializer_list<std::int32_t>{cols, top, 2});
    }
    ops::candidate_logprobs(logits, domain, sampled, ids ? &*ids : nullptr, allowed, sampled_out,
        candidates_out ? &*candidates_out : nullptr, cuda, top_ids ? &*top_ids : nullptr,
        top_out ? &*top_out : nullptr);
    CUDA_CHECK(cudaMemcpyAsync(destination, base, layout.floats() * sizeof(float),
                                cudaMemcpyDeviceToHost, cuda.stream()));
}

void DistributionReadout::enqueue_round(std::int32_t row, const Tensor& logits,
                                         const Tensor& sampled, const Tensor* allowed,
                                         std::int32_t domain, const TokenLogprobOptions& options,
                                         CudaExecutionView cuda) {
    if (row < 0 || row >= layout_.rows)
        throw std::invalid_argument("distribution readout row is invalid");
    auto* destination = static_cast<float*>(host_.data()) + row * layout_.round_floats;
    enqueue(row, logits, sampled, allowed, domain, options, destination, cuda);
}

void DistributionReadout::install_prompt(std::span<const TokenId> next_ids,
                                         CudaExecutionView cuda) {
    if (next_ids.size() > kMaximumPromptReadouts)
        throw std::invalid_argument("prompt readout exceeds frame capacity");
    if (!next_ids.empty())
        CUDA_CHECK(cudaMemcpyAsync(prompt_ids_.data, next_ids.data(), next_ids.size_bytes(),
                                    cudaMemcpyHostToDevice, cuda.stream()));
}

void DistributionReadout::enqueue_prompt(std::size_t index, const Tensor& logits,
                                         std::int32_t domain, const TokenLogprobOptions& options,
                                         CudaExecutionView cuda) {
    if (index >= kMaximumPromptReadouts || options.top)
        throw std::invalid_argument("prompt readout position or top count is invalid");
    auto id = prompt_ids_.slice(0, static_cast<std::int32_t>(index), 1);
    auto* destination = static_cast<std::byte*>(host_.data()) + layout_.prompt_offset +
                        index * kPromptFloats * sizeof(float);
    enqueue(0, logits, id, nullptr, domain, options, destination, cuda);
}

void DistributionReadout::enqueue_feature(const Tensor& hidden, CudaExecutionView cuda) {
    if (hidden.dtype != DType::BF16 || !hidden.is_contiguous() || hidden.numel() != layout_.hidden)
        throw std::invalid_argument("reasoning readout is not one normalized hidden row");
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(host_.data()) + layout_.feature_offset,
        hidden.data, std::size_t(layout_.hidden) * sizeof(std::uint16_t),
        cudaMemcpyDeviceToHost, cuda.stream()));
}

std::vector<TokenLogprobs> DistributionReadout::collect_round(std::int32_t row,
    std::uint32_t columns, std::span<const TokenId> tokens, const TokenLogprobOptions& options) const {
    if (row < 0 || row >= layout_.rows || columns > static_cast<std::uint32_t>(layout_.columns) ||
        tokens.size() > columns)
        throw std::invalid_argument("distribution readout result shape is invalid");
    const runtime::TokenLogprobReadoutLayout layout{columns, options.candidates.size(), options.top};
    const auto* source = static_cast<const float*>(host_.data()) + row * layout_.round_floats;
    std::vector<TokenLogprobs> result;
    result.reserve(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); ++i)
        result.push_back(runtime::assemble_token_logprobs(tokens[i], {source, layout.floats()},
            layout, i, options.candidates));
    return result;
}

TokenLogprobs DistributionReadout::collect_prompt(std::size_t index, TokenId token,
    const TokenLogprobOptions& options) const {
    if (index >= kMaximumPromptReadouts || options.top)
        throw std::invalid_argument("prompt readout result shape is invalid");
    const auto* source = reinterpret_cast<const float*>(
        static_cast<const std::byte*>(host_.data()) + layout_.prompt_offset) + index * kPromptFloats;
    const runtime::TokenLogprobReadoutLayout layout{1, options.candidates.size(), 0};
    return runtime::assemble_token_logprobs(token, {source, layout.floats()}, layout, 0,
        options.candidates);
}

std::vector<float> DistributionReadout::collect_feature() const {
    const auto* source = reinterpret_cast<const std::uint16_t*>(
        static_cast<const std::byte*>(host_.data()) + layout_.feature_offset);
    std::vector<float> result(layout_.hidden);
    std::transform(source, source + layout_.hidden, result.begin(), runtime::bf16_to_float);
    return result;
}

} // namespace ninfer::models::qwen
