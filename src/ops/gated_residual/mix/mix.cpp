#include "ninfer/ops/gated_residual.h"
#include "ops/gated_residual/mix/launch.h"
#include <stdexcept>
#include <algorithm>

namespace ninfer::ops {
namespace mix = detail::hc::mix;

namespace {
Weight describe_weight(const LinearProfile& profile) {
    Weight w;
    w.qtype  = profile.qtype;
    w.layout = profile.layout;
    w.ndim   = 2;
    w.n = w.shape[0] = w.padded_shape[0] = profile.n;
    w.k = w.shape[1] = w.padded_shape[1] = profile.k;
    if (profile.n > 0 && profile.k > 0)
        w.payload_bytes = std::uint64_t(profile.n) * profile.k * sizeof(std::uint16_t);
    return w;
}

mix::Weights block_weights(const HcMixWeights& w) {
    const bool adjacent = w.down.qdata && w.injection.qdata && w.down.payload == w.injection.payload &&
        w.down.payload_bytes >= std::size_t(detail::hc::kRank + detail::hc::kStreams) * detail::hc::kResidualWidth * 2 &&
        static_cast<const std::byte*>(w.down.qdata) + std::size_t(detail::hc::kRank) * detail::hc::kResidualWidth * 2 == w.injection.qdata;
    return {adjacent ? mix::Kind::Block : mix::Kind::Final, w.down, w.up, &w.injection};
}

void execute_mix(mix::Weights weights, const Tensor& normalized, MemoryArena& workspace,
                 Tensor& output, Tensor* injection, CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument("hc_mix: workspace belongs to another device");
    }
    const auto o = mix::validate(weights, normalized, output, injection, execution);
    const auto p = mix::select_plan(weights.kind, o.columns);
    auto scope   = workspace.scope();
    const auto scratch =
        mix::allocate_workspace(workspace, mix::layout(weights.kind, p, o.columns));
    mix::validate_scratch(weights, normalized, output, injection, scratch.storage);
    mix::launch(weights.kind, o, p, scratch, execution);
}
} // namespace

void hc_mix(const Tensor& normalized, const HcMixWeights& weights, MemoryArena& workspace,
            Tensor& block_input, Tensor& injection, CudaExecutionView execution) {
    execute_mix(block_weights(weights), normalized, workspace,
                block_input, &injection, execution);
}

void hc_mix(const Tensor& normalized, const HcFinalMixWeights& weights, MemoryArena& workspace,
            Tensor& sample_hidden, CudaExecutionView execution) {
    execute_mix({mix::Kind::Final, weights.down, weights.up}, normalized, workspace, sample_hidden,
                nullptr, execution);
}

std::size_t hc_mix_workspace_capacity_bytes(const HcMixWeights& weights, std::int32_t first,
                                            std::int32_t last) {
    return mix::capacity(block_weights(weights), first, last);
}

std::size_t hc_mix_workspace_capacity_bytes(const HcFinalMixWeights& weights, std::int32_t first,
                                            std::int32_t last) {
    return mix::capacity({mix::Kind::Final, weights.down, weights.up}, first, last);
}

std::size_t hc_mix_workspace_capacity_bytes(const HcMixProfile& p, std::int32_t first,
                                            std::int32_t last) {
    const auto down = describe_weight(p.down), injection = describe_weight(p.injection), up = describe_weight(p.up);
    return std::max(mix::capacity({mix::Kind::Block, down, up, &injection}, first, last),
                    mix::capacity({mix::Kind::Final, down, up, &injection}, first, last));
}

std::size_t hc_mix_workspace_capacity_bytes(const HcFinalMixProfile& p, std::int32_t first,
                                            std::int32_t last) {
    return hc_mix_workspace_capacity_bytes(
        HcFinalMixWeights{describe_weight(p.down), describe_weight(p.up)}, first, last);
}

} // namespace ninfer::ops
