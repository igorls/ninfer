#include "models/qwen3_5/execution/ffn.h"

#include "core/layout.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {

std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                std::int32_t last) {
    if (first <= 0 || last < first) { throw std::invalid_argument("FFN: invalid column interval"); }
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        return ops::sparse_moe_workspace_capacity_bytes(moe->routed_gate_up.qtype,
                                                        moe->routed_down.qtype, first, last);
    }
    const auto& p    = std::get<DenseParameters>(parameters);
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {gu.n / 2, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
            gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            down.qtype, down.n, down.k, p.down.policy, first, last));
    }
    return layout.peak_bytes(1);
}

void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace,
         DeviceExecutionView execution) {
    const auto stream  = execution.stream;
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        const auto storage =
            workspace.alloc_bytes(ffn_workspace_bytes(parameters, columns, columns));
        WorkspaceArena scratch(storage);
        ops::sparse_moe(hidden, *moe, ops::SparseMoeEpilogue::AddResidual, residual, hints, scratch,
                        execution);
        return;
    }
    const auto& p     = std::get<DenseParameters>(parameters);
    const auto& gu    = p.gate_up.weight;
    const auto& down  = p.down.weight;
    Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
    {
        auto call = workspace.scope();
        ops::linear_swiglu(hidden, gu, activation, p.gate_up.policy, workspace, stream);
    }
    ops::linear_add(activation, down, residual, p.down.policy, workspace, stream);
}

} // namespace ninfer::models::qwen3_5::execution
