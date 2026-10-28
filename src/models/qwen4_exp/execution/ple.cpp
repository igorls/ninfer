#include "models/qwen4_exp/execution/ple.h"
#include "models/qwen4_exp/execution/workspace.h"

namespace ninfer::models::qwen4_exp::execution {

void ple(const Tensor& embedding, const TextConfig& c, const PleParameters& p,
         const BatchView& batch, const ops::PleConvStateView& before,
         std::optional<ops::PleConvStateView> after, ops::PleConvRecordWindowView records,
         const HcParameters& next, Tensor& residual, Tensor& normalized, MemoryArena& arena,
         CudaExecutionView cuda) {
    const auto h = dimension(c.hidden), streams = dimension(c.hc.streams);
    const auto channels = dimension(c.residual_width());
    auto scope          = arena.scope();
    const auto t = batch.tokens, b = batch.batch, m = t * b;
    auto buffers = workspace::ple(arena, c, m, !records.values.data);
    if (!records.values.data)
        records = {buffers.records.view({channels, t, b}), batch.record_offsets};
    Tensor key, value;
    const auto input = embedding.view({dimension(c.ple.embedding),m});
    if (const auto* one = std::get_if<ops::SingleProjectionWeight>(&p.projection)) {
        auto scratch = arena.scope();
        ops::linear(input, one->weight, buffers.projection, one->policy, arena, cuda);
        const auto parent = buffers.projection.view({channels+h,t,b});
        key = Tensor(parent.data,parent.device,DType::BF16,{h,streams,t,b});
        key.nb[2]=parent.nb[1]; key.nb[3]=parent.nb[2];
        value=parent.slice(0,channels,h);
    } else {
        const auto& pair=std::get<ops::PairedProjectionWeights>(p.projection);
        auto* data=static_cast<std::byte*>(buffers.projection.data);
        auto key_matrix=Tensor(data,cuda.device(),DType::BF16,{channels,m});
        auto value_matrix=Tensor(data+std::size_t(channels)*m*2,cuda.device(),DType::BF16,{h,m});
        { auto scratch=arena.scope(); ops::linear(input,pair.first,key_matrix,ops::LinearPolicy::A16Only,arena,cuda); }
        { auto scratch=arena.scope(); ops::linear(input,pair.second,value_matrix,ops::LinearPolicy::A16Only,arena,cuda); }
        key=key_matrix.view({h,streams,t,b}); value=value_matrix.view({h,t,b});
    }
    auto residual_view   = residual.view({h, streams, t, b});
    auto normalized_view = normalized.view({h, streams, t, b});
    auto gated           = buffers.gated.view({h, streams, t, b});
    ops::ple_gate({key, value}, residual_view, p.gate, {batch.valid_columns}, records, gated, cuda);
    ops::ple_conv_add_norm(residual_view, gated, p.convolution, {batch.valid_columns}, before,
                           records, next.norm, p.gate.rms_epsilon, residual_view, normalized_view,
                           cuda);
    if (after)
        ops::ple_state_extract(before, {records.values, batch.valid_columns}, batch.valid_columns,
                               *after, cuda);
}

} // namespace ninfer::models::qwen4_exp::execution
