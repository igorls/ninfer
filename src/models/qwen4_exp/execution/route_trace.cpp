#include "models/qwen4_exp/execution/route_trace.h"

#include "core/device.h"

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen4_exp::execution {
namespace {

constexpr std::int32_t kRoutedExperts = 512;

} // namespace

std::size_t route_trace_host_ids(std::int32_t columns, std::uint32_t slices,
                                 std::int32_t proposal_steps, std::int32_t proposal_columns) {
    return static_cast<std::size_t>(kRouteTraceTopK) *
           (static_cast<std::size_t>(columns) * slices +
            static_cast<std::size_t>(proposal_steps) * static_cast<std::size_t>(proposal_columns));
}

RouteTrace::RouteTrace(Tensor device, std::int32_t* host, std::uint32_t text_layers, bool mtp,
                       std::int32_t proposal_steps, std::int32_t proposal_columns)
    : device_(device), host_(host), text_layers_(text_layers),
      slices_(text_layers + (mtp ? 1U : 0U)), columns_(device.ne[0] / kRouteTraceTopK),
      proposal_steps_(mtp ? proposal_steps : 0), proposal_columns_(mtp ? proposal_columns : 0) {
    if (device_.dtype != DType::I32 || device_.data == nullptr || host_ == nullptr ||
        text_layers_ == 0 || columns_ <= 0 || device_.ne[0] != columns_ * kRouteTraceTopK ||
        device_.ne[1] != static_cast<std::int32_t>(slices_) || !device_.is_contiguous() ||
        proposal_steps_ < 0 || proposal_columns_ < 0) {
        throw std::invalid_argument("routed-expert trace buffers are inconsistent");
    }
}

Tensor RouteTrace::slice(std::uint32_t layer, std::int32_t columns) const {
    if (layer >= slices_ || columns <= 0 || columns > columns_) {
        throw std::out_of_range("routed-expert trace slice is outside the planned buffer");
    }
    auto* base = static_cast<std::int32_t*>(device_.data) +
                 static_cast<std::size_t>(layer) * kRouteTraceTopK * columns_;
    return Tensor(base, device_.device, DType::I32, {kRouteTraceTopK, columns});
}

void RouteTrace::download_text(std::int32_t columns, cudaStream_t stream) const {
    if (columns <= 0 || columns > columns_) {
        throw std::out_of_range("routed-expert trace download exceeds the planned columns");
    }
    const std::size_t pitch = static_cast<std::size_t>(kRouteTraceTopK) * columns_ * sizeof(int);
    CUDA_CHECK(cudaMemcpy2DAsync(host_, pitch, device_.data, pitch,
                                 static_cast<std::size_t>(kRouteTraceTopK) * columns * sizeof(int),
                                 text_layers_, cudaMemcpyDeviceToHost, stream));
}

void RouteTrace::download_proposal(std::int32_t step, std::int32_t columns,
                                   cudaStream_t stream) const {
    if (step < 0 || step >= proposal_steps_ || columns <= 0 || columns > proposal_columns_ ||
        columns > columns_) {
        throw std::out_of_range("routed-expert proposal trace is outside the planned slots");
    }
    std::int32_t* destination = host_ + static_cast<std::size_t>(kRouteTraceTopK) *
                                            (static_cast<std::size_t>(columns_) * slices_ +
                                             static_cast<std::size_t>(step) * proposal_columns_);
    CUDA_CHECK(cudaMemcpyAsync(destination, slice(mtp_layer(), columns).data,
                               static_cast<std::size_t>(kRouteTraceTopK) * columns * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
}

std::uint16_t RouteTrace::expert(std::int32_t id) const {
    if (id < 0 || id >= kRoutedExperts) {
        throw std::logic_error("routed-expert trace holds an expert id outside [0,512)");
    }
    return static_cast<std::uint16_t>(id);
}

void RouteTrace::record_text(ExpertRouteKind kind, std::span<const RouteTraceColumn> columns) {
    if (columns.empty() || columns.size() > static_cast<std::size_t>(columns_)) {
        throw std::out_of_range("routed-expert trace record exceeds the planned columns");
    }
    RouteTraceRecord record{.kind        = kind,
                            .first_layer = 0,
                            .layer_count = static_cast<std::uint16_t>(text_layers_),
                            .columns     = {columns.begin(), columns.end()}};
    record.experts.reserve(columns.size() * text_layers_ * kRouteTraceTopK);
    for (std::size_t column = 0; column < columns.size(); ++column) {
        for (std::uint32_t layer = 0; layer < text_layers_; ++layer) {
            const std::int32_t* ids = host_ +
                                      static_cast<std::size_t>(layer) * kRouteTraceTopK * columns_ +
                                      column * kRouteTraceTopK;
            for (std::int32_t rank = 0; rank < kRouteTraceTopK; ++rank) {
                record.experts.push_back(expert(ids[rank]));
            }
        }
    }
    records_.push_back(std::move(record));
}

void RouteTrace::record_proposals(std::int32_t steps, std::int32_t batch,
                                  std::span<const RouteTraceColumn> columns) {
    if (steps <= 0 || steps > proposal_steps_ || batch <= 0 || batch > proposal_columns_ ||
        columns.size() != static_cast<std::size_t>(steps) * static_cast<std::size_t>(batch)) {
        throw std::out_of_range("routed-expert proposal record does not match its slots");
    }
    const std::int32_t* slots =
        host_ + static_cast<std::size_t>(kRouteTraceTopK) * columns_ * slices_;
    for (std::int32_t step = 0; step < steps; ++step) {
        RouteTraceRecord record{
            .kind        = ExpertRouteKind::MtpProposal,
            .first_layer = static_cast<std::uint16_t>(mtp_layer()),
            .layer_count = 1,
            .columns     = {columns.begin() + static_cast<std::ptrdiff_t>(step) * batch,
                            columns.begin() + static_cast<std::ptrdiff_t>(step + 1) * batch}};
        record.experts.reserve(static_cast<std::size_t>(batch) * kRouteTraceTopK);
        const std::int32_t* ids =
            slots + static_cast<std::size_t>(step) * proposal_columns_ * kRouteTraceTopK;
        for (std::int32_t i = 0; i < batch * kRouteTraceTopK; ++i) {
            record.experts.push_back(expert(ids[i]));
        }
        records_.push_back(std::move(record));
    }
}

std::vector<RouteTraceRecord> RouteTrace::take() noexcept {
    std::vector<RouteTraceRecord> out;
    out.swap(records_);
    return out;
}

} // namespace ninfer::models::qwen4_exp::execution
