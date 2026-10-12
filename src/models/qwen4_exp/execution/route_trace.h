#pragma once

// Optional routed-expert trace. Each MoE call of a traced unit writes its top-10 selection into
// one device layer slice; the unit mirrors its slices into pinned host memory before its host
// synchronization, and the Program turns the mirror into records after that synchronization.

#include "core/tensor.h"
#include "ninfer/types.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp::execution {

inline constexpr std::int32_t kRouteTraceTopK = 10;

// One traced column: the Program lane and the absolute position of its token.
struct RouteTraceColumn {
    std::uint32_t lane     = 0;
    std::uint32_t position = 0;
};

struct RouteTraceRecord {
    ExpertRouteKind kind      = ExpertRouteKind::Decode;
    std::uint16_t first_layer = 0;
    std::uint16_t layer_count = 0;
    std::vector<RouteTraceColumn> columns;
    std::vector<std::uint16_t> experts; // [column][layer][rank]
};

// Pinned host ids the trace needs: every device slice, plus one MTP slice per proposal step.
[[nodiscard]] std::size_t route_trace_host_ids(std::int32_t columns, std::uint32_t slices,
                                               std::int32_t proposal_steps,
                                               std::int32_t proposal_columns);

class RouteTrace {
public:
    // `device` is I32 [10 * columns, text_layers (+1 with MTP)]; `host` holds
    // route_trace_host_ids(columns, slices, proposal_steps, proposal_columns) ids.
    RouteTrace(Tensor device, std::int32_t* host, std::uint32_t text_layers, bool mtp,
               std::int32_t proposal_steps, std::int32_t proposal_columns);

    [[nodiscard]] std::uint32_t text_layers() const noexcept { return text_layers_; }

    [[nodiscard]] std::uint32_t mtp_layer() const noexcept { return text_layers_; }

    [[nodiscard]] std::int32_t columns() const noexcept { return columns_; }

    // I32 [10, columns] device slice of one layer (the MTP layer is mtp_layer()).
    [[nodiscard]] Tensor slice(std::uint32_t layer, std::int32_t columns) const;

    // Mirrors the text slices of a unit's `columns` columns into pinned host memory.
    void download_text(std::int32_t columns, cudaStream_t stream) const;
    // Mirrors the MTP slice of proposal `step` into that step's pinned host slot.
    void download_proposal(std::int32_t step, std::int32_t columns, cudaStream_t stream) const;

    // Records the mirrored text slices; call after the unit's host synchronization.
    void record_text(ExpertRouteKind kind, std::span<const RouteTraceColumn> columns);
    // Records `steps` mirrored proposal slices; columns are step-major (step * batch + row).
    void record_proposals(std::int32_t steps, std::int32_t batch,
                          std::span<const RouteTraceColumn> columns);

    [[nodiscard]] std::vector<RouteTraceRecord> take() noexcept;

private:
    [[nodiscard]] std::uint16_t expert(std::int32_t id) const;

    Tensor device_;
    std::int32_t* host_;
    std::uint32_t text_layers_;
    std::uint32_t slices_;
    std::int32_t columns_;
    std::int32_t proposal_steps_;
    std::int32_t proposal_columns_;
    std::vector<RouteTraceRecord> records_;
};

} // namespace ninfer::models::qwen4_exp::execution
