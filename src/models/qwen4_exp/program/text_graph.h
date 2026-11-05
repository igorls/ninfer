#pragma once

#include "core/decode_graph.h"

#include "models/qwen4_exp/execution/text.h"
#include "models/common/moe/schedule.h"

namespace ninfer::models::qwen4_exp {

struct GraphSegment {
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;

    void prepare(CudaExecutionView, const std::function<void()>&);
};

// One exact T/B traversal. Resident graphs surround PLE; offload retains layer handoffs.
// The activation frame and PLE slots belong to Resources and outlive every graph.
class TextGraph {
public:
    void prepare(execution::TextRun&, const execution::Parameters&, moe::Schedule*,
                 std::span<const std::uint32_t> bounds, std::array<Tensor, 2> ple,
                 execution::TextOutputs, std::int32_t columns, MemoryArena&, DeviceContext&);
    void select(std::uint32_t visible_tokens);
    void begin_prefix(CudaExecutionView);
    [[nodiscard]] bool advance_prefix(CudaExecutionView);
    void consume_ple(std::uint32_t slot, CudaExecutionView);
    [[nodiscard]] bool advance_suffix(CudaExecutionView);
    void cancel();

    [[nodiscard]] std::size_t workspace_bytes() const {
        return profiles_[installed_].workspace_bytes;
    }

private:
    struct Profile {
        std::uint32_t bound;
        std::vector<std::shared_ptr<const moe::LayerGraphDefinition>> layers;
        DecodeGraphDefinition resident_prefix, resident_suffix;
        std::size_t workspace_bytes = 0;
    };

    [[nodiscard]] bool advance(std::uint32_t end, CudaExecutionView);
    GraphSegment stem_, ple_[2], finish_;
    DecodeGraphExecutable resident_prefix_, resident_suffix_;
    std::vector<Profile> profiles_;
    std::vector<std::unique_ptr<moe::LayerGraphExecutable>> layers_;
    Tensor block_output_;
    std::vector<ExpertBankId> banks_;
    std::uint32_t ple_layer_ = 0;
    std::size_t installed_   = 0;
    bool resident_          = false;
    std::uint32_t next_layer_ = 0;
    bool waiting_layer_ = false;
};

} // namespace ninfer::models::qwen4_exp
