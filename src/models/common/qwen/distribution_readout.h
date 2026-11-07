#pragma once

#include "core/buffer.h"
#include "core/device.h"
#include "core/layout.h"
#include "ninfer/types.h"

#include <span>
#include <vector>

namespace ninfer::models::qwen {

// One compute frame owns these views through DeviceRetired. Options and token ids are
// owned by its submitted Work; neither this storage nor a completion event names a Session.
struct DistributionReadoutLayout {
    TensorRegion candidate_ids, round, prompt_ids;
    std::size_t round_floats = 0;
    std::size_t host_bytes = 0;
    std::size_t prompt_offset = 0, feature_offset = 0;
    std::int32_t columns = 0, rows = 0, hidden = 0;
};

[[nodiscard]] DistributionReadoutLayout
plan_distribution_readout(LayoutBuilder&, std::int32_t columns, std::int32_t rows,
                          std::int32_t hidden);

class DistributionReadout {
public:
    DistributionReadout(MemorySpan, const DistributionReadoutLayout&);
    void install(std::int32_t row, const TokenLogprobOptions&, CudaExecutionView);
    void enqueue_round(std::int32_t row, const Tensor& logits, const Tensor& sampled,
                        const Tensor* allowed, std::int32_t domain,
                        const TokenLogprobOptions&, CudaExecutionView);
    void install_prompt(std::span<const TokenId> next_ids, CudaExecutionView);
    void enqueue_prompt(std::size_t index, const Tensor& logits, std::int32_t domain,
                         const TokenLogprobOptions&, CudaExecutionView);
    void enqueue_feature(const Tensor& hidden, CudaExecutionView);
    [[nodiscard]] std::vector<TokenLogprobs>
    collect_round(std::int32_t row, std::uint32_t columns, std::span<const TokenId>,
                   const TokenLogprobOptions&) const;
    [[nodiscard]] TokenLogprobs collect_prompt(std::size_t index, TokenId,
                                               const TokenLogprobOptions&) const;
    [[nodiscard]] std::vector<float> collect_feature() const;

private:
    void enqueue(std::int32_t row, const Tensor& logits, const Tensor& sampled,
                  const Tensor* allowed, std::int32_t domain, const TokenLogprobOptions&,
                  void* destination, CudaExecutionView);
    DistributionReadoutLayout layout_;
    Tensor candidates_, round_, prompt_ids_;
    PinnedHostBuffer host_;
};

} // namespace ninfer::models::qwen
