#pragma once

#include "core/decode_graph.h"
#include "core/tensor.h"

#include <cstdint>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

// Owns one stream capture that `flash_next_moe` splits after each layer's route.
// The route kernel is in the segment being sealed; the expert kernels launch into
// the segment begun by split. Replay publishes the pointer table between those
// segments, because a host function inside the capture cannot copy weights.
class ExpertGatherCapture {
public:
    static ExpertGatherCapture*& current() noexcept;

    ExpertGatherCapture(cudaStream_t stream, std::vector<ninfer::DecodeGraphDefinition>* segments);
    ~ExpertGatherCapture();

    ExpertGatherCapture(const ExpertGatherCapture&)            = delete;
    ExpertGatherCapture& operator=(const ExpertGatherCapture&) = delete;

    void begin();
    void split(const ninfer::Tensor& routed_ids);
    void finish();

    [[nodiscard]] const std::int32_t* routed_ids() const noexcept { return routed_ids_; }
    [[nodiscard]] std::int32_t routed_id_count() const noexcept { return routed_id_count_; }

private:
    void seal_segment();
    void abort() noexcept;

    cudaStream_t stream_ = nullptr;
    std::vector<ninfer::DecodeGraphDefinition>* segments_ = nullptr;
    const std::int32_t* routed_ids_                       = nullptr;
    std::int32_t routed_id_count_                         = 0;
    bool capturing_                                       = false;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
