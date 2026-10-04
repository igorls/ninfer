#include "targets/qwen3_8_flash_next/impl/expert_gather.h"

#include "core/device.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next::detail {

ExpertGatherCapture*& ExpertGatherCapture::current() noexcept {
    static thread_local ExpertGatherCapture* active = nullptr;
    return active;
}

ExpertGatherCapture::ExpertGatherCapture(cudaStream_t stream,
                                         std::vector<ninfer::DecodeGraphDefinition>* segments)
    : stream_(stream), segments_(segments) {
    if (stream_ == nullptr || segments_ == nullptr) {
        throw std::invalid_argument("Flash-Next expert gather capture needs a stream and segments");
    }
    if (current() != nullptr) {
        throw std::logic_error("Flash-Next expert gather capture is already active");
    }
    current() = this;
}

ExpertGatherCapture::~ExpertGatherCapture() {
    abort();
    if (current() == this) { current() = nullptr; }
}

void ExpertGatherCapture::begin() {
    if (capturing_) {
        throw std::logic_error("Flash-Next expert gather capture is already recording");
    }
    CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    capturing_ = true;
}

void ExpertGatherCapture::seal_segment() {
    if (!capturing_) {
        throw std::logic_error("Flash-Next expert gather capture has no open segment");
    }
    capturing_ = false;
    ninfer::DecodeGraphDefinition definition;
    definition.finish_capture(stream_);
    segments_->push_back(std::move(definition));
}

void ExpertGatherCapture::split(const ninfer::Tensor& routed_ids) {
    if (routed_ids.data == nullptr || routed_ids.numel() <= 0 || routed_ids.numel() > 0x7fffffffLL) {
        throw std::invalid_argument("Flash-Next expert gather capture saw an empty routing buffer");
    }
    const auto* pointer = static_cast<const std::int32_t*>(routed_ids.data);
    const auto count    = static_cast<std::int32_t>(routed_ids.numel());
    if (routed_ids_ == nullptr) {
        routed_ids_      = pointer;
        routed_id_count_ = count;
    } else if (pointer != routed_ids_ || count != routed_id_count_) {
        throw std::logic_error("Flash-Next expert cache capture saw a different routing-id buffer");
    }
    seal_segment();
    begin();
}

void ExpertGatherCapture::finish() { seal_segment(); }

void ExpertGatherCapture::abort() noexcept {
    if (!capturing_) { return; }
    capturing_ = false;
    cudaGraph_t discard = nullptr;
    if (cudaStreamEndCapture(stream_, &discard) != cudaSuccess) { return; }
    if (discard != nullptr) { cudaGraphDestroy(discard); }
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
