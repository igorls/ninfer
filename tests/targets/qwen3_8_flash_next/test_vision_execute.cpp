#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/vision_adapter.h"
#include "targets/qwen3_8_flash_next/impl/text_decode.h"
#include "targets/qwen3_8_flash_next/impl/text_executor.h"
#include <ninfer/targets/qwen3_vision/vision.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::targets::qwen3_8_flash_next::detail;
using namespace ninfer::targets::qwen3_vision;

Weight make_bf16_weight(const void* data, std::int32_t rows, std::int32_t cols) {
    Weight w{};
    w.payload       = data;
    w.payload_bytes = static_cast<std::size_t>(rows) * cols * sizeof(std::uint16_t);
    w.qdata         = data;
    w.qtype         = QType::BF16_CTRL;
    w.layout        = QuantLayout::Contiguous;
    w.shape[0]      = rows;
    w.shape[1]      = cols;
    w.shape[2]      = 1;
    w.shape[3]      = 1;
    w.padded_shape[0] = rows;
    w.padded_shape[1] = cols;
    w.n             = rows;
    w.k             = cols;
    w.ndim          = 2;
    return w;
}

int test_vision_adapter() {
    VisionModelView v{};
    std::uint16_t dummy = 0;
    v.patch_embedding = make_bf16_weight(&dummy, 1152, 1536);
    v.patch_embedding_bias = Tensor(&dummy, DType::BF16, {1152});
    v.position_embedding = Tensor(&dummy, DType::BF16, {2304, 1152});

    for (std::size_t i = 0; i < v.layers.size(); ++i) {
        auto& l = v.layers[i];
        l.qkv = make_bf16_weight(&dummy, 3456, 1152);
        l.qkv_bias = Tensor(&dummy, DType::BF16, {3456});
        l.output = make_bf16_weight(&dummy, 1152, 1152);
        l.output_bias = Tensor(&dummy, DType::BF16, {1152});
        l.fc1 = make_bf16_weight(&dummy, 4304, 1152);
        l.fc1_bias = Tensor(&dummy, DType::BF16, {4304});
        l.fc2 = make_bf16_weight(&dummy, 1152, 4304);
        l.fc2_bias = Tensor(&dummy, DType::BF16, {1152});
        l.norm1_weight = Tensor(&dummy, DType::BF16, {1152});
        l.norm1_bias = Tensor(&dummy, DType::BF16, {1152});
        l.norm2_weight = Tensor(&dummy, DType::BF16, {1152});
        l.norm2_bias = Tensor(&dummy, DType::BF16, {1152});
    }

    v.merger_fc1 = make_bf16_weight(&dummy, 4608, 4608);
    v.merger_fc1_bias = Tensor(&dummy, DType::BF16, {4608});
    v.merger_fc2 = make_bf16_weight(&dummy, 2560, 4608);
    v.merger_fc2_bias = Tensor(&dummy, DType::BF16, {2560});
    v.merger_norm_weight = Tensor(&dummy, DType::BF16, {1152});
    v.merger_norm_bias = Tensor(&dummy, DType::BF16, {1152});

    const auto adapted = adapt_vision_weights(v);
    if (adapted.output_hidden != 2560) {
        std::cerr << "adapted.output_hidden != 2560\n";
        return 1;
    }
    if (adapted.patch_embed == nullptr || adapted.position_embed == nullptr ||
        adapted.merger.fc2 == nullptr) {
        std::cerr << "adapted weight pointer is null\n";
        return 1;
    }

    std::cout << "PASS: test_vision_adapter\n";
    return 0;
}

} // namespace

int main() {
    std::cout << "Starting Flash-Next Vision Execute Tests...\n";
    try {
        if (test_vision_adapter() != 0) return 1;
    } catch (const std::exception& ex) {
        std::cerr << "Fatal exception: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "OK Flash-Next Vision Execute Tests\n";
    return 0;
}
