#include "models/common/qwen/distribution_readout.h"
#include "runtime/contract/token_logprobs.h"

#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ninfer;

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void near(float actual, float expected) {
    require(actual == expected || (std::isfinite(actual) && std::isfinite(expected) &&
        std::abs(actual - expected) <= 2.0e-4F), "distribution differs from represented-input FP64 oracle");
}
void compare(const TokenLogprobs& actual, const TokenLogprobs& expected) {
    const auto token = [](const TokenLogprob& a, const TokenLogprob& b) {
        require(a.token == b.token, "distribution token order changed");
        near(a.logprob, b.logprob);
        near(a.raw_logprob, b.raw_logprob);
    };
    token(actual.sampled, expected.sampled);
    require(actual.top.size() == expected.top.size(), "top readout extent changed");
    require(actual.candidates.size() == expected.candidates.size(), "candidate readout extent changed");
    for (std::size_t i = 0; i < actual.top.size(); ++i) token(actual.top[i], expected.top[i]);
    for (std::size_t i = 0; i < actual.candidates.size(); ++i)
        token(actual.candidates[i], expected.candidates[i]);
}
}

int main() {
    try {
        DeviceContext device;
        LayoutBuilder builder;
        const auto readout_plan = models::qwen::plan_distribution_readout(builder, 4, 2, 5);
        const auto logits_plan = builder.add_tensor(DType::BF16, {12, 4}, 256, "test logits");
        const auto ids_plan = builder.add_tensor(DType::I32, {4}, 256, "test ids");
        const auto masks_plan = builder.add_tensor(DType::I32, {1, 4}, 256, "test masks");
        const auto hidden_plan = builder.add_tensor(DType::BF16, {5}, 256, "test hidden");
        DeviceBuffer backing(device.device(), builder.finish(256, "readout fixture"));
        models::qwen::DistributionReadout readout(backing.span(), readout_plan);
        auto logits = logits_plan.bind(backing.span());
        auto ids = ids_plan.bind(backing.span());
        auto masks = masks_plan.bind(backing.span());
        auto hidden = hidden_plan.bind(backing.span());
        std::vector<std::uint16_t> represented(48);
        for (int column = 0; column < 4; ++column)
            for (int row = 0; row < 12; ++row) {
                const float value = row < 8 ? float((row * 3 + column * 5) % 11 - 5) : 64.0F;
                represented[column * 12 + row] =
                    static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16U);
            }
        const std::vector<TokenId> sampled{0, 3, 5, 7};
        const std::vector<std::int32_t> allowed{0x55, 0xff, 0xaa, 0x89};
        const std::vector<std::uint16_t> feature{0x3f80, 0x4000, 0xbf80, 0, 0x3f00};
        const std::vector<TokenId> prompt_ids{3, 7};
        for (const auto& pair : {std::pair{logits.data, static_cast<const void*>(represented.data())},
                                 std::pair{ids.data, static_cast<const void*>(sampled.data())},
                                 std::pair{masks.data, static_cast<const void*>(allowed.data())},
                                 std::pair{hidden.data, static_cast<const void*>(feature.data())}}) {
            const auto bytes = pair.first == logits.data ? logits.bytes() :
                               pair.first == hidden.data ? hidden.bytes() : ids.bytes();
            CUDA_CHECK(cudaMemcpyAsync(pair.first, pair.second, bytes,
                cudaMemcpyHostToDevice, device.stream()));
        }
        TokenLogprobOptions options{.enabled = true, .top = 3, .candidates = {1, 3, 5}};
        TokenLogprobOptions prompt_options{.enabled = true, .candidates = {1, 3, 5}};
        readout.install(0, options, device.execution_view());
        readout.install(1, options, device.execution_view());
        readout.enqueue_round(0, logits, ids, &masks, 8, options, device.execution_view());
        auto first_logits = logits.slice(1, 0, 2);
        auto first_ids = ids.slice(0, 0, 2);
        readout.enqueue_round(1, first_logits, first_ids, nullptr, 8, options,
            device.execution_view());
        readout.install_prompt(prompt_ids, device.execution_view());
        readout.enqueue_prompt(1, logits.slice(1, 3, 1), 8, prompt_options, device.execution_view());
        readout.enqueue_prompt(0, logits.slice(1, 1, 1), 8, prompt_options, device.execution_view());
        readout.enqueue_feature(hidden, device.execution_view());
        device.synchronize();
        // Both row ownership and prompt copies survive subsequent reuse of the device scratch.
        const auto row0 = readout.collect_round(0, 4, std::span(sampled).first(3), options);
        const auto row1 = readout.collect_round(1, 2, std::span(sampled).first(2), options);
        for (std::size_t column = 0; column < row0.size(); ++column)
            compare(row0[column], runtime::compute_token_logprobs(
                std::span(represented).subspan(column * 12, 8),
                std::span(allowed).subspan(column, 1), sampled[column], options));
        for (std::size_t column = 0; column < row1.size(); ++column)
            compare(row1[column], runtime::compute_token_logprobs(
                std::span(represented).subspan(column * 12, 8), {}, sampled[column], options));
        for (std::size_t index = 0; index < 2; ++index) {
            const auto column = index == 0 ? 1 : 3;
            compare(readout.collect_prompt(index, prompt_ids[index], prompt_options),
                runtime::compute_token_logprobs(std::span(represented).subspan(column * 12, 8),
                    {}, prompt_ids[index], prompt_options));
        }
        const auto observed = readout.collect_feature();
        require(observed.size() == feature.size(), "hidden readout extent changed");
        for (std::size_t i = 0; i < feature.size(); ++i)
            require(observed[i] == runtime::bf16_to_float(feature[i]), "hidden row changed");
        bool rejected = false;
        try { readout.install(-1, options, device.execution_view()); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "invalid compact row was accepted");
        std::cout << "distribution readout rows, prompt reuse, masks, padded vocabulary and BF16 features passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
