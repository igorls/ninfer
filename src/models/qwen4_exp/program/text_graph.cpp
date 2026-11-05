#include "models/qwen4_exp/program/text_graph.h"
#include "core/nvtx.h"
#include "models/common/moe/schedule.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {

void GraphSegment::prepare(CudaExecutionView cuda, const std::function<void()>& body) {
    definition.capture(cuda.stream(), body);
    executable.instantiate(definition);
    executable.upload(cuda.stream());
}

void TextGraph::prepare(execution::TextRun& run, const execution::Parameters& parameters,
                        moe::Schedule* experts, std::span<const std::uint32_t> bounds,
                        std::array<Tensor, 2> ple, execution::TextOutputs output,
                        std::int32_t columns, MemoryArena& workspace, DeviceContext& device) {
    const auto cuda    = device.execution_view();
    const auto& config = parameters.model.config().text;
    ple_layer_         = config.ple.layer;
    block_output_      = run.block_output();
    resident_          = experts == nullptr;
    for (const auto& layer : parameters.text.layers) banks_.push_back(layer.moe.bank);
    workspace.reset_peak();
    if (!resident_) stem_.prepare(cuda, [&] { run.stem(); });
    for (std::size_t slot = 0; slot < ple.size(); ++slot)
        ple_[slot].prepare(cuda, [&] { run.consume_ple(ple[slot]); });
    if (!resident_) finish_.prepare(cuda, [&] { run.finish(output); });
    const auto fixed_workspace = workspace.peak_used();
    profiles_.reserve(bounds.size());
    for (const auto bound : bounds) {
        run.set_visible_tokens(bound);
        workspace.reset_peak();
        Profile profile{bound, {}};
        if (resident_) {
            const auto layers = [&](std::uint32_t first, std::uint32_t last) {
                for (auto i = first; i < last; ++i) {
                    run.mixer(i);
                    run.resident_moe(i);
                    run.combine(i);
                }
            };
            profile.resident_prefix.capture(cuda.stream(), [&] {
                run.stem();
                layers(0, ple_layer_);
            });
            profile.resident_suffix.capture(cuda.stream(), [&] {
                layers(ple_layer_, static_cast<std::uint32_t>(banks_.size()));
                run.finish(output);
            });
            profile.workspace_bytes = std::max(fixed_workspace, workspace.peak_used());
            profiles_.push_back(std::move(profile));
            continue;
        }
        profile.layers.reserve(banks_.size());
        for (std::size_t i = 0; i < banks_.size(); ++i) {
            const auto* shared = profiles_.empty() ? nullptr : &profiles_.front().layers[i];
            if (shared &&
                std::holds_alternative<execution::GdnParameters>(parameters.text.layers[i].mixer)) {
                profile.layers.push_back(*shared);
                continue;
            }
            profile.layers.push_back(experts->capture_layer(
                    banks_[i], cuda, columns, [&] { run.mixer(i); },
                    [&](const moe::GpuCall& call) { run.gpu_moe(call); }, [&] { run.combine(i); },
                    shared ? *shared : nullptr));
        }
        profile.workspace_bytes = std::max(fixed_workspace, workspace.peak_used());
        if (!profiles_.empty())
            profile.workspace_bytes =
                std::max(profile.workspace_bytes, profiles_.front().workspace_bytes);
        profiles_.push_back(std::move(profile));
    }
    if (resident_) {
        resident_prefix_.instantiate(profiles_.front().resident_prefix);
        resident_suffix_.instantiate(profiles_.front().resident_suffix);
    } else {
        layers_.reserve(banks_.size());
        for (const auto& layer : profiles_.front().layers) layers_.push_back(layer->instantiate());
    }
    // Exercise every update before accepting requests, including the return to the first tier.
    for (const auto bound : bounds) {
        select(bound);
        if (resident_) {
            resident_prefix_.upload(cuda.stream());
            resident_suffix_.upload(cuda.stream());
        } else {
            for (auto& layer : layers_) layer->upload(cuda.stream());
        }
        device.synchronize();
    }
    select(bounds.front());
}

void TextGraph::select(std::uint32_t visible) {
    const auto it = std::lower_bound(profiles_.begin(), profiles_.end(), visible,
                                     [](const auto& profile, auto n) { return profile.bound < n; });
    if (it == profiles_.end()) throw std::logic_error("Qwen4Exp graph context exceeds capacity");
    const auto next = static_cast<std::size_t>(it - profiles_.begin());
    if (next == installed_) return;
    if (resident_) {
        resident_prefix_.update(it->resident_prefix);
        resident_suffix_.update(it->resident_suffix);
    } else {
        for (std::size_t i = 0; i < layers_.size(); ++i)
            if (it->layers[i] != profiles_[installed_].layers[i]) layers_[i]->update(*it->layers[i]);
    }
    installed_ = next;
}

void TextGraph::traverse(std::uint32_t begin, std::uint32_t end, CudaExecutionView cuda) {
    for (auto i = begin; i < end; ++i) {
        nvtx::ScopedRange range(nvtx::Name::TextLayer, nvtx::Category::Runtime, i);
        layers_[i]->replay(block_output_, cuda);
    }
}

void TextGraph::prefix(CudaExecutionView cuda) {
    if (resident_) {
        resident_prefix_.launch(cuda.stream());
        return;
    }
    stem_.executable.launch(cuda.stream());
    traverse(0, ple_layer_, cuda);
}

void TextGraph::consume_ple(std::uint32_t slot, CudaExecutionView cuda) {
    if (slot >= 2) throw std::logic_error("invalid PLE graph slot");
    ple_[slot].executable.launch(cuda.stream());
}

void TextGraph::suffix(CudaExecutionView cuda) {
    if (resident_) {
        resident_suffix_.launch(cuda.stream());
        return;
    }
    traverse(ple_layer_, banks_.size(), cuda);
    finish_.executable.launch(cuda.stream());
}

} // namespace ninfer::models::qwen4_exp
