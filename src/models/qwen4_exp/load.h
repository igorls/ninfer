#pragma once

#include "models/qwen4_exp/model.h"

#include <filesystem>
#include <memory>

namespace ninfer::artifact {
class Reader;
} // namespace ninfer::artifact

namespace ninfer::models::qwen4_exp {

// Cold load plan borrows its Reader until materialization. Every artifact Binding and Use of the
// selected components is resolved here; a missing or unconsumed one fails the plan.
class LoadPlan {
public:
    ~LoadPlan();
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] const ModelWeights& weights() const;
    [[nodiscard]] const FrontendResources& resources() const;
    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;
    [[nodiscard]] std::size_t binding_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit LoadPlan(std::unique_ptr<Impl> impl);
    friend LoadPlan plan_load(const artifact::Reader&, LoadOptions);
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
};

[[nodiscard]] LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options = {});
[[nodiscard]] std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                                       const StartupObserver* observer = nullptr);

} // namespace ninfer::models::qwen4_exp
