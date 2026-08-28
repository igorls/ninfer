#pragma once

#include "artifact/materializer.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    std::array<std::vector<std::byte>, 6> frontend;
    TextModelView text;
    std::optional<VisionModelView> vision;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
