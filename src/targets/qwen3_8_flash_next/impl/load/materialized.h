#pragma once

#include "artifact/materializer.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"

#include <array>
#include <cstddef>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

class LoadedTextModelData {
public:
    LoadedTextModelData(BindingPlan plan, artifact::MaterializedArtifact materialized);

    LoadedTextModelData(const LoadedTextModelData&)            = delete;
    LoadedTextModelData& operator=(const LoadedTextModelData&) = delete;
    LoadedTextModelData(LoadedTextModelData&&)                 = delete;
    LoadedTextModelData& operator=(LoadedTextModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    std::array<std::vector<std::byte>, 6> frontend;
    TextModelView runtime;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
