#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace {

std::filesystem::path artifact_path() {
    if (const char* value = std::getenv("NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS");
        value != nullptr && *value != '\0') {
        return value;
    }
    return "" /* real-artifact cases run only when NINFER_WEIGHTS is set explicitly */;
}

} // namespace

int main() {
    const std::filesystem::path path = artifact_path();
    if (!std::filesystem::is_regular_file(path)) {
        std::cerr << "skip: real Flash-Next artifact is required: " << path << '\n';
        return 77;
    }

    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    const auto plan = ninfer::targets::qwen3_8_flash_next::detail::bind_artifact(
        binder, {.vision = true, .mtp = true});

    const auto& materialization = plan.materialization;
    if (materialization.object_count != 1'566 || materialization.device_objects.size() != 1'429 ||
        materialization.mapped_tensor_objects.size() != 131 ||
        materialization.host_objects.size() != 6 ||
        materialization.device_capacity_bytes != 81'285'117'440ULL) {
        std::cerr << "unexpected Flash-Next load plan: objects=" << materialization.object_count
                  << " device=" << materialization.device_objects.size()
                  << " mapped=" << materialization.mapped_tensor_objects.size()
                  << " host=" << materialization.host_objects.size()
                  << " device_bytes=" << materialization.device_capacity_bytes << '\n';
        return 1;
    }

    const auto& bindings = plan.bindings;
    if (bindings.ple.shards.size() != 128 || bindings.text_layers.size() != 48 ||
        bindings.vision.layers.size() != 27 || !bindings.features.vision ||
        !bindings.features.mtp) {
        std::cerr << "Flash-Next semantic binding topology is incomplete\n";
        return 1;
    }
    return 0;
}
