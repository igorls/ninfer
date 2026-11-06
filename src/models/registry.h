#pragma once

#include <string_view>

namespace ninfer::artifact {
struct Directory;
}

namespace ninfer::models {

enum class Architecture { Qwen3_5, Qwen3_5Moe, Qwen4Exp };

[[nodiscard]] Architecture resolve_architecture(std::string_view architecture,
                                                std::string_view model_type);
// The architecture pair recorded by the artifact's text component config.
[[nodiscard]] Architecture resolve_architecture(const artifact::Directory& directory);
[[nodiscard]] std::string_view architecture_name(Architecture architecture) noexcept;

} // namespace ninfer::models
