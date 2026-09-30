#pragma once

#include "models/qwen3_5/frontend/tokenizer.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

namespace ninfer::models::qwen3_5 {

// Byte views borrow the model's Host backing. Tokenizer owns its decoded vocabulary and tables;
// loading and request preparation share this one immutable interpretation.
struct FrontendResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view chat_template_jinja;
    std::string_view generation_config_json;
    std::string_view preprocessor_config_json;
    std::string_view video_preprocessor_config_json;
    std::shared_ptr<const frontend::Tokenizer> tokenizer;
    std::uint32_t public_token_count = 0;
};

// The model facts a package validates its frontend resources against; any architecture whose
// prompts use this Frontend fills it from its own config.
struct FrontendGeometry {
    struct VisionPatch {
        std::uint32_t patch_size          = 0;
        std::uint32_t temporal_patch_size = 0;
        std::uint32_t spatial_merge_size  = 0;
    };

    std::uint32_t embedding_rows = 0;
    std::optional<VisionPatch> vision;
    std::optional<std::uint32_t> selector_top_k;
};

void parse_resources(FrontendResources& resources, const FrontendGeometry& geometry);

} // namespace ninfer::models::qwen3_5
