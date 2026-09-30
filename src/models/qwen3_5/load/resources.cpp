#include "models/qwen3_5/load/bindings.h"
#include "models/qwen3_5/frontend/resources.h"

namespace ninfer::models::qwen3_5::loading {

FrontendResources bind_resources(artifact::Binder& binder, const Config& config) {
    const auto resource = [&](std::string_view component, std::string_view role) {
        const auto bytes = binder.host_object(binder.resource(component, role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    FrontendResources out;
    out.tokenizer_json         = resource("text", "tokenizer.json");
    out.tokenizer_config_json  = resource("text", "tokenizer_config.json");
    out.chat_template_jinja    = resource("text", "chat_template.jinja");
    out.generation_config_json = resource("text", "generation_config.json");
    if (config.vision) {
        out.preprocessor_config_json       = resource("vision", "preprocessor_config.json");
        out.video_preprocessor_config_json = resource("vision", "video_preprocessor_config.json");
    }
    FrontendGeometry geometry{.embedding_rows = config.text.vocab_size};
    if (config.vision) {
        geometry.vision = FrontendGeometry::VisionPatch{config.vision->patch_size,
                                                        config.vision->temporal_patch_size,
                                                        config.vision->spatial_merge_size};
    }
    if (config.draft && config.draft->dflash2) {
        geometry.selector_top_k = config.draft->dflash2->selector_top_k;
    }
    parse_resources(out, geometry);
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
