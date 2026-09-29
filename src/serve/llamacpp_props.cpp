#include "serve/llamacpp_props.h"

#include <nlohmann/json.hpp>

namespace ninfer::serve {

std::string make_llamacpp_props(const LlamaCppProps& props) {
    // Only values the loaded Engine determines are reported. llama.cpp's sampling `params`,
    // special tokens and build_info are omitted: NInfer's sampling defaults depend on the
    // request's thinking mode, and the UI treats every omitted field as unknown.
    const nlohmann::ordered_json payload = {
        {"default_generation_settings", {{"n_ctx", props.n_ctx}}},
        {"total_slots", props.total_slots},
        {"model_alias", props.model_alias},
        {"model_path", props.model_path},
        {"role", "model"},
        // Audio and llama.cpp's input_video content part have no NInfer request mapping.
        {"modalities", {{"vision", props.vision}, {"audio", false}, {"video", false}}},
        {"chat_template", props.chat_template},
    };
    return payload.dump();
}

} // namespace ninfer::serve
