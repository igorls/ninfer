#pragma once

// llama.cpp server properties (GET /props). llama.cpp's web UI reads them at startup to learn the
// served model, its context size, parallel slots, input modalities and chat template.

#include <cstdint>
#include <string>

namespace ninfer::serve {

struct LlamaCppProps {
    std::string model_alias;       // public model id, as /v1/models and responses report it
    std::string model_path;        // artifact file name
    std::uint32_t n_ctx       = 0; // per-request context ceiling
    std::uint32_t total_slots = 0; // effective Engine concurrency
    bool vision               = false;
    std::string chat_template;
};

std::string make_llamacpp_props(const LlamaCppProps& props);

} // namespace ninfer::serve
