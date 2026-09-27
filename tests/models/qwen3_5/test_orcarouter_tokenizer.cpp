#include "models/qwen3_5/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

// The OrcaRouter Qwen3.8-27B export stores the letters-only word split in tokenizer.json and keeps
// its added-token definitions only there. The fixture holds token ids and decodes produced by the
// Hugging Face `tokenizers` Rust library from that stored tokenizer.json, including combining
// marks, multilingual text, special tokens parsed and literal, and a long input.
namespace fi = ninfer::models::qwen3_5::frontend;

namespace {

std::string read(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("cannot read " + path.string()); }
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

} // namespace

int main() {
    const char* source = std::getenv("NINFER_ORCAROUTER_MODEL_DIR");
    if (source == nullptr || *source == '\0') {
        std::cout << "SKIP: NINFER_ORCAROUTER_MODEL_DIR is not set\n";
        return 77;
    }
    try {
        const std::filesystem::path dir(source);
        const std::string tokenizer_json = read(dir / "tokenizer.json");
        const std::string config         = read(dir / "tokenizer_config.json");
        const std::string generation     = read(dir / "generation_config.json");
        const fi::Tokenizer tokenizer({tokenizer_json, config, generation});
        const auto fixture = nlohmann::json::parse(
            read(std::filesystem::path(NINFER_SOURCE_DIR) /
                 "tests/fixtures/qwen3_8_27b_orcarouter_tokenizer.json"));
        for (const auto& item : fixture.at("cases")) {
            const auto text = item.at("text").get<std::string>();
            const auto ids  = item.at("input_ids").get<std::vector<int>>();
            const fi::EncodeOptions options{.parse_added_tokens =
                                                item.at("parse_added_tokens").get<bool>()};
            const std::array<std::size_t, 2> boundaries{0, text.size()};
            const auto result = tokenizer.encode_with_boundaries(text, boundaries, options);
            if (result.input_ids != ids || result.boundaries.back().exact_frontier != ids.size() ||
                tokenizer.decode(ids) != item.at("decoded").get<std::string>() ||
                tokenizer.decode(ids, {.skip_special_tokens = true}) !=
                    item.at("decoded_skip_special_tokens").get<std::string>()) {
                std::cerr << "reference mismatch: " << item.at("name") << '\n';
                return 1;
            }
            const auto limited = tokenizer.encode(
                text, {.parse_added_tokens = options.parse_added_tokens, .max_tokens = 32});
            if (limited != std::vector<int>(ids.begin(),
                                            ids.begin() + std::min<std::size_t>(32, ids.size()))) {
                std::cerr << "bounded reference mismatch: " << item.at("name") << '\n';
                return 1;
            }
        }
        std::cout << fixture.at("cases").size() << " independent tokenizer cases passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
