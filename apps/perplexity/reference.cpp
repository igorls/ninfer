#include "reference.h"

#include <nlohmann/json.hpp>

#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string_view>

namespace ninfer::perplexity {
namespace {

constexpr std::string_view kMagic = "NINFKLD1";

template <class T>
void read_exact(std::ifstream& input, std::vector<T>& values, std::size_t count,
                const std::string& label) {
    values.resize(count);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(count * sizeof(T)));
    if (!input) { throw std::runtime_error("reference file is truncated in " + label); }
}

} // namespace

ReferenceCorpus load_reference(const std::filesystem::path& path) {
    static_assert(std::endian::native == std::endian::little,
                  "reference files are little endian");
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot open reference file: " + path.string()); }
    std::array<char, 8> magic{};
    std::uint64_t header_bytes = 0;
    input.read(magic.data(), magic.size());
    input.read(reinterpret_cast<char*>(&header_bytes), sizeof(header_bytes));
    if (!input || std::string_view(magic.data(), magic.size()) != kMagic) {
        throw std::runtime_error("not a NInfer KL reference file: " + path.string());
    }
    if (header_bytes == 0 || header_bytes > (1ULL << 30)) {
        throw std::runtime_error("reference header size is invalid");
    }
    std::string text(static_cast<std::size_t>(header_bytes), '\0');
    input.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!input) { throw std::runtime_error("reference header is truncated"); }
    const nlohmann::json header = nlohmann::json::parse(text);
    if (header.at("format").get<std::string>() != "ninfer-kld-reference-v1") {
        throw std::runtime_error("unsupported reference format");
    }

    ReferenceCorpus out;
    out.source    = path;
    out.corpus_id = header.at("corpus_id").get<std::string>();
    out.model     = header.at("model").dump();
    out.top_k     = header.at("top_k").get<std::uint32_t>();
    if (out.top_k == 0) { throw std::runtime_error("reference top_k must be positive"); }
    for (const nlohmann::json& item : header.at("sequences")) {
        ReferenceSequence sequence;
        sequence.id                = item.at("id").get<std::string>();
        sequence.domain            = item.at("domain").get<std::string>();
        const auto tokens          = item.at("tokens").get<std::size_t>();
        if (tokens < 2) { throw std::runtime_error(sequence.id + ": fewer than two tokens"); }
        const std::size_t targets  = tokens - 1;
        read_exact(input, sequence.tokens, tokens, sequence.id + " tokens");
        read_exact(input, sequence.top_tokens, targets * out.top_k, sequence.id + " top tokens");
        read_exact(input, sequence.top_logprobs, targets * out.top_k,
                   sequence.id + " top log probabilities");
        read_exact(input, sequence.target_logprobs, targets, sequence.id + " target log probabilities");
        out.sequences.push_back(std::move(sequence));
    }
    if (input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("reference file has trailing bytes after its sequences");
    }
    if (out.sequences.empty()) { throw std::runtime_error("reference file has no sequences"); }
    return out;
}

} // namespace ninfer::perplexity
