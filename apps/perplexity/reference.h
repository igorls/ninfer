#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ninfer::perplexity {

// One independent sequence of a reference file (eval/kld/build_reference.py). Target i predicts
// tokens[i+1]; its top-K rows list the reference model's most likely tokens, most likely first.
struct ReferenceSequence {
    std::string id;
    std::string domain;
    std::vector<TokenId> tokens;
    std::vector<TokenId> top_tokens;    // (tokens-1) x top_k, target-major
    std::vector<float> top_logprobs;    // (tokens-1) x top_k, natural log
    std::vector<float> target_logprobs; // tokens-1
};

struct ReferenceCorpus {
    std::filesystem::path source;
    std::string corpus_id;
    std::string model;
    std::uint32_t top_k = 0;
    std::vector<ReferenceSequence> sequences;
};

// Reads a "ninfer-kld-reference-v1" file: b"NINFKLD1", uint64 header bytes, UTF-8 JSON header,
// then per header sequence int32 tokens[T], int32 top_ids[T-1][K], float32 top_logprobs[T-1][K],
// float32 target_logprobs[T-1], all little endian.
[[nodiscard]] ReferenceCorpus load_reference(const std::filesystem::path& path);

} // namespace ninfer::perplexity
