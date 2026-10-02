#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = ninfer::EnginePurpose::CausalScoring;
    options.max_context   = 2048;
    options.kv_cache      = ninfer::KvCacheStorage::Fp8E4M3Row256;
    ninfer::Engine engine(options);
    const auto& effective = engine.options();
    if (effective.max_concurrency != 1 || effective.prefill_chunk != 1024 ||
        effective.kv_capacity.mode != ninfer::KvCapacityMode::Explicit ||
        effective.kv_capacity.explicit_tokens != effective.max_context ||
        effective.context_cache.enabled ||
        effective.speculative.backend != ninfer::SpeculativeBackend::None ||
        effective.kv_cache != ninfer::KvCacheStorage::Fp8E4M3Row256) {
        std::cerr << "causal scoring options were not normalized correctly\n";
        return 1;
    }

    std::string text;
    const std::string paragraph =
        "NInfer scores each target token from the preceding hidden state. "
        "Every evaluation window owns fresh state and a fresh KV address space.\n";
    std::vector<ninfer::TokenId> tokens;
    while (tokens.size() < 1537) {
        text += paragraph;
        tokens = engine.tokenize_text(text);
    }
    tokens.resize(1537);

    const std::vector<float> all      = engine.score_tokens(tokens, 1);
    const std::vector<float> suffix   = engine.score_tokens(tokens, 513);
    const std::vector<float> repeated = engine.score_tokens(tokens, 513);
    if (all.size() != 1536 || suffix.size() != 1024 || repeated.size() != suffix.size()) {
        std::cerr << "causal scoring returned an invalid result shape\n";
        return 1;
    }
    for (const float value : all) {
        if (!std::isfinite(value) || value > 0.0F) {
            std::cerr << "causal scoring returned an invalid log probability\n";
            return 1;
        }
    }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        if (!std::isfinite(suffix[i]) || suffix[i] > 0.0F) {
            std::cerr << "causal scoring returned a non-finite logprob\n";
            return 1;
        }
        if (suffix[i] != repeated[i]) {
            std::cerr << "a repeated score window inherited prior State/KV\n";
            return 1;
        }
    }
    // Reference readout: per target the actual token, a fixed token and the same token again.
    constexpr std::uint32_t kReferences = 3;
    ninfer::CausalScoreReadout readout{.reference_count = kReferences, .reference_tokens = {}};
    for (std::size_t i = 513; i < tokens.size(); ++i) {
        readout.reference_tokens.insert(readout.reference_tokens.end(),
                                        {tokens[i], tokens[0], tokens[i]});
    }
    const ninfer::CausalScores scores = engine.score_tokens(tokens, 513, readout);
    if (scores.target_logprobs != suffix ||
        scores.reference_logprobs.size() != suffix.size() * kReferences ||
        scores.argmax_tokens.size() != suffix.size() ||
        scores.argmax_logprobs.size() != suffix.size()) {
        std::cerr << "the readout changed the target scores or returned an invalid shape\n";
        return 1;
    }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        const float* row = scores.reference_logprobs.data() + i * kReferences;
        if (row[0] != suffix[i] || row[2] != suffix[i]) {
            std::cerr << "a reference slot disagrees with the target score of the same token\n";
            return 1;
        }
        const float best = scores.argmax_logprobs[i];
        if (!std::isfinite(best) || best > 0.0F || best < row[0] || best < row[1]) {
            std::cerr << "the argmax log probability is not the column maximum\n";
            return 1;
        }
    }
    bool rejected = false;
    try {
        readout.reference_tokens.pop_back();
        (void)engine.score_tokens(tokens, 513, readout);
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) {
        std::cerr << "a readout with the wrong reference count was accepted\n";
        return 1;
    }
    // Hidden row readout: the capture leaves the scores untouched, is deterministic, and row i
    // is predictor position first_target - 1 + i.
    const ninfer::CausalScoreReadout capture{.capture_hidden_rows = true};
    const ninfer::CausalScores rows_all   = engine.score_tokens(tokens, 1, capture);
    const ninfer::CausalScores rows_again = engine.score_tokens(tokens, 1, capture);
    const ninfer::CausalScores rows_last =
        engine.score_tokens(tokens, static_cast<std::uint32_t>(tokens.size() - 1), capture);
    const std::size_t hidden = rows_all.hidden_size;
    if (hidden == 0 || rows_all.hidden_rows.size() != 1536 * hidden ||
        rows_again.hidden_size != hidden || rows_last.hidden_size != hidden ||
        rows_last.hidden_rows.size() != hidden || !scores.hidden_rows.empty() ||
        scores.hidden_size != 0) {
        std::cerr << "the hidden row readout returned an invalid shape\n";
        return 1;
    }
    if (rows_all.target_logprobs != all) {
        std::cerr << "capturing hidden rows changed the target scores\n";
        return 1;
    }
    if (rows_all.hidden_rows != rows_again.hidden_rows) {
        std::cerr << "the hidden row readout is not deterministic\n";
        return 1;
    }
    if (!std::equal(rows_last.hidden_rows.begin(), rows_last.hidden_rows.end(),
                    rows_all.hidden_rows.end() - static_cast<std::ptrdiff_t>(hidden))) {
        std::cerr << "the last hidden row does not match a single-target readout\n";
        return 1;
    }
    for (const std::uint16_t word : rows_all.hidden_rows) {
        if ((word & 0x7F80U) == 0x7F80U) {
            std::cerr << "a hidden row holds a non-finite BF16 value\n";
            return 1;
        }
    }
    // Causal prefix consistency is reported, not asserted: a shorter window may tile attention
    // differently and change summation order.
    std::vector<ninfer::TokenId> prefix(tokens.begin(), tokens.begin() + 1000);
    const ninfer::CausalScores rows_prefix = engine.score_tokens(prefix, 1, capture);
    float prefix_difference                = 0.0F;
    for (std::size_t i = 0; i < rows_prefix.hidden_rows.size(); ++i) {
        const auto value = [](std::uint16_t word) {
            float out;
            const std::uint32_t bits = static_cast<std::uint32_t>(word) << 16U;
            std::memcpy(&out, &bits, sizeof(out));
            return out;
        };
        prefix_difference = std::max(
            prefix_difference, std::abs(value(rows_prefix.hidden_rows[i]) - value(rows_all.hidden_rows[i])));
    }
    std::cout << "hidden rows: prefix window max |difference| " << prefix_difference << "\n";
    std::cout << "OK causal_score_real\n";
    return 0;
}
