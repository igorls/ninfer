#pragma once

#include "serve/serve_options.h"
#include "serve/typesafe_systemone.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// POST /v1/rerank. The public body is Jina-shaped. Scoring is an in-process System One Choice
// whose option labels never appear on this response.

struct RerankSettings {
    std::string served_model_id;
    std::string advertised_model_id = kDefaultRerankModelId;
    std::size_t max_documents       = kDefaultRerankMaxDocuments;
    double weight_exact             = kDefaultRerankWeightExact;
    double weight_substitute        = kDefaultRerankWeightSubstitute;
    double weight_complement        = kDefaultRerankWeightComplement;
    double weight_irrelevant        = kDefaultRerankWeightIrrelevant;
};

struct RerankDocument {
    std::string text;
};

struct RerankRequest {
    std::string query;
    std::vector<RerankDocument> documents;
    std::size_t top_n        = 0;
    bool return_documents    = true;
};

struct RerankHit {
    std::size_t index        = 0;
    double relevance_score   = 0.0;
};

// Throws ApiException. Unknown top-level fields are ignored. `return_documents` defaults to true.
// `top_n` defaults to the document count and is clamped to [1, N].
[[nodiscard]] RerankRequest parse_rerank_request(const nlohmann::ordered_json& body,
                                                 const RerankSettings& settings);

// One Choice question per document. Question ids are "0".."N-1". Option order is exact,
// substitute, complement, irrelevant, which is the order the relevance weights read.
[[nodiscard]] SystemOneRequest build_rerank_choice_request(const RerankRequest& request);

// `probabilities` is the Choice distribution in option order. The score is
// w_exact*P0 + w_substitute*P1 + w_complement*P2 + w_irrelevant*P3.
[[nodiscard]] double rerank_relevance_score(const std::vector<double>& probabilities,
                                            const RerankSettings& settings);

// Best-first by relevance_score descending. Equal scores keep the lower original index first.
// `top_n` above the score count keeps every document.
[[nodiscard]] std::vector<RerankHit> rank_rerank_documents(const std::vector<double>& scores,
                                                          std::size_t top_n);

[[nodiscard]] nlohmann::ordered_json make_rerank_response(const RerankRequest& request,
                                                         const std::vector<RerankHit>& hits,
                                                         std::string_view advertised_model_id,
                                                         std::int64_t total_tokens);

// OpenAI `data` plus TypeSafe `models`. The advertised rerank id is a second `data` entry when
// it differs from the served id. The TypeSafe list is unchanged.
[[nodiscard]] nlohmann::ordered_json make_service_models_list(std::string_view served_model_id,
                                                             std::string_view rerank_model_id,
                                                             std::int64_t created,
                                                             std::int64_t loaded_unix_seconds,
                                                             std::uint32_t max_model_len);

} // namespace ninfer::serve
