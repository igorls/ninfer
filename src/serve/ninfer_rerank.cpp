#include "serve/ninfer_rerank.h"

#include "serve/openai_common.h"
#include "serve/request_validation.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::serve {
namespace {

using Json = nlohmann::ordered_json;

constexpr std::size_t kRerankClasses = 4;

const char* kClassKeys[kRerankClasses] = {"exact", "substitute", "complement", "irrelevant"};

const char* kClassDescriptions[kRerankClasses] = {
    "Document answers the query directly / is an exact match",
    "Document is a useful near-match or substitute answer",
    "Document is related / complementary but not sufficient alone",
    "Document is unrelated to the query",
};

std::size_t clamp_top_n(int requested, std::size_t count) {
    if (requested < 1) { return 1; }
    const auto value = static_cast<std::size_t>(requested);
    return value > count ? count : value;
}

std::string document_text(const Json& item, const std::string& param) {
    if (item.is_string()) {
        std::string text = item.get<std::string>();
        if (text.empty()) { bad_request("document text must be nonempty", param); }
        return text;
    }
    if (!item.is_object()) {
        bad_request("each document must be a string or an object with text", param);
    }
    const auto found = item.find("text");
    if (found == item.end() || !found->is_string() || found->get_ref<const std::string&>().empty()) {
        bad_request("document text must be a nonempty string", param);
    }
    return found->get<std::string>();
}

} // namespace

RerankRequest parse_rerank_request(const Json& body, const RerankSettings& settings) {
    if (!body.is_object()) { bad_request("request body must be an object"); }
    if (settings.max_documents == 0) {
        bad_request("rerank is not configured to accept documents", "documents");
    }

    if (!body.contains("query") || !body.at("query").is_string() ||
        body.at("query").get_ref<const std::string&>().empty()) {
        bad_request("query must be a nonempty string", "query");
    }

    if (body.contains("model") && !body.at("model").is_null()) {
        if (!body.at("model").is_string()) { bad_request("model must be a string", "model"); }
        const std::string& model = body.at("model").get_ref<const std::string&>();
        if (model.empty()) { bad_request("model must be a nonempty string", "model"); }
        if (model != settings.advertised_model_id) {
            validate_openai_model(model, settings.served_model_id);
        }
    }

    if (!body.contains("documents") || !body.at("documents").is_array() ||
        body.at("documents").empty() || body.at("documents").size() > settings.max_documents) {
        bad_request("documents must be an array of 1 to " + std::to_string(settings.max_documents) +
                        " entries",
                    "documents");
    }

    RerankRequest request;
    request.query             = body.at("query").get<std::string>();
    request.return_documents  = optional_bool(body, "return_documents", true);
    request.documents.reserve(body.at("documents").size());
    std::size_t index = 0;
    for (const Json& item : body.at("documents")) {
        const std::string param = "documents[" + std::to_string(index) + "]";
        request.documents.push_back(RerankDocument{document_text(item, param)});
        ++index;
    }

    const std::optional<int> top_n = optional_int(body, "top_n");
    request.top_n = top_n ? clamp_top_n(*top_n, request.documents.size()) : request.documents.size();
    return request;
}

SystemOneRequest build_rerank_choice_request(const RerankRequest& request) {
    SystemOneRequest choice;
    choice.state_text  = request.query;
    choice.temperature = 1.0;
    choice.questions.reserve(request.documents.size());
    for (std::size_t index = 0; index < request.documents.size(); ++index) {
        SystemOneQuestion question;
        question.id   = std::to_string(index);
        question.type = SystemOneQuestionType::Choice;
        question.instructions = "How relevant is this document to the query?\n\nDocument:\n" +
                                request.documents[index].text;
        question.two_token_labels = false;
        question.choice_options.reserve(kRerankClasses);
        for (std::size_t option = 0; option < kRerankClasses; ++option) {
            question.choice_options.push_back(SystemOneChoiceOption{
                .key         = kClassKeys[option],
                .description = kClassDescriptions[option],
                .label       = choice_label_for_index(option, false),
            });
        }
        choice.questions.push_back(std::move(question));
    }
    return choice;
}

double rerank_relevance_score(const std::vector<double>& probabilities,
                              const RerankSettings& settings) {
    if (probabilities.size() != kRerankClasses) {
        throw std::invalid_argument("rerank expected four class probabilities");
    }
    const double weights[kRerankClasses] = {settings.weight_exact, settings.weight_substitute,
                                            settings.weight_complement, settings.weight_irrelevant};
    double score = 0.0;
    for (std::size_t index = 0; index < kRerankClasses; ++index) {
        score += weights[index] * probabilities[index];
    }
    return score;
}

std::vector<RerankHit> rank_rerank_documents(const std::vector<double>& scores, std::size_t top_n) {
    std::vector<RerankHit> hits;
    hits.reserve(scores.size());
    for (std::size_t index = 0; index < scores.size(); ++index) {
        hits.push_back(RerankHit{.index = index, .relevance_score = scores[index]});
    }
    std::stable_sort(hits.begin(), hits.end(), [](const RerankHit& left, const RerankHit& right) {
        if (left.relevance_score != right.relevance_score) {
            return left.relevance_score > right.relevance_score;
        }
        return left.index < right.index;
    });
    if (top_n < hits.size()) { hits.resize(top_n); }
    return hits;
}

nlohmann::ordered_json make_rerank_response(const RerankRequest& request,
                                            const std::vector<RerankHit>& hits,
                                            std::string_view advertised_model_id,
                                            std::int64_t total_tokens) {
    Json results = Json::array();
    for (const RerankHit& hit : hits) {
        Json result{{"index", hit.index}, {"relevance_score", hit.relevance_score}};
        if (request.return_documents) {
            result["document"] = Json{{"text", request.documents.at(hit.index).text}};
        }
        results.push_back(std::move(result));
    }
    return Json{{"model", std::string(advertised_model_id)},
                {"results", std::move(results)},
                {"usage", Json{{"total_tokens", total_tokens}}}};
}

nlohmann::ordered_json make_service_models_list(std::string_view served_model_id,
                                                std::string_view rerank_model_id,
                                                std::int64_t created,
                                                std::int64_t loaded_unix_seconds,
                                                std::uint32_t max_model_len) {
    Json payload = Json::parse(make_models_list(std::string(served_model_id), created, max_model_len));
    if (!rerank_model_id.empty() && rerank_model_id != served_model_id) {
        payload["data"].push_back(Json::parse(
            make_model_object(std::string(rerank_model_id), created, max_model_len)));
    }
    payload["models"] = systemone_model_entries(served_model_id, loaded_unix_seconds);
    return payload;
}

} // namespace ninfer::serve
