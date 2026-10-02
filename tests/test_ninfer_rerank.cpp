#include "serve/http_server.h"
#include "serve/ninfer_rerank.h"
#include "serve/typesafe_systemone.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;
using namespace ninfer::serve;

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

bool close_to(double left, double right, double eps = 1e-12) {
    return std::fabs(left - right) < eps;
}

RerankSettings settings_for(std::size_t max_documents = kDefaultRerankMaxDocuments) {
    RerankSettings settings;
    settings.served_model_id = "qwen3.8-27b";
    settings.max_documents   = max_documents;
    return settings;
}

ApiError api_error(const Json& body, const RerankSettings& settings = settings_for()) {
    try {
        (void)parse_rerank_request(body, settings);
    } catch (const ApiException& exception) { return exception.error(); }
    ApiError error;
    error.status  = 0;
    error.message = "no exception";
    return error;
}

void collect_keys(const Json& value, std::vector<std::string>& keys) {
    if (value.is_object()) {
        for (auto field = value.begin(); field != value.end(); ++field) {
            keys.push_back(field.key());
            collect_keys(*field, keys);
        }
    } else if (value.is_array()) {
        for (const Json& item : value) { collect_keys(item, keys); }
    }
}

bool has_key(const Json& value, const std::string& key) {
    std::vector<std::string> keys;
    collect_keys(value, keys);
    for (const std::string& found : keys) {
        if (found == key) { return true; }
    }
    return false;
}

} // namespace

int main() {
    int failures = 0;
    const RerankSettings settings = settings_for();

    {
        const Json body{{"query", "where is the invoice"},
                        {"documents", Json::array({"drawer copy", Json{{"text", "weather"},
                                                                        {"source", "ignored"}}})},
                        {"vendor_extension", true}};
        const RerankRequest request = parse_rerank_request(body, settings);
        failures += check(request.query == "where is the invoice" && request.documents.size() == 2 &&
                              request.documents[0].text == "drawer copy" &&
                              request.documents[1].text == "weather" && request.return_documents &&
                              request.top_n == 2,
                          "omitted return_documents defaults to true and top_n defaults to N");
    }

    {
        const RerankRequest request = parse_rerank_request(
            Json{{"model", "qwen3.8-27b"},
                 {"query", "q"},
                 {"documents", Json::array({"a", "b", "c"})},
                 {"top_n", 0},
                 {"return_documents", false}},
            settings);
        failures += check(!request.return_documents && request.top_n == 1,
                          "top_n below 1 clamps to 1");
        const RerankRequest high = parse_rerank_request(
            Json{{"model", kDefaultRerankModelId},
                 {"query", "q"},
                 {"documents", Json::array({"a", "b", "c"})},
                 {"top_n", 99},
                 {"return_documents", true}},
            settings);
        failures += check(high.top_n == 3 && high.return_documents, "top_n above N clamps to N");
        const RerankRequest negative = parse_rerank_request(
            Json{{"query", "q"}, {"documents", Json::array({"a", "b"})}, {"top_n", -3}}, settings);
        failures += check(negative.top_n == 1, "negative top_n clamps to 1");
    }

    {
        const ApiError empty_query = api_error(Json{{"query", ""}, {"documents", Json::array({"a"})}});
        const ApiError missing_query = api_error(Json{{"documents", Json::array({"a"})}});
        const ApiError empty_model =
            api_error(Json{{"model", ""}, {"query", "q"}, {"documents", Json::array({"a"})}});
        const ApiError unknown_model = api_error(
            Json{{"model", "jina-reranker"}, {"query", "q"}, {"documents", Json::array({"a"})}});
        const ApiError empty_text =
            api_error(Json{{"query", "q"}, {"documents", Json::array({Json{{"text", ""}}})}});
        const ApiError bare_object =
            api_error(Json{{"query", "q"}, {"documents", Json::array({Json::object()})}});
        const ApiError too_many = api_error(
            Json{{"query", "q"}, {"documents", Json::array({"a", "b", "c"})}}, settings_for(2));
        const ApiError empty_docs =
            api_error(Json{{"query", "q"}, {"documents", Json::array()}});
        const ApiError not_object = api_error(Json::array());
        failures += check(empty_query.status == 400 && empty_query.param == "query" &&
                              missing_query.status == 400 && missing_query.param == "query",
                          "query is a required nonempty string");
        failures += check(empty_model.status == 400 && empty_model.param == "model",
                          "an empty model string was accepted");
        failures += check(unknown_model.status == 404 && unknown_model.code == "model_not_found" &&
                              unknown_model.param == "model",
                          "a model other than the served or advertised id was accepted");
        failures +=
            check(empty_text.status == 400 && empty_text.param == "documents[0]" &&
                      bare_object.status == 400 && bare_object.param == "documents[0]" &&
                      too_many.status == 400 && too_many.param == "documents" &&
                      empty_docs.status == 400 && not_object.status == 400,
                  "document bounds and empty text were not rejected");
        const ApiError bad_top = api_error(
            Json{{"query", "q"}, {"documents", Json::array({"a"})}, {"top_n", "2"}});
        const ApiError bad_flag = api_error(Json{{"query", "q"},
                                                 {"documents", Json::array({"a"})},
                                                 {"return_documents", "true"}});
        failures += check(bad_top.status == 400 && bad_top.param == "top_n" && bad_flag.status == 400 &&
                              bad_flag.param == "return_documents",
                          "non-integer top_n or non-boolean return_documents was accepted");
    }

    {
        const double weights_only_exact = rerank_relevance_score({1.0, 0.0, 0.0, 0.0}, settings);
        const double substitute         = rerank_relevance_score({0.0, 1.0, 0.0, 0.0}, settings);
        const double complement         = rerank_relevance_score({0.0, 0.0, 1.0, 0.0}, settings);
        const double irrelevant         = rerank_relevance_score({0.0, 0.0, 0.0, 1.0}, settings);
        const double mixed              = rerank_relevance_score({0.5, 0.5, 0.0, 0.0}, settings);
        failures += check(close_to(weights_only_exact, 1.0) && close_to(substitute, 0.6) &&
                              close_to(complement, 0.25) && close_to(irrelevant, 0.0) &&
                              close_to(mixed, 0.8),
                          "default relevance weights did not match 1, 0.6, 0.25, 0");
        RerankSettings custom = settings;
        custom.weight_exact = 2.0;
        custom.weight_irrelevant = -1.0;
        failures += check(close_to(rerank_relevance_score({0.5, 0.0, 0.0, 0.5}, custom), 0.5),
                          "configured weights did not replace the defaults");
        bool wrong_width = false;
        try {
            (void)rerank_relevance_score({1.0, 0.0}, settings);
        } catch (const std::invalid_argument&) { wrong_width = true; }
        failures += check(wrong_width, "a distribution that is not four classes was scored");
    }

    {
        const std::vector<RerankHit> ranked = rank_rerank_documents({0.1, 0.9, 0.9, 0.25}, 3);
        failures += check(ranked.size() == 3 && ranked[0].index == 1 && ranked[1].index == 2 &&
                              ranked[2].index == 3 && close_to(ranked[0].relevance_score, 0.9) &&
                              close_to(ranked[1].relevance_score, 0.9),
                          "ties did not keep the lower original index, or top_n did not truncate");
        const std::vector<RerankHit> all = rank_rerank_documents({0.1, 0.9, 0.9, 0.25}, 10);
        failures += check(all.size() == 4 && all[3].index == 0,
                          "top_n above the document count dropped a result");
    }

    {
        RerankRequest request;
        request.query = "where is the invoice";
        request.documents.push_back(RerankDocument{"alpha passage"});
        request.documents.push_back(RerankDocument{"beta passage"});
        request.return_documents = true;
        request.top_n            = 2;
        const SystemOneRequest choice = build_rerank_choice_request(request);
        failures += check(choice.state_text == request.query && choice.temperature == 1.0 &&
                              choice.questions.size() == 2 && choice.questions[0].id == "0" &&
                              choice.questions[1].id == "1" &&
                              choice.questions[0].type == SystemOneQuestionType::Choice,
                          "the internal batch is one Choice per document on the query");
        const SystemOneQuestion& first = choice.questions[0];
        failures += check(first.instructions ==
                                  "How relevant is this document to the query?\n\nDocument:\nalpha passage" &&
                              first.choice_options.size() == 4 && first.choice_options[0].key == "exact" &&
                              first.choice_options[1].key == "substitute" &&
                              first.choice_options[2].key == "complement" &&
                              first.choice_options[3].key == "irrelevant" &&
                              first.choice_options[0].label == "A" &&
                              first.choice_options[3].label == "D" &&
                              first.choice_options[0].description.find("exact match") != std::string::npos,
                          "Choice criteria were not positional exact, substitute, complement, irrelevant");
        const std::string prompt = build_question_prompt(first);
        const auto exact_at      = prompt.find("exact match");
        const auto substitute_at = prompt.find("substitute answer");
        const auto complement_at = prompt.find("complementary");
        const auto irrelevant_at = prompt.find("unrelated");
        failures += check(exact_at != std::string::npos && exact_at < substitute_at &&
                              substitute_at < complement_at && complement_at < irrelevant_at &&
                              prompt.find("alpha passage") != std::string::npos &&
                              prompt.find("beta passage") == std::string::npos,
                          "the question prompt lost criteria order or leaked another document");
        const Json messages = build_systemone_messages(choice);
        failures += check(messages.is_array() && !messages.empty() &&
                              messages[0].at("content").get<std::string>().find(request.query) !=
                                  std::string::npos,
                          "the shared state is not the query");

        const std::vector<RerankHit> hits{{.index = 1, .relevance_score = 0.8},
                                          {.index = 0, .relevance_score = 0.2}};
        const Json response = make_rerank_response(request, hits, kDefaultRerankModelId, 1842);
        failures += check(response.at("model") == kDefaultRerankModelId &&
                              response.at("usage").at("total_tokens") == 1842 &&
                              response.at("results").size() == 2 &&
                              response.at("results")[0].at("index") == 1 &&
                              close_to(response.at("results")[0].at("relevance_score").get<double>(), 0.8) &&
                              response.at("results")[0].at("document").at("text") == "beta passage",
                          "the response did not keep advertised id, original index, document text and usage");
        const std::string dumped = response.dump();
        failures += check(dumped.find("\"model\"") < dumped.find("\"results\"") &&
                              dumped.find("\"results\"") < dumped.find("\"usage\""),
                          "response field order changed");
        for (const char* label : {"exact", "substitute", "complement", "irrelevant"}) {
            failures += check(!has_key(response, label),
                              std::string("public response exposed the Choice label ") + label);
        }
        request.return_documents = false;
        const Json hidden = make_rerank_response(request, hits, kDefaultRerankModelId, 4);
        failures += check(!hidden.at("results")[0].contains("document") &&
                              !has_key(hidden, "document"),
                          "return_documents false still emitted document");
    }

    {
        const Json models = make_service_models_list("qwen3.8-27b", kDefaultRerankModelId, 100,
                                                     100, 8192);
        failures += check(models.at("data").size() == 2 && models.at("data")[0].at("id") == "qwen3.8-27b" &&
                              models.at("data")[1].at("id") == kDefaultRerankModelId &&
                              models.at("data")[1].at("object") == "model" &&
                              models.at("data")[1].at("owned_by") == "ninfer" &&
                              models.at("data")[1].at("max_model_len") == 8192,
                          "GET /v1/models data did not advertise the rerank id");
        bool rerank_in_typesafe = false;
        for (const Json& entry : models.at("models")) {
            if (entry.at("name") == kDefaultRerankModelId) { rerank_in_typesafe = true; }
        }
        failures += check(!rerank_in_typesafe && models.at("models").size() == 3,
                          "the rerank id was added to the TypeSafe model list");
        const Json same = make_service_models_list("qwen3.8-27b", "qwen3.8-27b", 100, 100, 8192);
        failures += check(same.at("data").size() == 1 && same.at("data")[0].at("id") == "qwen3.8-27b",
                          "a rerank id equal to the served id was duplicated");
    }

    {
        httplib::Response missing;
        write_authentication_failure("/v1/rerank", missing, false);
        const Json missing_body = Json::parse(missing.body);
        httplib::Response wrong;
        write_authentication_failure("/v1/rerank", wrong, true);
        const Json wrong_body = Json::parse(wrong.body);
        failures += check(missing.status == 401 && wrong.status == 401 &&
                              missing_body.contains("error") && wrong_body.contains("error") &&
                              missing_body.at("error").at("code") == "invalid_api_key" &&
                              wrong_body.at("error").at("type") == "invalid_request_error" &&
                              !missing_body.contains("detail") && !wrong_body.contains("detail"),
                          "rerank auth was not an OpenAI 401 for both a missing and a wrong key");

        httplib::Response systemone_missing;
        write_authentication_failure("/v1/systemone", systemone_missing, false);
        httplib::Response systemone_wrong;
        write_authentication_failure("/systemone", systemone_wrong, true);
        const Json systemone_missing_body = Json::parse(systemone_missing.body);
        failures += check(systemone_missing.status == 403 && systemone_wrong.status == 401 &&
                              systemone_missing_body.contains("detail"),
                          "System One auth envelope changed while adding rerank");

        ServeOptions options;
        options.max_request_bytes = 1234;
        httplib::Request request;
        request.path = "/v1/rerank";
        httplib::Response limited;
        limited.status = 413;
        const auto handled = handle_unrendered_http_error(options, request, limited);
        const Json limited_body = Json::parse(limited.body);
        failures += check(handled == httplib::Server::HandlerResponse::Handled &&
                              limited.status == 413 && limited_body.contains("error") &&
                              limited_body.at("error").at("code") == "request_too_large" &&
                              !limited_body.contains("detail") && limited.has_header("x-request-id") &&
                              !limited.has_header("x-typesafe-request-id"),
                          "a rerank payload-limit error used the TypeSafe envelope");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
