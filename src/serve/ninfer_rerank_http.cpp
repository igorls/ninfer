#include "serve/ninfer_rerank.h"

#include "serve/http_server.h"
#include "serve/http_transport.h"

#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::ordered_json;

RerankSettings rerank_settings(const ServeOptions& options, const std::string& served_model_id) {
    return RerankSettings{
        .served_model_id     = served_model_id,
        .advertised_model_id = options.rerank_model_id,
        .max_documents       = options.rerank_max_documents,
        .weight_exact        = options.rerank_weight_exact,
        .weight_substitute   = options.rerank_weight_substitute,
        .weight_complement   = options.rerank_weight_complement,
        .weight_irrelevant   = options.rerank_weight_irrelevant,
    };
}

void write_internal(httplib::Response& response, const char* what) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = what;
    write_openai_error(response, error);
}

} // namespace

void HttpServer::handle_rerank(const httplib::Request& req, httplib::Response& res) {
    const RerankSettings settings = rerank_settings(options_, public_model_id_);
    RerankRequest request;
    try {
        request = parse_rerank_request(parse_json_body(req), settings);
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }

    SystemOneExecution execution;
    try {
        execution = execute_systemone(build_rerank_choice_request(request), req, "ninfer_rerank");
        if (execution.answers.size() != request.documents.size()) {
            throw std::runtime_error("rerank produced a different number of scores than documents");
        }
        std::vector<double> scores;
        scores.reserve(execution.answers.size());
        for (const SystemOneAnswer& answer : execution.answers) {
            scores.push_back(rerank_relevance_score(answer.probabilities, settings));
        }
        const Json payload = make_rerank_response(request, rank_rerank_documents(scores, request.top_n),
                                                  settings.advertised_model_id,
                                                  execution.usage.input_tokens);
        if (execution.lifetime) {
            set_owned_json_content(res, payload.dump(), execution.lifetime);
        } else {
            res.set_content(payload.dump(), "application/json");
        }
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
    } catch (const std::exception& exception) {
        write_internal(res, exception.what());
    }
}

} // namespace ninfer::serve
