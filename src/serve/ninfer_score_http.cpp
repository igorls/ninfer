#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// POST /v1/score: closed-set scoring of many isolated questions against one shared prefix.
//
// Every question becomes one or more ordinary chat requests made of the shared messages plus one
// user message, so questions never see each other. Two forms, chosen per question:
//   - token form: every candidate is one token; one request reads the distribution over the
//     candidates at the first generated position;
//   - text form: some candidate spans several tokens; one request per candidate renders it as
//     the continued final assistant turn, opened exactly as generation would open an answer, and
//     reads its tokens' log-probabilities at their prompt positions, whose sum is the
//     candidate's conditional log-probability.
// The first request carries an explicit shared-prefix boundary at the end of the shared messages
// and publishes it; every later one reads it, read-only in the context cache so that a large
// call does not displace other conversations' cached state.

namespace ninfer::serve {
namespace {

using Json = nlohmann::ordered_json;

constexpr std::size_t kMaximumScoreQuestions = 256;
constexpr double kLogprobFloor               = -9999.0;

struct ScoreBranch {
    std::size_t question = 0;
    std::optional<std::size_t> option; // text form: which candidate this branch scores
    std::vector<std::uint32_t> option_positions;
    OpenAIChatRequest request;
    std::optional<GenerationOutcome> outcome;
    std::optional<ApiError> error;
};

struct ScoreQuestion {
    std::string id;
    bool text_form = false;
    std::vector<std::string> option_texts; // text form, in request order
    std::vector<std::size_t> branches;     // indices into the branch list
};

[[noreturn]] void score_bad_request(std::string message, std::string param,
                                    std::string code = {}) {
    ApiError error;
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

// Marks the end of the shared messages as an explicit shared-prefix boundary.
Json with_prefix_boundary(Json messages) {
    Json& last = messages.back();
    if (last.contains("content") && last["content"].is_string()) {
        last["content"] = Json::array({Json{{"type", "text"}, {"text", last["content"]}}});
    }
    if (last.contains("content") && last["content"].is_array() && !last["content"].empty() &&
        last["content"].back().is_object()) {
        last["content"].back()["prompt_cache_breakpoint"] = Json{{"mode", "explicit"}};
    }
    return messages;
}

double finite_logprob(float value) {
    return std::isfinite(value) ? static_cast<double>(value) : kLogprobFloor;
}

std::string utf8_token(const std::string& bytes) {
    return Json::parse(Json(bytes).dump(-1, ' ', false, Json::error_handler_t::replace))
        .get<std::string>();
}

Json entry_json(const TokenLogprobEntry& entry) {
    Json bytes = Json::array();
    for (const char byte : entry.bytes) {
        bytes.push_back(static_cast<int>(static_cast<unsigned char>(byte)));
    }
    return Json{{"token", utf8_token(entry.bytes)},
                {"token_id", entry.token_id},
                {"logprob", finite_logprob(entry.logprob)},
                {"raw_logprob", finite_logprob(entry.raw_logprob)},
                {"bytes", std::move(bytes)}};
}

Json error_json(const ApiError& error) {
    return Json{{"message", error.message},
                {"type", error.type},
                {"param", error.param.empty() ? Json(nullptr) : Json(error.param)},
                {"code", error.code.empty() ? Json(nullptr) : Json(error.code)}};
}

// Entropy in nats and the top-two ratio of a distribution given as log-probabilities.
void distribution_stats(const std::vector<double>& logprobs, double& entropy, double& margin) {
    entropy = 0.0;
    std::vector<double> sorted;
    for (const double lp : logprobs) {
        if (lp <= kLogprobFloor) { continue; }
        const double p = std::exp(lp);
        entropy -= p * lp;
        sorted.push_back(lp);
    }
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    margin = sorted.size() >= 2 ? std::exp(sorted[0] - sorted[1])
                                : (sorted.empty() ? 0.0 : std::numeric_limits<double>::infinity());
}

Json finite_json(double value) {
    return std::isfinite(value) ? Json(value) : Json(nullptr);
}

} // namespace

void HttpServer::handle_score(const httplib::Request& req, httplib::Response& res) {
    std::vector<ScoreQuestion> questions;
    std::vector<ScoreBranch> branches;
    std::string model;
    try {
        const Json body = parse_json_body(req);
        if (!body.is_object()) { score_bad_request("request body must be an object", ""); }
        if (!body.contains("model") || !body.at("model").is_string()) {
            score_bad_request("model must be a string", "model");
        }
        model = body.at("model").get<std::string>();
        validate_openai_model(model, public_model_id_);
        if (!body.contains("messages") || !body.at("messages").is_array() ||
            body.at("messages").empty()) {
            score_bad_request("messages must be a nonempty array holding the shared prefix",
                              "messages");
        }
        if (!body.contains("questions") || !body.at("questions").is_array() ||
            body.at("questions").empty() || body.at("questions").size() > kMaximumScoreQuestions) {
            score_bad_request("questions must be an array of 1 to 256 entries", "questions");
        }
        if (body.contains("stream") && body.at("stream").is_boolean() &&
            body.at("stream").get<bool>()) {
            score_bad_request("score responses are not streamed", "stream");
        }
        const Json default_candidates =
            body.contains("candidates") ? body.at("candidates") : Json(nullptr);

        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        const auto parse_branch = [&](const ScoreQuestion& question, Json chat_body) {
            try {
                return parse_chat_completion_request(chat_body, limits);
            } catch (const ApiException& exception) {
                ApiError error = exception.error();
                error.message  = "question \"" + question.id + "\": " + error.message;
                throw ApiException(std::move(error));
            }
        };
        const auto common_fields = [&](Json& chat_body, bool first) {
            chat_body["model"]       = model;
            chat_body["max_tokens"]  = 1;
            chat_body["temperature"] = 0;
            chat_body["logprobs"]    = true;
            // Only the first branch writes: it publishes the shared prefix.
            chat_body["prompt_cache_read_only"] = !first;
            // The explicit boundary is the only write candidate. Without this the request also
            // carries the default implicit candidate at the end of its own question, and a
            // target that keeps one checkpoint per request (Flash-Next) would publish that
            // instead of the shared prefix.
            chat_body["prompt_cache_options"] = Json{{"mode", "explicit"}};
            if (body.contains("chat_template_kwargs")) {
                chat_body["chat_template_kwargs"] = body.at("chat_template_kwargs");
            }
        };

        std::size_t index = 0;
        for (const Json& question : body.at("questions")) {
            const std::string param = "questions[" + std::to_string(index) + "]";
            if (!question.is_object() || !question.contains("content") ||
                !question.at("content").is_string() ||
                question.at("content").get_ref<const std::string&>().empty()) {
                score_bad_request("each question needs a nonempty string content", param);
            }
            ScoreQuestion entry;
            entry.id = question.contains("id") && question.at("id").is_string()
                           ? question.at("id").get<std::string>()
                           : std::to_string(index);
            const Json& candidates =
                question.contains("candidates") ? question.at("candidates") : default_candidates;
            if (!candidates.is_array() || candidates.empty()) {
                score_bad_request("question \"" + entry.id + "\" has no candidates", param);
            }
            // Text candidates that are not one token switch the question to the text form.
            for (const Json& candidate : candidates) {
                if (candidate.is_number_integer()) { continue; }
                if (!candidate.is_string() || candidate.get_ref<const std::string&>().empty()) {
                    score_bad_request("question \"" + entry.id +
                                          "\": candidates are nonempty strings or token ids",
                                      param + ".candidates");
                }
                if (service_->tokenize_text(candidate.get<std::string>()).size() != 1) {
                    entry.text_form = true;
                }
            }
            if (entry.text_form) {
                for (const Json& candidate : candidates) {
                    if (!candidate.is_string()) {
                        score_bad_request("question \"" + entry.id +
                                              "\": a token id cannot be mixed with text "
                                              "candidates that span several tokens",
                                          param + ".candidates");
                    }
                    entry.option_texts.push_back(candidate.get<std::string>());
                }
                if (question.contains("response_format")) {
                    score_bad_request("question \"" + entry.id +
                                          "\": response_format does not apply to text candidates",
                                      param);
                }
            }

            const bool first  = branches.empty();
            Json messages     = first ? with_prefix_boundary(body.at("messages"))
                                      : body.at("messages");
            messages.push_back(Json{{"role", "user"}, {"content", question.at("content")}});
            if (!entry.text_form) {
                ScoreBranch branch;
                branch.question = index;
                Json chat_body{{"messages", messages},
                               {"top_logprobs", body.value("top_logprobs", 0)},
                               {"logprob_candidates", candidates}};
                common_fields(chat_body, first);
                if (question.contains("response_format")) {
                    chat_body["response_format"] = question.at("response_format");
                }
                branch.request = parse_branch(entry, std::move(chat_body));
                entry.branches.push_back(branches.size());
                branches.push_back(std::move(branch));
            } else {
                // Each option is the continued final assistant turn, which the template opens
                // exactly as the generation prompt does; so the option's first token sits at the
                // count of the prompt up to the question, and its length is the count of the
                // continued prompt minus that, with nothing rendered after it.
                Json base_body{{"messages", messages}};
                common_fields(base_body, false);
                const OpenAIChatRequest base = parse_branch(entry, base_body);
                const int first_token =
                    service_->count_prompt_tokens(base.generation, [&req] {
                        return client_disconnected(req);
                    });
                for (std::size_t k = 0; k < entry.option_texts.size(); ++k) {
                    ScoreBranch branch;
                    branch.question = index;
                    branch.option   = k;
                    Json chat_body  = base_body;
                    chat_body["messages"].push_back(
                        Json{{"role", "assistant"}, {"content", entry.option_texts[k]}});
                    common_fields(chat_body, first && k == 0);
                    branch.request = parse_branch(entry, chat_body);
                    branch.request.generation.continuation =
                        ninfer::PromptContinuationMode::ContinueFinalAssistant;
                    const int with_option = service_->count_prompt_tokens(
                        branch.request.generation, [&req] { return client_disconnected(req); });
                    const int length = with_option - first_token;
                    if (length <= 0 || first_token <= 0 ||
                        static_cast<std::size_t>(length) > kMaximumPromptReadouts) {
                        score_bad_request("question \"" + entry.id + "\": candidate \"" +
                                              entry.option_texts[k] +
                                              "\" cannot be scored as a continuation",
                                          param + ".candidates", "invalid_logprob_candidate");
                    }
                    for (int t = 0; t < length; ++t) {
                        branch.option_positions.push_back(
                            static_cast<std::uint32_t>(first_token - 1 + t));
                    }
                    branch.request.generation.logprob_prompt_positions = branch.option_positions;
                    entry.branches.push_back(branches.size());
                    branches.push_back(std::move(branch));
                }
            }
            questions.push_back(std::move(entry));
            ++index;
        }
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }

    std::shared_ptr<RequestLifetime> lifetime;
    const auto run_branch = [&](ScoreBranch& branch) {
        const OpenAIChatRequest& request = branch.request;
        const std::uint64_t req_id       = ++request_seq_;
        const RequestLogMetadata metadata{.model                  = request.model,
                                          .stream                 = false,
                                          .output_tokens_explicit = true};
        PreparedRequest prepared;
        try {
            prepared = service_->prepare(request.generation, GenerationConsumerMode::Aggregate,
                                         [&req] { return client_disconnected(req); });
        } catch (const ApiException& exception) {
            record_request_rejected(make_request_rejection_log_context(
                req_id, "ninfer_score", request.generation, metadata, exception.error()));
            branch.error = exception.error();
            return;
        } catch (const std::exception& exception) {
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = exception.what();
            branch.error  = std::move(error);
            return;
        }
        auto lifecycle = begin_request(make_request_log_context(
            req_id, "ninfer_score", request.generation, metadata, prepared, client_label(req)));
        try {
            branch.outcome =
                service_->run(prepared, nullptr, [&req] { return client_disconnected(req); });
            lifecycle->done(*branch.outcome);
        } catch (const ApiException& exception) {
            lifecycle->failure(make_generation_request_failure(exception.error()));
            branch.error = exception.error();
        } catch (const std::exception& exception) {
            lifecycle->failure(
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = exception.what();
            branch.error  = std::move(error);
        }
        if (!lifetime) { lifetime = prepared.lifetime; }
    };

    // One branch at a time. The first publishes the prefix; each later one reads it. Prefill is
    // one sequence at a time on every target, so concurrent branches gain little when they hit,
    // and a target that keeps a published prefix as a private continuation (Flash-Next) admits
    // one reader at a time: concurrent branches there miss and prefill cold.
    for (ScoreBranch& branch : branches) { run_branch(branch); }

    try {
        Json rendered         = Json::array();
        std::int64_t prompt   = 0;
        std::int64_t cached   = 0;
        std::int64_t produced = 0;
        for (std::size_t index = 0; index < questions.size(); ++index) {
            const ScoreQuestion& question = questions[index];
            Json entry{{"id", question.id}, {"index", index}};
            const auto fail = [&](const std::optional<ApiError>& error) {
                ApiError missing;
                missing.status  = 500;
                missing.type    = "internal_error";
                missing.message = "the question produced no scored position";
                entry["error"]  = error_json(error ? *error : missing);
                rendered.push_back(std::move(entry));
            };
            std::vector<double> option_logprobs;
            if (!question.text_form) {
                const ScoreBranch& branch = branches[question.branches.front()];
                if (branch.error || !branch.outcome || branch.outcome->token_logprobs.empty()) {
                    fail(branch.error);
                    continue;
                }
                const GenerationOutcome& outcome     = *branch.outcome;
                const TokenLogprobPosition& position = outcome.token_logprobs.front();
                const Json sampled                   = entry_json(position.sampled);
                entry["form"]                        = "token";
                entry["token"]                       = sampled.at("token");
                entry["token_id"]                    = sampled.at("token_id");
                Json candidates                      = Json::array();
                double inside                        = 0.0;
                for (const TokenLogprobEntry& value : position.candidates) {
                    candidates.push_back(entry_json(value));
                    option_logprobs.push_back(finite_logprob(value.logprob));
                    if (std::isfinite(value.raw_logprob)) { inside += std::exp(value.raw_logprob); }
                }
                entry["candidate_logprobs"] = std::move(candidates);
                Json top                    = Json::array();
                for (const TokenLogprobEntry& value : position.top) { top.push_back(entry_json(value)); }
                entry["top_logprobs"] = std::move(top);
                // Vocabulary-wide probability the model put outside the candidate list.
                entry["outside_mass"]  = std::clamp(1.0 - inside, 0.0, 1.0);
                entry["cached_tokens"] = outcome.metrics.prefix_cache_hit_tokens;
                prompt += outcome.prompt_tokens;
                cached += outcome.metrics.prefix_cache_hit_tokens;
                produced += outcome.completion_tokens;
            } else {
                // Each option's conditional log-probability is the sum over its tokens; the
                // distribution over options renormalises those sums.
                std::vector<double> sums;
                std::optional<ApiError> error;
                Json candidates = Json::array();
                std::uint64_t question_cached = 0;
                for (const std::size_t b : question.branches) {
                    const ScoreBranch& branch = branches[b];
                    if (branch.error || !branch.outcome ||
                        branch.outcome->prompt_logprobs.size() != branch.option_positions.size()) {
                        error = branch.error;
                        if (!error) {
                            error.emplace();
                            error->status  = 500;
                            error->type    = "internal_error";
                            error->message = "a candidate produced no scored positions";
                        }
                        break;
                    }
                    const GenerationOutcome& outcome = *branch.outcome;
                    double sum                       = 0.0;
                    Json tokens                      = Json::array();
                    for (const PromptLogprobPosition& position : outcome.prompt_logprobs) {
                        Json token = entry_json(position.value.sampled);
                        token.erase("raw_logprob");
                        token["position"] = position.position;
                        tokens.push_back(std::move(token));
                        sum += finite_logprob(position.value.sampled.logprob);
                    }
                    sums.push_back(sum);
                    candidates.push_back(Json{{"token", question.option_texts[*branch.option]},
                                              {"raw_logprob", sum},
                                              {"tokens", std::move(tokens)}});
                    question_cached = std::max<std::uint64_t>(
                        question_cached, outcome.metrics.prefix_cache_hit_tokens);
                    prompt += outcome.prompt_tokens;
                    cached += outcome.metrics.prefix_cache_hit_tokens;
                    produced += outcome.completion_tokens;
                }
                if (error) {
                    fail(error);
                    continue;
                }
                const double maximum = *std::max_element(sums.begin(), sums.end());
                double total         = 0.0;
                for (const double sum : sums) { total += std::exp(sum - maximum); }
                const double normaliser = maximum + std::log(total);
                std::size_t best        = 0;
                for (std::size_t k = 0; k < sums.size(); ++k) {
                    candidates[k]["logprob"] = sums[k] - normaliser;
                    option_logprobs.push_back(sums[k] - normaliser);
                    if (sums[k] > sums[best]) { best = k; }
                }
                entry["form"]               = "text";
                entry["token"]              = question.option_texts[best];
                entry["candidate_logprobs"] = std::move(candidates);
                entry["cached_tokens"]      = question_cached;
            }
            double entropy = 0.0, margin = 0.0;
            distribution_stats(option_logprobs, entropy, margin);
            entry["entropy"] = entropy;
            entry["margin"]  = finite_json(margin);
            rendered.push_back(std::move(entry));
        }
        Json payload{{"object", "score"},
                     {"model", model},
                     {"results", std::move(rendered)},
                     {"usage", Json{{"prompt_tokens", prompt},
                                    {"cached_tokens", cached},
                                    {"completion_tokens", produced}}}};
        if (lifetime) {
            set_owned_json_content(res, payload.dump(), lifetime);
        } else {
            res.set_content(payload.dump(), "application/json");
        }
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
    }
}

} // namespace ninfer::serve
