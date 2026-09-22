#include "serve/typesafe_systemone.h"

#include "serve/http_server.h"
#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::ordered_json;

constexpr double kLogprobFloor = -9999.0;

struct SystemOneBranch {
    std::size_t question_index = 0;
    OpenAIChatRequest request;
    std::optional<GenerationOutcome> outcome;
    std::optional<ApiError> error;
};

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

} // namespace

[[noreturn]] void systemone_bad_request(std::string message, std::string param, std::string code) {
    ApiError error;
    error.status  = 422; // Unprocessable Entity per TypeSafe API specification
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

void write_typesafe_error(httplib::Response& response, const ApiError& error) {
    response.status                = error.status != 0 ? error.status : 422;
    nlohmann::ordered_json err_obj = {
        {"message", error.message},
        {"type", error.type.empty() ? "invalid_request_error" : error.type}};
    if (!error.param.empty()) { err_obj["param"] = error.param; }
    if (!error.code.empty()) { err_obj["code"] = error.code; }
    nlohmann::ordered_json body = {{"error", std::move(err_obj)}, {"message", error.message}};
    if (!error.param.empty()) { body["param"] = error.param; }
    response.set_content(body.dump(), "application/json");
}

std::string choice_token_for_index(std::size_t index) {
    if (index < 26) { return std::string(1, static_cast<char>('A' + index)); }
    if (index < 52) { return std::string(1, static_cast<char>('a' + (index - 26))); }
    if (index < kMaximumSystemOneChoices) {
        return std::string(1, static_cast<char>('0' + (index - 52)));
    }
    return {};
}

std::int64_t systemone_billed_input_tokens(int prompt_tokens, std::uint32_t cached_tokens,
                                           bool first) {
    if (prompt_tokens <= 0) { return 0; }
    if (first) { return prompt_tokens; }
    const auto cached = static_cast<int>(std::min<std::uint32_t>(
        cached_tokens, static_cast<std::uint32_t>(std::numeric_limits<int>::max())));
    const int fresh   = prompt_tokens - cached;
    return fresh > 0 ? fresh : 0;
}

std::vector<double> softmax_probabilities(const std::vector<double>& logprobs) {
    if (logprobs.empty()) { return {}; }
    if (logprobs.size() == 1) { return {1.0}; }
    const double max_lp = *std::max_element(logprobs.begin(), logprobs.end());
    std::vector<double> probabilities(logprobs.size(), 0.0);
    double sum = 0.0;
    for (std::size_t i = 0; i < logprobs.size(); ++i) {
        if (logprobs[i] > -9900.0) {
            probabilities[i] = std::exp(logprobs[i] - max_lp);
            sum += probabilities[i];
        }
    }
    if (sum > 0.0) {
        for (double& p : probabilities) { p /= sum; }
    } else {
        const double uniform = 1.0 / static_cast<double>(probabilities.size());
        std::fill(probabilities.begin(), probabilities.end(), uniform);
    }
    return probabilities;
}

double calculate_choice_confidence(const std::vector<double>& probabilities) {
    const std::size_t n = probabilities.size();
    if (n == 0) { return 0.0; }
    if (n == 1) { return 1.0; }
    const double max_p = *std::max_element(probabilities.begin(), probabilities.end());
    const double conf  = (static_cast<double>(n) * max_p - 1.0) / (static_cast<double>(n) - 1.0);
    return std::clamp(conf, 0.0, 1.0);
}

double calculate_expected_score(const std::vector<double>& probabilities) {
    double score = 0.0;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        score += static_cast<double>(i) * probabilities[i];
    }
    return score;
}

std::string build_question_prompt(const SystemOneQuestion& question) {
    std::string prompt;
    switch (question.type) {
    case SystemOneQuestionType::Noul: {
        prompt += "Question: " + question.instructions + "\n";
        if (!question.true_criteria.empty() || !question.false_criteria.empty()) {
            prompt += "\nCriteria:\n";
            if (!question.true_criteria.empty()) {
                prompt += "- Yes: " + question.true_criteria + "\n";
            }
            if (!question.false_criteria.empty()) {
                prompt += "- No: " + question.false_criteria + "\n";
            }
        }
        prompt += "\nAnswer with only Yes or No:\n";
        break;
    }
    case SystemOneQuestionType::Choice: {
        prompt += "Question: " + question.instructions + "\n\nOptions:\n";
        for (const auto& opt : question.choice_options) {
            if (opt.key == opt.token) {
                if (!opt.description.empty()) {
                    prompt += "- " + opt.token + ": " + opt.description + "\n";
                } else {
                    prompt += "- " + opt.token + "\n";
                }
            } else {
                if (!opt.description.empty()) {
                    prompt += "- " + opt.token + ": [" + opt.key + "] " + opt.description + "\n";
                } else {
                    prompt += "- " + opt.token + ": [" + opt.key + "]\n";
                }
            }
        }
        if (question.choice_options.size() <= 26) {
            prompt += "\nSelect the best option. Answer with only the option letter:\n";
        } else {
            prompt += "\nSelect the best option. Answer with only the option letter or digit:\n";
        }
        break;
    }
    case SystemOneQuestionType::Score: {
        prompt += "Question: " + question.instructions + "\n\nRating levels:\n";
        for (std::size_t i = 0; i < question.score_levels.size(); ++i) {
            prompt += "- " + std::to_string(i) + ": " + question.score_levels[i] + "\n";
        }
        prompt +=
            "\nSelect the rating level that best applies. Answer with only the level number:\n";
        break;
    }
    }
    return prompt;
}

SystemOneRequest parse_systemone_request(const nlohmann::ordered_json& body) {
    if (!body.is_object()) { systemone_bad_request("request body must be an object", ""); }
    if (!body.contains("model") || !body.at("model").is_string()) {
        systemone_bad_request("model must be a string", "model");
    }
    std::string requested_model = body.at("model").get<std::string>();
    if (requested_model.empty()) { systemone_bad_request("model cannot be empty", "model"); }

    if (!body.contains("state")) { systemone_bad_request("state is required", "state"); }
    const auto& state_val = body.at("state");
    std::string state_text;
    if (state_val.is_string()) {
        state_text = state_val.get<std::string>();
    } else if (state_val.is_object() || state_val.is_array()) {
        state_text = state_val.dump(2);
    } else {
        systemone_bad_request("state must be a string, object, or array", "state");
    }

    if (!body.contains("questions") || !body.at("questions").is_object() ||
        body.at("questions").empty()) {
        systemone_bad_request("questions must be a nonempty object", "questions");
    }
    if (body.at("questions").size() > kMaximumSystemOneQuestions) {
        systemone_bad_request("questions cannot exceed 256 entries", "questions");
    }

    SystemOneRequest req;
    req.requested_model = std::move(requested_model);
    req.state_text      = std::move(state_text);

    double temperature = 1.0;
    if (body.contains("temperature") && !body.at("temperature").is_null()) {
        if (!body.at("temperature").is_number()) {
            systemone_bad_request("temperature must be a number", "temperature");
        }
        temperature = body.at("temperature").get<double>();
        if (temperature <= 0.0) {
            systemone_bad_request("temperature must be positive", "temperature");
        }
    } else {
        const auto t_pos = req.requested_model.rfind("-t");
        if (t_pos != std::string::npos && t_pos + 2 < req.requested_model.size()) {
            try {
                const double parsed_t = std::stod(req.requested_model.substr(t_pos + 2));
                if (parsed_t > 0.0) { temperature = parsed_t; }
            } catch (...) {}
        }
    }
    req.temperature = temperature;

    for (auto it = body.at("questions").begin(); it != body.at("questions").end(); ++it) {
        const std::string q_id = it.key();
        if (q_id.empty()) {
            systemone_bad_request("question id cannot be empty", "questions");
        }
        const auto& q_obj         = it.value();
        const std::string q_param = "questions." + q_id;

        if (!q_obj.is_object()) { systemone_bad_request("question must be an object", q_param); }
        if (!q_obj.contains("type") || !q_obj.at("type").is_string()) {
            systemone_bad_request("question type must be a string", q_param + ".type");
        }
        const std::string type_str = q_obj.at("type").get<std::string>();

        if (!q_obj.contains("instructions")) {
            systemone_bad_request("instructions is required", q_param + ".instructions");
        }
        const auto& inst_val = q_obj.at("instructions");
        std::string inst_text;
        if (inst_val.is_string()) {
            inst_text = inst_val.get<std::string>();
        } else if (inst_val.is_object() || inst_val.is_array()) {
            inst_text = inst_val.dump(2);
        } else {
            systemone_bad_request("instructions must be a string, object, or array",
                                  q_param + ".instructions");
        }
        if (inst_text.empty()) {
            systemone_bad_request("instructions cannot be empty", q_param + ".instructions");
        }

        SystemOneQuestion q;
        q.id           = q_id;
        q.instructions = std::move(inst_text);

        if (type_str == "noul") {
            q.type       = SystemOneQuestionType::Noul;
            q.candidates = {"Yes", "No"};
            if (q_obj.contains("criteria")) {
                const auto& crit = q_obj.at("criteria");
                if (!crit.is_null()) {
                    if (!crit.is_object()) {
                        systemone_bad_request("criteria for noul must be an object",
                                              q_param + ".criteria");
                    }
                    if (crit.contains("true")) {
                        const auto& t_val = crit.at("true");
                        q.true_criteria =
                            t_val.is_string() ? t_val.get<std::string>() : t_val.dump(2);
                    }
                    if (crit.contains("false")) {
                        const auto& f_val = crit.at("false");
                        q.false_criteria =
                            f_val.is_string() ? f_val.get<std::string>() : f_val.dump(2);
                    }
                }
            }
        } else if (type_str == "choice") {
            q.type = SystemOneQuestionType::Choice;
            if (!q_obj.contains("criteria") || !q_obj.at("criteria").is_object() ||
                q_obj.at("criteria").empty()) {
                systemone_bad_request("criteria for choice must be a nonempty object",
                                      q_param + ".criteria");
            }
            const auto& crit = q_obj.at("criteria");
            if (crit.size() > kMaximumSystemOneChoices) {
                systemone_bad_request(
                    "criteria for choice cannot exceed 62 options",
                    q_param + ".criteria");
            }
            bool all_single_char = true;
            for (auto opt_it = crit.begin(); opt_it != crit.end(); ++opt_it) {
                const std::string& k = opt_it.key();
                if (k.size() != 1 || static_cast<unsigned char>(k[0]) <= 32 ||
                    static_cast<unsigned char>(k[0]) >= 127) {
                    all_single_char = false;
                    break;
                }
            }
            std::size_t opt_index = 0;
            for (auto opt_it = crit.begin(); opt_it != crit.end(); ++opt_it, ++opt_index) {
                const std::string opt_key = opt_it.key();
                const auto& opt_val       = opt_it.value();
                std::string opt_desc;
                if (opt_val.is_string()) {
                    opt_desc = opt_val.get<std::string>();
                } else if (opt_val.is_object() || opt_val.is_array()) {
                    opt_desc = opt_val.dump(2);
                } else if (opt_val.is_null()) {
                    opt_desc = "";
                } else {
                    systemone_bad_request(
                        "choice criteria description must be a string, object, array, or null",
                        q_param + ".criteria." + opt_key);
                }
                const std::string cand_token =
                    all_single_char ? opt_key : choice_token_for_index(opt_index);
                q.choice_options.push_back(SystemOneChoiceOption{
                    .key         = opt_key,
                    .description = std::move(opt_desc),
                    .token       = cand_token,
                });
                q.candidates.push_back(cand_token);
            }
        } else if (type_str == "score") {
            q.type = SystemOneQuestionType::Score;
            if (!q_obj.contains("criteria") || !q_obj.at("criteria").is_array()) {
                systemone_bad_request("criteria for score must be an array", q_param + ".criteria");
            }
            const auto& crit = q_obj.at("criteria");
            if (crit.size() < kMinimumScoreLevels || crit.size() > kMaximumScoreLevels) {
                systemone_bad_request("criteria for score must be an array of 2 to 10 levels",
                                      q_param + ".criteria");
            }
            for (std::size_t lvl_i = 0; lvl_i < crit.size(); ++lvl_i) {
                const auto& lvl_val = crit[lvl_i];
                std::string lvl_desc;
                if (lvl_val.is_string()) {
                    lvl_desc = lvl_val.get<std::string>();
                } else if (lvl_val.is_object() || lvl_val.is_array()) {
                    lvl_desc = lvl_val.dump(2);
                } else if (lvl_val.is_null()) {
                    lvl_desc = "";
                } else {
                    systemone_bad_request(
                        "score level description must be a string, object, array, or null",
                        q_param + ".criteria[" + std::to_string(lvl_i) + "]");
                }
                q.score_levels.push_back(std::move(lvl_desc));
                q.candidates.push_back(std::to_string(lvl_i));
            }
        } else {
            systemone_bad_request("unknown question type '" + type_str + "'", q_param + ".type");
        }

        req.questions.push_back(std::move(q));
    }

    return req;
}

nlohmann::ordered_json make_systemone_response_json(const SystemOneResponse& response) {
    nlohmann::ordered_json answers = nlohmann::ordered_json::object();
    for (const auto& answer : response.answers) {
        nlohmann::ordered_json item;
        switch (answer.type) {
        case SystemOneQuestionType::Noul:
            item["type"] = "noul";
            item["noul"] = answer.noul;
            break;
        case SystemOneQuestionType::Choice: {
            item["type"]                 = "choice";
            item["choice"]               = answer.choice_winner;
            nlohmann::ordered_json probs = nlohmann::ordered_json::object();
            for (const auto& [k, p] : answer.choice_probabilities) { probs[k] = p; }
            item["probabilities"] = std::move(probs);
            item["confidence"]    = answer.choice_confidence;
            break;
        }
        case SystemOneQuestionType::Score: {
            item["type"]                  = "score";
            item["score"]                 = answer.score;
            nlohmann::ordered_json legend = nlohmann::ordered_json::object();
            for (const auto& [k, desc] : answer.score_legend) { legend[k] = desc; }
            item["legend"]               = std::move(legend);
            nlohmann::ordered_json probs = nlohmann::ordered_json::object();
            for (const auto& [k, p] : answer.score_probabilities) { probs[k] = p; }
            item["probabilities"] = std::move(probs);
            item["confidence"]    = answer.score_confidence;
            break;
        }
        }
        answers[answer.id] = std::move(item);
    }
    return nlohmann::ordered_json{
        {"model", response.model},
        {"answers", std::move(answers)},
        {"usage", nlohmann::ordered_json{{"input_tokens", response.input_tokens},
                                         {"output_tokens", response.output_tokens}}}};
}

void HttpServer::handle_systemone(const httplib::Request& req, httplib::Response& res) {
    SystemOneRequest sys_request;
    try {
        const Json body = parse_json_body(req);
        if (body.contains("stream") && body.at("stream").is_boolean() &&
            body.at("stream").get<bool>()) {
            systemone_bad_request("systemone responses are not streamed", "stream");
        }
        sys_request = parse_systemone_request(body);
        for (const SystemOneQuestion& question : sys_request.questions) {
            for (const std::string& candidate : question.candidates) {
                if (service_->tokenize_text(candidate).size() != 1) {
                    systemone_bad_request(
                        "candidate \"" + candidate + "\" must be exactly one token",
                        "questions." + question.id + ".criteria");
                }
            }
        }
    } catch (const ApiException& exception) {
        write_typesafe_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 422;
        error.type    = "invalid_request_error";
        error.message = exception.what();
        write_typesafe_error(res, error);
        return;
    }

    std::vector<SystemOneBranch> branches;
    branches.reserve(sys_request.questions.size());

    RequestLimits limits;
    limits.default_max_tokens = options_.default_max_tokens;

    const Json base_messages =
        Json::array({Json{{"role", "system"},
                          {"content", "You are an evaluation assistant. State to evaluate:\n" +
                                          sys_request.state_text}}});

    for (std::size_t i = 0; i < sys_request.questions.size(); ++i) {
        const auto& question = sys_request.questions[i];
        const bool first     = (i == 0);
        Json messages        = first ? with_prefix_boundary(base_messages) : base_messages;
        messages.push_back(Json{{"role", "user"}, {"content", build_question_prompt(question)}});

        Json chat_body{{"model", public_model_id_},
                       {"messages", std::move(messages)},
                       {"max_tokens", 1},
                       {"temperature", 0},
                       {"logprobs", true},
                       {"logprob_candidates", question.candidates},
                       {"prompt_cache_read_only", !first},
                       {"prompt_cache_options", Json{{"mode", "explicit"}}},
                       {"chat_template_kwargs", Json{{"enable_thinking", false}}}};

        SystemOneBranch branch;
        branch.question_index = i;
        try {
            branch.request = parse_chat_completion_request(chat_body, limits);
        } catch (const ApiException& exception) {
            ApiError error = exception.error();
            error.status   = 422;
            error.message  = "question \"" + question.id + "\": " + error.message;
            write_typesafe_error(res, error);
            return;
        } catch (const std::exception& exception) {
            ApiError error;
            error.status  = 422;
            error.type    = "invalid_request_error";
            error.message = "question \"" + question.id + "\": " + exception.what();
            write_typesafe_error(res, error);
            return;
        }
        branches.push_back(std::move(branch));
    }

    std::shared_ptr<RequestLifetime> lifetime;
    for (SystemOneBranch& branch : branches) {
        const OpenAIChatRequest& request = branch.request;
        const std::uint64_t req_id       = ++request_seq_;
        const RequestLogMetadata metadata{
            .model = request.model, .stream = false, .output_tokens_explicit = true};
        PreparedRequest prepared;
        try {
            prepared = service_->prepare(request.generation, GenerationConsumerMode::Aggregate,
                                         [&req] { return client_disconnected(req); });
        } catch (const ApiException& exception) {
            record_request_rejected(make_request_rejection_log_context(
                req_id, "typesafe_systemone", request.generation, metadata, exception.error()));
            branch.error = exception.error();
            write_typesafe_error(res, exception.error());
            return;
        } catch (const std::exception& exception) {
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = exception.what();
            branch.error  = error;
            write_typesafe_error(res, error);
            return;
        }

        auto lifecycle =
            begin_request(make_request_log_context(req_id, "typesafe_systemone", request.generation,
                                                   metadata, prepared, client_label(req)));
        try {
            branch.outcome =
                service_->run(prepared, nullptr, [&req] { return client_disconnected(req); });
            lifecycle->done(*branch.outcome);
        } catch (const ApiException& exception) {
            lifecycle->failure(make_generation_request_failure(exception.error()));
            branch.error = exception.error();
            write_typesafe_error(res, exception.error());
            return;
        } catch (const std::exception& exception) {
            lifecycle->failure(
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = exception.what();
            branch.error  = error;
            write_typesafe_error(res, error);
            return;
        }

        if (!lifetime) { lifetime = prepared.lifetime; }
    }

    SystemOneResponse response;
    response.model = sys_request.requested_model;
    response.answers.reserve(sys_request.questions.size());

    for (std::size_t i = 0; i < sys_request.questions.size(); ++i) {
        const auto& question = sys_request.questions[i];
        const auto& branch   = branches[i];
        if (!branch.outcome || branch.outcome->token_logprobs.empty() ||
            branch.outcome->token_logprobs.front().candidates.empty()) {
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = "question \"" + question.id + "\" produced no candidate logprobs";
            write_typesafe_error(res, error);
            return;
        }
        const GenerationOutcome& outcome = *branch.outcome;
        const TokenLogprobPosition& pos  = outcome.token_logprobs.front();
        if (pos.candidates.size() != question.candidates.size()) {
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = "question \"" + question.id + "\" candidate count mismatch";
            write_typesafe_error(res, error);
            return;
        }
        response.input_tokens += systemone_billed_input_tokens(
            outcome.prompt_tokens, outcome.metrics.prefix_cache_hit_tokens, i == 0);

        std::vector<double> candidate_lps;
        candidate_lps.reserve(pos.candidates.size());
        for (const auto& cand : pos.candidates) {
            const float lp = std::isfinite(cand.raw_logprob) ? cand.raw_logprob : cand.logprob;
            double val     = finite_logprob(lp);
            if (sys_request.temperature > 0.0 && sys_request.temperature != 1.0 &&
                val > -9900.0) {
                val /= sys_request.temperature;
            }
            candidate_lps.push_back(val);
        }
        const std::vector<double> probs = softmax_probabilities(candidate_lps);

        SystemOneAnswer answer;
        answer.id   = question.id;
        answer.type = question.type;

        switch (question.type) {
        case SystemOneQuestionType::Noul: {
            answer.noul = probs.empty() ? 0.0 : probs[0];
            break;
        }
        case SystemOneQuestionType::Choice: {
            std::size_t winner_idx = 0;
            double max_p           = -1.0;
            for (std::size_t opt_i = 0; opt_i < question.choice_options.size(); ++opt_i) {
                const double p = opt_i < probs.size() ? probs[opt_i] : 0.0;
                answer.choice_probabilities.emplace_back(question.choice_options[opt_i].key, p);
                if (p > max_p) {
                    max_p      = p;
                    winner_idx = opt_i;
                }
            }
            answer.choice_winner =
                question.choice_options.empty() ? "" : question.choice_options[winner_idx].key;
            answer.choice_confidence = calculate_choice_confidence(probs);
            break;
        }
        case SystemOneQuestionType::Score: {
            for (std::size_t lvl_i = 0; lvl_i < question.score_levels.size(); ++lvl_i) {
                const std::string key = std::to_string(lvl_i);
                const double p        = lvl_i < probs.size() ? probs[lvl_i] : 0.0;
                answer.score_legend.emplace_back(key, question.score_levels[lvl_i]);
                answer.score_probabilities.emplace_back(key, p);
            }
            answer.score            = calculate_expected_score(probs);
            answer.score_confidence = calculate_choice_confidence(probs);
            break;
        }
        }
        response.answers.push_back(std::move(answer));
    }

    const Json payload = make_systemone_response_json(response);
    if (lifetime) {
        set_owned_json_content(res, payload.dump(), lifetime);
    } else {
        res.set_content(payload.dump(), "application/json");
    }
}

} // namespace ninfer::serve
