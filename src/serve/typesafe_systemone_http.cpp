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
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::ordered_json;

// One Engine request. Every question has a first branch that reads the first answer token. A
// Choice with two-token labels adds one branch per label letter: the answer continues with that
// letter and the branch reads the digit, so log P(A7) = log P(A) + log P(7 | A). Labels are then
// normalised over the option set exactly like one-token labels.
struct SystemOneBranch {
    std::size_t question = 0;
    std::optional<std::size_t> letter;
    OpenAIChatRequest request;
    std::optional<GenerationOutcome> outcome;
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

std::size_t letter_group_size(const SystemOneQuestion& question, std::size_t letter) {
    return std::min<std::size_t>(10, question.choice_options.size() - letter * 10);
}

// Vocabulary-wide log-probabilities of the branch's candidates at its answer position.
std::vector<double> candidate_logprobs(const SystemOneBranch& branch, std::size_t expected) {
    const GenerationOutcome& outcome = *branch.outcome;
    if (outcome.token_logprobs.empty() ||
        outcome.token_logprobs.front().candidates.size() != expected) {
        throw std::runtime_error("System One branch returned no candidate distribution");
    }
    std::vector<double> logprobs;
    logprobs.reserve(expected);
    for (const auto& candidate : outcome.token_logprobs.front().candidates) {
        logprobs.push_back(static_cast<double>(
            std::isfinite(candidate.raw_logprob) ? candidate.raw_logprob : candidate.logprob));
    }
    return logprobs;
}

ApiError internal_error(const char* what) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = what;
    return error;
}

} // namespace

void write_typesafe_failure(httplib::Response& response, const SystemOneError& error) {
    response.status = error.status();
    if (error.status() == 405) { response.set_header("Allow", "POST"); }
    response.set_content(error.body().dump(), "application/json");
}

void write_typesafe_error(httplib::Response& response, const ApiError& error) {
    write_typesafe_failure(response, systemone_error_from(error));
}

void HttpServer::handle_systemone(const httplib::Request& req, httplib::Response& res) {
    SystemOneRequest request;
    std::vector<SystemOneBranch> branches;
    try {
        request = parse_systemone_request(parse_systemone_body(req.body));

        // Labels are printable ASCII characters, "Yes"/"No" and digits: each one native token.
        std::unordered_map<std::string, ninfer::TokenId> token_ids;
        const auto token_id = [&](const std::string& text) {
            auto found = token_ids.find(text);
            if (found == token_ids.end()) {
                const auto tokens = service_->tokenize_text(text);
                if (tokens.size() != 1) {
                    throw std::runtime_error("System One label \"" + text +
                                             "\" is not one token");
                }
                found = token_ids.emplace(text, tokens.front()).first;
            }
            return found->second;
        };

        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        const auto branch_request = [&](Json messages, Json candidates, bool publish) {
            return parse_chat_completion_request(
                Json{{"model", public_model_id_},
                     {"messages", std::move(messages)},
                     {"max_tokens", 1},
                     {"temperature", 0},
                     {"logprobs", true},
                     {"logprob_candidates", std::move(candidates)},
                     {"prompt_cache_read_only", !publish},
                     {"prompt_cache_options", Json{{"mode", publish ? "explicit" : "implicit"}}},
                     {"chat_template_kwargs", Json{{"enable_thinking", false}}}},
                limits);
        };

        // Several questions about one state: the first prefill publishes the state and the rest
        // read it. A single question never reuses its own prefix, so it publishes nothing. A
        // two-token Choice also publishes its question, which each letter branch then extends
        // by only the answer opener and one letter.
        const Json base          = build_systemone_messages(request);
        const bool publish_state = request.questions.size() > 1;
        for (std::size_t q = 0; q < request.questions.size(); ++q) {
            const SystemOneQuestion& question = request.questions[q];
            const std::string prompt          = build_question_prompt(question);
            const std::vector<std::string> first_tokens = first_token_candidates(question);

            Json ids = Json::array();
            for (const std::string& candidate : first_tokens) { ids.push_back(token_id(candidate)); }
            const bool state_here = publish_state && q == 0;
            Json messages         = state_here ? with_prefix_boundary(base) : base;
            if (question.two_token_labels) {
                messages.push_back(Json{
                    {"role", "user"},
                    {"content", Json::array({Json{{"type", "text"},
                                                  {"text", prompt},
                                                  {"prompt_cache_breakpoint",
                                                   Json{{"mode", "explicit"}}}}})}});
            } else {
                messages.push_back(Json{{"role", "user"}, {"content", prompt}});
            }
            SystemOneBranch first;
            first.question = q;
            first.request  = branch_request(std::move(messages), std::move(ids),
                                            state_here || question.two_token_labels);
            branches.push_back(std::move(first));
            if (!question.two_token_labels) { continue; }

            // The pre-tokenizer must split a label into its letter and digit, or the answer
            // distribution would sit on a merged token no branch reads.
            for (const SystemOneChoiceOption& option : question.choice_options) {
                const std::vector<ninfer::TokenId> expected{token_id(option.label.substr(0, 1)),
                                                            token_id(option.label.substr(1))};
                if (service_->tokenize_text(option.label) != expected) {
                    throw std::runtime_error("System One label \"" + option.label +
                                             "\" does not split into a letter and a digit");
                }
            }
            for (std::size_t letter = 0; letter < first_tokens.size(); ++letter) {
                Json continued = base;
                continued.push_back(Json{{"role", "user"}, {"content", prompt}});
                continued.push_back(Json{{"role", "assistant"}, {"content", first_tokens[letter]}});
                Json digits = Json::array();
                for (std::size_t digit = 0; digit < letter_group_size(question, letter); ++digit) {
                    digits.push_back(token_id(std::string(1, static_cast<char>('0' + digit))));
                }
                SystemOneBranch branch;
                branch.question = q;
                branch.letter   = letter;
                branch.request  = branch_request(std::move(continued), std::move(digits), false);
                branch.request.generation.continuation =
                    ninfer::PromptContinuationMode::ContinueFinalAssistant;
                branches.push_back(std::move(branch));
            }
        }
    } catch (const SystemOneError& error) {
        write_typesafe_failure(res, error);
        return;
    } catch (const ApiException& exception) {
        write_typesafe_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        write_typesafe_error(res, internal_error(exception.what()));
        return;
    }

    std::shared_ptr<RequestLifetime> lifetime;
    SystemOneUsage usage;
    for (SystemOneBranch& branch : branches) {
        const OpenAIChatRequest& chat = branch.request;
        const std::uint64_t req_id    = ++request_seq_;
        const RequestLogMetadata metadata{
            .model = chat.model, .stream = false, .output_tokens_explicit = true};
        PreparedRequest prepared;
        try {
            prepared = service_->prepare(chat.generation, GenerationConsumerMode::Aggregate,
                                         [&req] { return client_disconnected(req); });
        } catch (const ApiException& exception) {
            record_request_rejected(make_request_rejection_log_context(
                req_id, "typesafe_systemone", chat.generation, metadata, exception.error()));
            write_typesafe_error(res, exception.error());
            return;
        } catch (const std::exception& exception) {
            write_typesafe_error(res, internal_error(exception.what()));
            return;
        }

        auto lifecycle =
            begin_request(make_request_log_context(req_id, "typesafe_systemone", chat.generation,
                                                   metadata, prepared, client_label(req)));
        try {
            branch.outcome =
                service_->run(prepared, nullptr, [&req] { return client_disconnected(req); });
            lifecycle->done(*branch.outcome);
        } catch (const ApiException& exception) {
            lifecycle->failure(make_generation_request_failure(exception.error()));
            write_typesafe_error(res, exception.error());
            return;
        } catch (const std::exception& exception) {
            lifecycle->failure(
                make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
            write_typesafe_error(res, internal_error(exception.what()));
            return;
        }

        if (!lifetime) {
            lifetime = prepared.lifetime;
            // Shared observation count, once per request rather than once per question.
            usage.vision_tokens = prepared.preparation.vision_tokens;
        }
    }

    std::vector<SystemOneAnswer> answers(request.questions.size());
    try {
        std::vector<std::vector<double>> first(request.questions.size());
        std::vector<std::vector<std::vector<double>>> digits(request.questions.size());
        for (std::size_t k = 0; k < branches.size(); ++k) {
            const SystemOneBranch& branch     = branches[k];
            const SystemOneQuestion& question = request.questions[branch.question];
            usage.input_tokens += systemone_billed_input_tokens(
                branch.outcome->prompt_tokens, branch.outcome->metrics.prefix_cache_hit_tokens,
                k == 0);
            if (branch.letter) {
                digits[branch.question].push_back(
                    candidate_logprobs(branch, letter_group_size(question, *branch.letter)));
            } else {
                first[branch.question] =
                    candidate_logprobs(branch, first_token_candidates(question).size());
            }
        }
        for (std::size_t q = 0; q < request.questions.size(); ++q) {
            const SystemOneQuestion& question = request.questions[q];
            std::vector<double> label_logprobs;
            if (question.two_token_labels) {
                for (std::size_t i = 0; i < question.choice_options.size(); ++i) {
                    label_logprobs.push_back(first[q].at(i / 10) + digits[q].at(i / 10).at(i % 10));
                }
            } else {
                label_logprobs = std::move(first[q]);
            }
            std::vector<double> probabilities =
                softmax_probabilities(label_logprobs, request.temperature);
            if (question.type == SystemOneQuestionType::Noul) { probabilities.resize(1); }
            answers[q].probabilities = std::move(probabilities);
        }
    } catch (const std::exception& exception) {
        write_typesafe_error(res, internal_error(exception.what()));
        return;
    }

    const Json payload = make_systemone_response_json(
        request, systemone_executed_model(public_model_id_, request.temperature), answers, usage);
    if (lifetime) {
        set_owned_json_content(res, payload.dump(), lifetime);
    } else {
        res.set_content(payload.dump(), "application/json");
    }
}

} // namespace ninfer::serve
