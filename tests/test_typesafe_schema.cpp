#include "serve/typesafe_systemone.h"
#include "serve/http_server.h"
#include "serve/openai_chat.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <iostream>
#include <limits>
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

bool close_to(double a, double b, double eps = 1e-5) { return std::fabs(a - b) < eps; }

} // namespace

int main() {
    int failures = 0;

    // 1. Mathematics: Softmax normalized probabilities
    {
        const auto empty_p = softmax_probabilities({});
        failures += check(empty_p.empty(), "softmax empty input");

        const auto single_p = softmax_probabilities({2.5});
        failures +=
            check(single_p.size() == 1 && close_to(single_p[0], 1.0), "softmax single candidate");

        const auto equal_p = softmax_probabilities({1.0, 1.0, 1.0});
        failures += check(equal_p.size() == 3 && close_to(equal_p[0], 1.0 / 3.0) &&
                              close_to(equal_p[1], 1.0 / 3.0) && close_to(equal_p[2], 1.0 / 3.0),
                          "softmax equal logprobs");

        // Extreme difference (numerical stability, no overflow/NaN)
        const auto extreme_p = softmax_probabilities({1000.0, -1000.0});
        failures += check(extreme_p.size() == 2 && close_to(extreme_p[0], 1.0) &&
                              close_to(extreme_p[1], 0.0),
                          "softmax extreme logprobs");

        // Sum to 1.0 check
        const auto multi_p = softmax_probabilities({-0.1, -1.2, -3.4, -0.5});
        double sum         = 0.0;
        for (double p : multi_p) { sum += p; }
        failures += check(close_to(sum, 1.0), "softmax probabilities sum to 1.0");

        // Independent formula on moderate represented inputs, evaluated in long double.
        const std::vector<double> oracle_lps{-2.0, -3.5, -6.0};
        for (const double temperature : {0.25, 1.0, 4.0}) {
            long double denominator = 0.0L;
            for (const double lp : oracle_lps) {
                denominator += std::exp(static_cast<long double>(lp) / temperature);
            }
            const auto actual = softmax_probabilities(oracle_lps, temperature);
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const long double expected =
                    std::exp(static_cast<long double>(oracle_lps[i]) / temperature) / denominator;
                failures += check(close_to(actual[i], static_cast<double>(expected), 1e-12),
                                  "temperature softmax agrees with independent formula");
            }
        }
        // Reproduced by a real Qwen3.8 decision: scaling negative scores before applying the
        // old -9900 floor incorrectly turned a confident B into a uniform/A result.
        const auto cold = softmax_probabilities({-8.0, -0.2}, 1e-6);
        failures += check(cold[0] == 0.0 && cold[1] == 1.0,
                          "low temperature preserves the actual winner");
        const auto tiny = softmax_probabilities({-8.0, -0.2},
                                                std::numeric_limits<double>::denorm_min());
        failures += check(tiny[0] == 0.0 && tiny[1] == 1.0,
                          "smallest positive temperature remains well-defined");
        const auto shifted = softmax_probabilities({-10008.0, -10000.2}, 0.25);
        const auto unshifted = softmax_probabilities({-8.0, -0.2}, 0.25);
        failures += check(close_to(shifted[0], unshifted[0], 1e-12) &&
                              close_to(shifted[1], unshifted[1], 1e-12),
                          "finite scores remain valid below the old sentinel threshold");
        const double masked = -std::numeric_limits<double>::infinity();
        const auto mixed = softmax_probabilities({masked, -4.0, -2.0}, 10000.0);
        failures += check(mixed[0] == 0.0 && mixed[2] > mixed[1],
                          "unavailable candidates stay zero at high temperature");
        const auto unavailable = softmax_probabilities({masked, masked}, 0.1);
        failures += check(unavailable[0] == 0.5 && unavailable[1] == 0.5,
                          "all unavailable candidates retain uniform fallback");
    }

    // 2. Confidence. Choice: 1 - (1 - max) / (1 - 1/n). Score: 1 - E|level - mode| divided by the
    // same distance for a uniform distribution, floor(n^2/4)/n. Expected values are TypeSafe's
    // published documentation examples.
    {
        failures += check(close_to(calculate_choice_confidence({}), 0.0), "choice confidence empty");
        failures += check(close_to(calculate_choice_confidence({0.7}), 1.0), "choice single option");
        failures += check(close_to(calculate_choice_confidence({1.0 / 3, 1.0 / 3, 1.0 / 3}), 0.0),
                          "choice uniform is 0");
        failures += check(close_to(calculate_choice_confidence({0.61, 0.35, 0.04}), 0.415),
                          "choice doc example 0.61 of 3");
        failures += check(close_to(calculate_choice_confidence({0.34, 0.40, 0.02, 0.24}), 0.2),
                          "choice doc example 0.40 of 4");
        failures += check(close_to(calculate_choice_confidence({0.0, 0.74, 0.0, 0.0, 0.26}), 0.675),
                          "choice doc example 0.74 of 5");

        failures += check(close_to(calculate_score_confidence({0.0, 0.57, 0.43}), 0.355),
                          "score doc example 3 levels split");
        failures += check(close_to(calculate_score_confidence({0.0, 0.74, 0.26}), 0.61),
                          "score doc example 3 levels");
        failures += check(close_to(calculate_score_confidence({0.0, 0.0, 0.48, 0.52}), 0.52),
                          "score doc example 4 levels credits the adjacent level");
        failures += check(close_to(calculate_score_confidence({0.0, 0.14, 0.86, 0.0, 0.0}),
                                   1.0 - 0.14 / 1.2),
                          "score doc example 5 levels");
        failures += check(close_to(calculate_score_confidence({0.0, 0.0, 0.0, 1.0}), 1.0),
                          "score certain level");
        failures += check(close_to(calculate_score_confidence({0.25, 0.25, 0.25, 0.25}), 0.0),
                          "score uniform is 0");
        failures += check(close_to(calculate_score_confidence({1.0}), 1.0), "score single level");
        failures += check(close_to(calculate_score_confidence({0.5, 0.0, 0.0, 0.0, 0.5}), 0.0),
                          "score split between the ends clamps to 0");
        // Ties take the first level: 0.47/0.47/0.06 measures distance from level 0.
        failures += check(
            close_to(calculate_score_confidence({0.47, 0.47, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0}),
                     1.0 - (0.47 + 2 * 0.06) / 2.0),
            "score tie measured from the first mode");

        failures += check(close_to(calculate_expected_score({0.0, 0.95, 0.05}), 1.05),
                          "expected score");
        failures += check(close_to(calculate_expected_score({0.25, 0.25, 0.25, 0.25}), 1.5),
                          "expected score uniform");
    }

    // 3. Labels
    {
        failures += check(choice_label_for_index(0, false) == "A" &&
                              choice_label_for_index(25, false) == "Z" &&
                              choice_label_for_index(26, false) == "a" &&
                              choice_label_for_index(52, false) == "0" &&
                              choice_label_for_index(61, false) == "9" &&
                              choice_label_for_index(62, false).empty(),
                          "single-token labels A-Z, a-z, 0-9");
        failures += check(choice_label_for_index(0, true) == "A0" &&
                              choice_label_for_index(9, true) == "A9" &&
                              choice_label_for_index(10, true) == "B0" &&
                              choice_label_for_index(254, true) == "Z4",
                          "two-token labels letter then digit");
        failures += check(systemone_billed_input_tokens(400, 0, true) == 400 &&
                              systemone_billed_input_tokens(420, 380, false) == 40 &&
                              systemone_billed_input_tokens(50, 80, false) == 0,
                          "state billed once, later branches bill fresh tokens");
        failures += check(systemone_executed_model("qwen3.8-27b", 1.0) == "qwen3.8-27b" &&
                              systemone_executed_model("qwen3.8-27b", 1.5) == "qwen3.8-27b-t1.5",
                          "answers report the executed model and its temperature");
    }

    const auto plus = [](Json body, const char* key, Json value) {
        body[key] = std::move(value);
        return body;
    };
    const auto request_with = [](Json questions) {
        return Json{{"model", "jev-latest"}, {"state", "Help! My payouts have been failing."},
                    {"questions", std::move(questions)}};
    };

    // 4. Prompts for the request shapes production clients send are fixed text: a change moves
    // every calibrated threshold.
    {
        const SystemOneRequest request = parse_systemone_request(request_with(Json{
            {"inicial", {{"type", "noul"},
                         {"instructions", "Is this the initial petition?"},
                         {"criteria", {{"true", "Petition"}, {"false", "Other document"}}}}},
            {"department", {{"type", "choice"},
                            {"instructions", "Which team?"},
                            {"criteria", {{"billing", "Payments"}, {"technical", nullptr}}}}},
            {"move", {{"type", "choice"},
                      {"instructions", "Pick"},
                      {"criteria", {{"A", "left"}, {"B", "right"}, {"C", nullptr}}}}},
            {"frustration", {{"type", "score"},
                             {"instructions", "How frustrated?"},
                             {"criteria", Json::array({"Calm", "Angry"})}}}}));
        failures += check(build_question_prompt(request.questions[0]) ==
                              "Question: Is this the initial petition?\n\nCriteria:\n"
                              "- Yes: Petition\n- No: Other document\n\n"
                              "Answer with only Yes or No:\n",
                          "noul prompt text unchanged");
        failures += check(build_question_prompt(request.questions[1]) ==
                              "Question: Which team?\n\nOptions:\n- A: [billing] Payments\n"
                              "- B: [technical]\n\n"
                              "Select the best option. Answer with only the option letter:\n",
                          "lettered choice prompt text unchanged");
        failures += check(build_question_prompt(request.questions[2]) ==
                              "Question: Pick\n\nOptions:\n- A: left\n- B: right\n- C\n\n"
                              "Select the best option. Answer with only the option letter:\n",
                          "verbatim single-character keys unchanged");
        failures += check(build_question_prompt(request.questions[3]) ==
                              "Question: How frustrated?\n\nRating levels:\n- 0: Calm\n"
                              "- 1: Angry\n\nSelect the rating level that best applies. "
                              "Answer with only the level number:\n",
                          "score prompt text unchanged");
        failures += check(first_token_candidates(request.questions[0]) ==
                                  std::vector<std::string>{"Yes", "No"} &&
                              first_token_candidates(request.questions[1]) ==
                                  std::vector<std::string>{"A", "B"} &&
                              first_token_candidates(request.questions[3]) ==
                                  std::vector<std::string>{"0", "1"},
                          "first-token candidates");
    }

    // 5. Inputs Jev accepts that earlier builds rejected or rendered wrongly.
    {
        const SystemOneRequest request = parse_systemone_request(request_with(Json{
            {"nulls", {{"type", "noul"},
                       {"instructions", "Urgent?"},
                       {"criteria", {{"true", nullptr}, {"false", nullptr}}}}},
            {"criteria_only", {{"type", "noul"}, {"criteria", {{"true", "Time-sensitive"}}}}},
            {"bare_choice", {{"type", "choice"}, {"criteria", {{"x", nullptr}, {"y", nullptr}}}}},
            {"one_level", {{"type", "score"}, {"criteria", Json::array({"Only"})}}},
            {"structured_levels",
             {{"type", "score"},
              {"instructions", "Rate"},
              {"criteria", Json::array({"Calm", Json{{"level", "Upset"}}, Json::array({"a", "b"})})}}},
            {"extra_field", {{"type", "noul"}, {"instructions", "Q?"}, {"foo", 1}}}}));
        const std::string nulls = build_question_prompt(request.questions[0]);
        failures += check(nulls.find("null") == std::string::npos &&
                              nulls.find("Criteria") == std::string::npos,
                          "null noul criteria are undescribed");
        failures += check(build_question_prompt(request.questions[1])
                              .starts_with("Question: Based on the criteria below"),
                          "criteria-only noul gets a default question");
        failures += check(build_question_prompt(request.questions[2])
                              .starts_with("Question: Which option best fits the state?"),
                          "choice without instructions");
        failures += check(request.questions[3].score_levels.size() == 1,
                          "one-level score accepted");
        failures += check(request.questions[4].score_legend[1] == Json{{"level", "Upset"}} &&
                              request.questions[4].score_legend[2] == Json::array({"a", "b"}),
                          "structured levels kept for the legend");
    }

    // 6. Large choices: up to 62 options take one token each; 63 to 255 take a letter and a digit.
    {
        const auto options = [](std::size_t n) {
            Json criteria = Json::object();
            for (std::size_t i = 0; i < n; ++i) { criteria["o" + std::to_string(i)] = nullptr; }
            return criteria;
        };
        const auto question_for = [&](std::size_t n) {
            return parse_systemone_request(request_with(Json{
                {"d", {{"type", "choice"}, {"instructions", "Which?"}, {"criteria", options(n)}}}}))
                .questions[0];
        };
        const SystemOneQuestion q62 = question_for(62);
        failures += check(!q62.two_token_labels && q62.choice_options[61].label == "9",
                          "62 options stay single-token");
        const SystemOneQuestion q63 = question_for(63);
        failures += check(q63.two_token_labels && q63.choice_options[0].label == "A0" &&
                              q63.choice_options[62].label == "G2",
                          "63 options use letter-digit labels");
        failures += check(first_token_candidates(q63) ==
                              std::vector<std::string>{"A", "B", "C", "D", "E", "F", "G"},
                          "two-token first position reads the label letters");
        const std::string prompt = build_question_prompt(q63);
        failures += check(prompt.find("- A0: [o0]\n") != std::string::npos &&
                              prompt.find("only the option code") != std::string::npos,
                          "two-token prompt");
        const SystemOneQuestion q255 = question_for(255);
        failures += check(q255.choice_options[254].label == "Z4" &&
                              first_token_candidates(q255).size() == 26,
                          "255 options span A0 to Z4");
    }

    // 7. Rejections in TypeSafe's shapes and statuses.
    {
        const auto rejected = [&](const Json& body, int status, const std::string& label) {
            try {
                (void)parse_systemone_request(body);
            } catch (const SystemOneError& error) {
                if (error.status() == status) { return error.detail(); }
                std::cerr << "FAIL (status " << error.status() << "): " << label << '\n';
                ++failures;
                return Json();
            }
            std::cerr << "FAIL (accepted): " << label << '\n';
            ++failures;
            return Json();
        };
        const Json noul{{"type", "noul"}, {"instructions", "Urgent?"}};

        Json detail = rejected(Json{{"model", "jev-latest"}, {"questions", {{"n", noul}}}}, 422,
                               "missing state");
        failures += check(detail.is_array() && detail[0]["type"] == "missing" &&
                              detail[0]["loc"] == Json::array({"body", "state"}),
                          "missing state is a pydantic error list");
        detail = rejected(Json{{"model", "jev-latest"}, {"state", 5}, {"questions", {{"n", noul}}}},
                          422, "numeric state");
        failures += check(detail.size() == 3 && detail[0]["type"] == "string_type" &&
                              detail[0]["loc"] == Json::array({"body", "state", "str"}),
                          "content union reports each member");
        detail = rejected(Json{{"state", "s"}, {"questions", Json::object()}}, 422,
                          "missing model and empty questions");
        failures += check(detail.size() == 2 && detail[1]["type"] == "too_short",
                          "every schema violation is listed");
        detail = rejected(request_with(Json{{"n", {{"instructions", "Q?"}}}}), 422, "missing type");
        failures += check(detail[0]["type"] == "union_tag_not_found", "missing type tag");
        detail = rejected(request_with(Json{{"n", {{"type", "noul"}, {"instructions", 5}}}}), 422,
                          "numeric instructions");
        failures += check(detail[0]["loc"] ==
                              Json::array({"body", "questions", "n", "noul", "instructions", "str"}),
                          "question errors carry the type in their location");
        detail = rejected(request_with(Json{{"s", {{"type", "score"},
                                                   {"instructions", "Q?"},
                                                   {"criteria", Json::array({"a", nullptr})}}}}),
                          422, "null score level");
        failures += check(detail[0]["loc"] ==
                              Json::array({"body", "questions", "s", "score", "criteria", 1, "str"}),
                          "null score level");
        rejected(request_with(Json{{"s", {{"type", "score"}, {"criteria", Json::array()}}}}), 422,
                 "zero score levels");
        rejected(request_with(Json{{"n", {{"type", "noul"}, {"criteria", "yes"}}}}), 422,
                 "string noul criteria");
        rejected(request_with(Json{{"d", {{"type", "choice"}, {"criteria", Json::array()}}}}), 422,
                 "array choice criteria");

        failures += check(rejected(plus(request_with(Json{{"n", noul}}), "stream", true),
                                   400, "stream") == Json{{"error_type", "api_usage_error"},
                                                          {"message", "Invalid request."}},
                          "unknown top-level fields are a usage error");
        rejected(request_with(Json{{"n", {{"type", "binary"}, {"instructions", "Q?"}}}}), 400,
                 "unknown type");
        rejected(Json{{"model", ""}, {"state", "s"}, {"questions", {{"n", noul}}}}, 400,
                 "empty model");
        failures += check(rejected(request_with(Json{{"", noul}}), 400, "empty id") ==
                              "Question key cannot be empty.",
                          "empty question id");
        failures += check(rejected(request_with(Json{{"n", {{"type", "noul"}, {"instructions", ""}}}}),
                                   400, "noul without content") ==
                              "Noul question must have criteria or instructions: n",
                          "noul needs instructions or criteria");
        failures += check(rejected(request_with(Json{{"d", {{"type", "choice"},
                                                            {"criteria", Json::object()}}}}),
                                   400, "empty choice") ==
                              "Choice question must have at least one choice: d",
                          "empty choice criteria");
        Json criteria = Json::object();
        for (int i = 0; i < 256; ++i) { criteria["o" + std::to_string(i)] = nullptr; }
        failures += check(rejected(request_with(Json{{"d", {{"type", "choice"},
                                                            {"criteria", criteria}}}}),
                                   400, "256 options") ==
                              "Too many choices. Must have at most 255 choices.",
                          "choice option limit");
        Json levels = Json::array();
        for (int i = 0; i < 11; ++i) { levels.push_back("level"); }
        failures += check(rejected(request_with(Json{{"s", {{"type", "score"}, {"criteria", levels}}}}),
                                   400, "11 levels") ==
                              "Too many score levels. Must have at most 10 levels.",
                          "score level limit");
        for (const double invalid : {-0.5, std::numeric_limits<double>::infinity()}) {
            rejected(plus(request_with(Json{{"n", noul}}), "temperature", invalid), 400,
                     "invalid temperature");
        }
        rejected(Json{{"model", "qwen3.8-27b-tinf"}, {"state", "s"}, {"questions", {{"n", noul}}}},
                 400, "infinite temperature suffix");
        failures += check(close_to(parse_systemone_request(
                                       Json{{"model", "qwen3.8-27b-t1.5"}, {"state", "s"},
                                            {"questions", {{"n", noul}}}})
                                       .temperature,
                                   1.5) &&
                              close_to(parse_systemone_request(
                                           Json{{"model", "gpt-turbo"}, {"state", "s"},
                                                {"questions", {{"n", noul}}}})
                                           .temperature,
                                       1.0),
                          "only a numeric -t suffix sets the temperature");

        try {
            (void)parse_systemone_body("{not json");
            failures += check(false, "invalid JSON rejected");
        } catch (const SystemOneError& error) {
            failures += check(error.status() == 422 && error.detail()[0]["type"] == "json_invalid",
                              "invalid JSON is 422 json_invalid");
        }
        try {
            (void)parse_systemone_body("");
            failures += check(false, "empty body rejected");
        } catch (const SystemOneError& error) {
            failures += check(error.status() == 422 && error.detail()[0]["type"] == "missing",
                              "empty body is 422 missing");
        }
    }

    // 8. Answers in TypeSafe's shape.
    {
        const SystemOneRequest request = parse_systemone_request(request_with(Json{
            {"is_urgent", {{"type", "noul"}, {"instructions", "Urgent?"}}},
            {"department", {{"type", "choice"},
                            {"instructions", "Which team?"},
                            {"criteria", {{"billing", nullptr}, {"technical", nullptr}, {"sales", nullptr}}}}},
            {"frustration", {{"type", "score"},
                             {"instructions", "How frustrated?"},
                             {"criteria", Json::array({"Calm", Json{{"level", "Frustrated"}}, "Very angry"})}}}}));
        std::vector<SystemOneAnswer> answers(3);
        answers[0].probabilities = {0.95};
        answers[1].probabilities = {0.12, 0.88, 0.0};
        answers[2].probabilities = {0.0, 0.95, 0.05};
        const Json out = make_systemone_response_json(request, "qwen3.8-27b", answers,
                                                      SystemOneUsage{.input_tokens = 318});
        failures += check(out["model"] == "qwen3.8-27b" && out["usage"]["input_tokens"] == 318 &&
                              out["usage"]["output_tokens"] == 0 &&
                              !out["usage"].contains("vision_tokens"),
                          "model and usage");
        failures += check(out["answers"]["is_urgent"] == Json{{"type", "noul"}, {"noul", 0.95}},
                          "noul answer");
        const Json& choice = out["answers"]["department"];
        std::vector<std::string> keys;
        for (auto field = choice.begin(); field != choice.end(); ++field) { keys.push_back(field.key()); }
        failures += check(keys == std::vector<std::string>{"type", "choice", "confidence", "probabilities"},
                          "choice fields in Jev's order");
        failures += check(choice["choice"] == "technical" &&
                              close_to(choice["confidence"].get<double>(), 0.82) &&
                              close_to(choice["probabilities"]["technical"].get<double>(), 0.88),
                          "choice answer");
        const Json& score = out["answers"]["frustration"];
        failures += check(close_to(score["score"].get<double>(), 1.05) &&
                              close_to(score["confidence"].get<double>(), 0.925) &&
                              score["legend"]["1"] == Json{{"level", "Frustrated"}} &&
                              score["legend"]["2"] == "Very angry",
                          "score answer returns levels as sent");
        const Json with_vision = make_systemone_response_json(
            request, "qwen3.8-27b", answers, SystemOneUsage{.input_tokens = 1, .vision_tokens = 80});
        failures += check(with_vision["usage"]["vision_tokens"] == 80, "vision usage evidence");
    }

    // 9. Transport errors, auth and model discovery in TypeSafe's shape.
    {
        ApiError context;
        context.status  = 400;
        context.code    = "context_length_exceeded";
        context.message = "prompt exceeds the context";
        failures += check(systemone_error_from(context).detail() ==
                              Json{{"error_type", "max_tokens_exceeded"}},
                          "context overflow is max_tokens_exceeded");
        ApiError overloaded;
        overloaded.status  = 429;
        overloaded.message = "inference request queue is full";
        const SystemOneError busy = systemone_error_from(overloaded);
        failures += check(busy.status() == 429 && busy.detail()["error_type"] == "rate_limit_error",
                          "queue overflow keeps 429 for SDK retries");
        failures += check(systemone_missing_api_key().status() == 403 &&
                              systemone_invalid_api_key().status() == 401,
                          "missing key 403, wrong key 401");

        httplib::Response response;
        write_typesafe_failure(response, systemone_method_not_allowed());
        failures += check(response.status == 405 && response.get_header_value("Allow") == "POST" &&
                              Json::parse(response.body) == Json{{"detail", "Method Not Allowed"}},
                          "405 body");

        const Json models = systemone_model_entries("qwen3.8-27b", 1790430686);
        failures += check(models.size() == 3 && models[0]["name"] == "qwen3.8-27b" &&
                              models[1]["name"] == "jev-latest" &&
                              models[0]["release_date"] == "2026-09-26",
                          "TypeSafe model entries");
    }

    // 10. Image observations traverse the same multimodal adapter as native chat.
    {
        Json body{{"model", "qwen3.8-27b"}, {"state", Json{{"task", "observe"}}},
                  {"images", Json::array({"https://example.test/a.png", "data:image/png;base64,AA=="})},
                  {"questions", {{"move", {{"type", "choice"}, {"instructions", "Choose a direction"},
                    {"criteria", {{"A", "left"}, {"B", "right"}}}}}}}};
        const auto request = parse_systemone_request(body);
        auto messages = build_systemone_messages(request);
        messages.push_back(Json{{"role", "user"}, {"content", build_question_prompt(request.questions[0])}});
        const auto chat = parse_chat_completion_request(Json{{"model", "qwen3.8-27b"}, {"messages", messages}}, RequestLimits{});
        failures += check(chat.generation.media_item_count() == 2 &&
            chat.generation.messages[1].content[0].kind == ContentKind::Image &&
            chat.generation.messages[1].content[1].kind == ContentKind::Image,
            "System One images become Engine image inputs");
        failures += check(request.state_text == body["state"].dump(2), "JSON state remains text");
        for (const auto& invalid : std::vector<Json>{nullptr, "image.png", Json::array({42}), Json::array({""})}) {
            body["images"] = invalid;
            try {
                (void)parse_systemone_request(body);
                failures += check(false, "invalid images rejected");
            } catch (const SystemOneError& error) {
                failures += check(error.status() == 400, "invalid images are a usage error");
            }
        }
        body["images"] = Json::array();
        failures += check(build_systemone_messages(parse_systemone_request(body)).size() == 1, "empty images preserves text route");
        body.erase("images");
        failures += check(build_systemone_messages(parse_systemone_request(body)).size() == 1, "omitted images preserves text route");
    }

    if (failures == 0) {
        std::cout << "PASS: all typesafe schema tests succeeded\n";
    } else {
        std::cerr << "FAIL: " << failures << " typesafe schema test checks failed\n";
    }

    return failures == 0 ? 0 : 1;
}
