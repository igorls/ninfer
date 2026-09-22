#include "serve/typesafe_systemone.h"
#include "serve/http_server.h"

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

template <typename Func>
int check_422_rejection(Func&& func, const std::string& label) {
    try {
        func();
        std::cerr << "FAIL (expected ApiException): " << label << '\n';
        return 1;
    } catch (const ApiException& e) {
        if (e.error().status != 422) {
            std::cerr << "FAIL (expected status 422, got " << e.error().status << "): " << label
                      << '\n';
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL (unexpected exception type: " << e.what() << "): " << label << '\n';
        return 1;
    }
}

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
    }

    // 2. Mathematics: Choice & Score confidence formula
    // max(0.0, min(1.0, (N * max_p - 1.0) / (N - 1.0)))
    {
        failures += check(close_to(calculate_choice_confidence({}), 0.0), "confidence empty is 0.0");
        failures +=
            check(close_to(calculate_choice_confidence({0.7}), 1.0), "confidence single option");

        // Uniform distribution: confidence must be 0.0
        failures +=
            check(close_to(calculate_choice_confidence({1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}), 0.0),
                  "confidence uniform distribution is 0.0");

        // Certain distribution: confidence must be 1.0
        failures += check(close_to(calculate_choice_confidence({1.0, 0.0, 0.0}), 1.0),
                          "confidence 100% winner is 1.0");

        // N=3, max_p=0.88 -> (3 * 0.88 - 1) / 2 = (2.64 - 1) / 2 = 1.64 / 2 = 0.82
        failures += check(close_to(calculate_choice_confidence({0.88, 0.12, 0.0}), 0.82),
                          "confidence calculation for N=3, p=0.88");

        // N=2, max_p=0.9 -> (2 * 0.9 - 1) / 1 = 0.8
        failures += check(close_to(calculate_choice_confidence({0.9, 0.1}), 0.8),
                          "confidence calculation for N=2, p=0.9");
    }

    // 3. Mathematics: Expected Score formula
    // sum(i * P_i) for i = 0 .. N-1
    {
        // 3 levels: [0.0, 0.95, 0.05] -> 0*0 + 1*0.95 + 2*0.05 = 1.05
        failures += check(close_to(calculate_expected_score({0.0, 0.95, 0.05}), 1.05),
                          "expected score 3 levels [0, 0.95, 0.05] == 1.05");

        // 2 levels: [1.0, 0.0] -> 0.0
        failures += check(close_to(calculate_expected_score({1.0, 0.0}), 0.0),
                          "expected score level 0 == 0.0");

        // 2 levels: [0.0, 1.0] -> 1.0
        failures += check(close_to(calculate_expected_score({0.0, 1.0}), 1.0),
                          "expected score level 1 == 1.0");

        // 4 levels uniform: [0.25, 0.25, 0.25, 0.25] -> 0*0.25 + 1*0.25 + 2*0.25 + 3*0.25 = 1.5
        failures += check(close_to(calculate_expected_score({0.25, 0.25, 0.25, 0.25}), 1.5),
                          "expected score uniform == 1.5");
    }

    // 4. Token indexing for choice candidates
    {
        failures += check(choice_token_for_index(0) == "A", "token index 0 is A");
        failures += check(choice_token_for_index(1) == "B", "token index 1 is B");
        failures += check(choice_token_for_index(25) == "Z", "token index 25 is Z");
        failures += check(choice_token_for_index(26) == "a", "token index 26 is a");
        failures += check(choice_token_for_index(51) == "z", "token index 51 is z");
        failures += check(choice_token_for_index(52) == "0", "token index 52 is 0");
        failures += check(choice_token_for_index(61) == "9", "token index 61 is 9");

        failures += check(choice_token_for_index(62).empty(), "token index 62 is past the alphabet");

        // Across the 62 letter and digit indices there must be no collisions.
        std::vector<std::string> all_tokens;
        all_tokens.reserve(kMaximumSystemOneChoices);
        for (std::size_t idx = 0; idx < kMaximumSystemOneChoices; ++idx) {
            all_tokens.push_back(choice_token_for_index(idx));
        }
        std::vector<std::string> unique_tokens = all_tokens;
        std::sort(unique_tokens.begin(), unique_tokens.end());
        unique_tokens.erase(std::unique(unique_tokens.begin(), unique_tokens.end()),
                            unique_tokens.end());
        failures += check(unique_tokens.size() == kMaximumSystemOneChoices,
                          "all 62 choice candidate tokens are strictly unique");
    }

    {
        failures += check(systemone_billed_input_tokens(400, 0, true) == 400,
                          "first question bills its whole prompt");
        failures += check(systemone_billed_input_tokens(420, 380, false) == 40,
                          "later question bills only tokens past the shared state");
        failures += check(systemone_billed_input_tokens(100, 100, false) == 0,
                          "a fully cached later question adds no input tokens");
        failures += check(systemone_billed_input_tokens(50, 80, false) == 0,
                          "a cache hit larger than the prompt does not bill a negative");
    }

    // 5. Valid Request Parsing: Noul
    {
        const Json req_json = {
            {"model", "jev-latest"},
            {"state", "Help! My payouts have been failing for 3 days."},
            {"questions",
             {{"is_urgent",
               {{"type", "noul"},
                {"instructions", "Does this convey urgency?"},
                {"criteria",
                 {{"true", "Explicitly time-sensitive"}, {"false", "No urgency expressed"}}}}}}}};
        const SystemOneRequest req = parse_systemone_request(req_json);
        failures += check(req.requested_model == "jev-latest", "requested model preserved");
        failures += check(req.state_text == "Help! My payouts have been failing for 3 days.",
                          "state text parsed");
        failures += check(req.questions.size() == 1, "1 question parsed");
        failures += check(req.questions[0].id == "is_urgent", "question id is is_urgent");
        failures +=
            check(req.questions[0].type == SystemOneQuestionType::Noul, "question type is Noul");
        failures += check(req.questions[0].instructions == "Does this convey urgency?",
                          "instructions parsed");
        failures += check(req.questions[0].true_criteria == "Explicitly time-sensitive",
                          "true criteria parsed");
        failures += check(req.questions[0].false_criteria == "No urgency expressed",
                          "false criteria parsed");
        failures += check(req.questions[0].candidates == std::vector<std::string>({"Yes", "No"}),
                          "noul candidates are Yes and No");

        const std::string prompt = build_question_prompt(req.questions[0]);
        failures += check(prompt.find("Does this convey urgency?") != std::string::npos,
                          "prompt has question");
        failures += check(prompt.find("Explicitly time-sensitive") != std::string::npos,
                          "prompt has true criteria");
        failures +=
            check(prompt.find("Yes or No") != std::string::npos, "prompt asks for Yes or No");
    }

    // 6. Valid Request Parsing: Choice
    {
        const Json req_json        = {{"model", "qwen3.8-27b"},
                                      {"state", Json{{"ticket", 1234}, {"status", "open"}}},
                                      {"questions",
                                       {{"department",
                                         {{"type", "choice"},
                                          {"instructions", "Which team should handle this?"},
                                          {"criteria",
                                           {{"billing", "Payments, invoicing, refunds"},
                                            {"technical", "Bugs, outages, integrations"},
                                            {"sales", nullptr}}}}}}}};
        const SystemOneRequest req = parse_systemone_request(req_json);
        failures += check(req.requested_model == "qwen3.8-27b", "requested model qwen3.8-27b");
        failures +=
            check(req.state_text.find("ticket") != std::string::npos, "state json formatted");
        failures += check(req.questions.size() == 1, "1 question parsed");
        failures +=
            check(req.questions[0].type == SystemOneQuestionType::Choice, "question type Choice");
        failures += check(req.questions[0].choice_options.size() == 3, "3 options parsed");
        failures +=
            check(req.questions[0].choice_options[0].key == "billing", "opt 0 key is billing");
        failures += check(req.questions[0].choice_options[0].token == "A", "opt 0 token is A");
        failures +=
            check(req.questions[0].choice_options[1].key == "technical", "opt 1 key is technical");
        failures += check(req.questions[0].choice_options[1].token == "B", "opt 1 token is B");
        failures += check(req.questions[0].choice_options[2].key == "sales", "opt 2 key is sales");
        failures +=
            check(req.questions[0].choice_options[2].description.empty(), "opt 2 desc is empty");
        failures += check(req.questions[0].choice_options[2].token == "C", "opt 2 token is C");
        failures += check(req.questions[0].candidates == std::vector<std::string>({"A", "B", "C"}),
                          "choice candidates are A, B, C");

        const std::string prompt = build_question_prompt(req.questions[0]);
        failures +=
            check(prompt.find("- A: [billing]") != std::string::npos, "prompt has billing option");
        failures += check(prompt.find("- B: [technical]") != std::string::npos,
                          "prompt has technical option");
        failures +=
            check(prompt.find("- C: [sales]") != std::string::npos, "prompt has sales option");
    }

    // 7. Valid Request Parsing: Score
    {
        const Json req_json = {
            {"model", "jev-preview"},
            {"state", Json::array({"log 1", "log 2"})},
            {"questions",
             {{"frustration",
               {{"type", "score"},
                {"instructions", "How frustrated is the customer?"},
                {"criteria", Json::array({"Calm", "Frustrated", "Very angry"})}}}}}};
        const SystemOneRequest req = parse_systemone_request(req_json);
        failures += check(req.requested_model == "jev-preview", "requested model jev-preview");
        failures +=
            check(req.state_text.find("log 1") != std::string::npos, "array state formatted");
        failures += check(req.questions.size() == 1, "1 question parsed");
        failures +=
            check(req.questions[0].type == SystemOneQuestionType::Score, "question type Score");
        failures += check(req.questions[0].score_levels.size() == 3, "3 score levels");
        failures += check(req.questions[0].score_levels[0] == "Calm", "level 0 Calm");
        failures += check(req.questions[0].score_levels[1] == "Frustrated", "level 1 Frustrated");
        failures += check(req.questions[0].score_levels[2] == "Very angry", "level 2 Very angry");
        failures += check(req.questions[0].candidates == std::vector<std::string>({"0", "1", "2"}),
                          "score candidates are 0, 1, 2");

        const std::string prompt = build_question_prompt(req.questions[0]);
        failures += check(prompt.find("- 0: Calm") != std::string::npos, "prompt has level 0");
        failures +=
            check(prompt.find("- 1: Frustrated") != std::string::npos, "prompt has level 1");
        failures +=
            check(prompt.find("- 2: Very angry") != std::string::npos, "prompt has level 2");
    }

    // 8. Valid Request Parsing: Mixed questions in one request
    {
        const Json req_json = {
            {"model", "jev-1.13.0"},
            {"state", "Customer message here"},
            {"questions",
             {{"q_noul", {{"type", "noul"}, {"instructions", "Urgent?"}}},
              {"q_choice",
               {{"type", "choice"},
                {"instructions", "Action?"},
                {"criteria", {{"escalate", "Escalate to manager"}, {"resolve", "Close ticket"}}}}},
              {"q_score",
               {{"type", "score"},
                {"instructions", "Satisfaction rating?"},
                {"criteria", Json::array({"Poor", "Average", "Good", "Excellent"})}}}}}};
        const SystemOneRequest req = parse_systemone_request(req_json);
        failures += check(req.questions.size() == 3, "3 questions parsed in mixed request");
        failures += check(req.questions[0].type == SystemOneQuestionType::Noul, "mixed q0 Noul");
        failures +=
            check(req.questions[1].type == SystemOneQuestionType::Choice, "mixed q1 Choice");
        failures += check(req.questions[2].type == SystemOneQuestionType::Score, "mixed q2 Score");
    }

    // 9. Single token keys used directly for Choice
    {
        const Json req_json = {
            {"model", "jev-latest"},
            {"state", "Pick an option"},
            {"questions",
             {{"opt_select",
               {{"type", "choice"},
                {"instructions", "Select option A or B"},
                {"criteria", {{"A", "Option A description"}, {"B", "Option B description"}}}}}}}};
        const SystemOneRequest req = parse_systemone_request(req_json);
        failures += check(req.questions.size() == 1, "direct key choice parsed");
        failures += check(req.questions[0].choice_options.size() == 2, "2 choice options");
        failures += check(req.questions[0].choice_options[0].key == "A" &&
                              req.questions[0].choice_options[0].token == "A",
                          "opt A token is A directly");
        failures += check(req.questions[0].choice_options[1].key == "B" &&
                              req.questions[0].choice_options[1].token == "B",
                          "opt B token is B directly");
        failures += check(req.questions[0].candidates == std::vector<std::string>({"A", "B"}),
                          "candidates are A and B directly");

        const std::string prompt = build_question_prompt(req.questions[0]);
        failures += check(prompt.find("- A: Option A description") != std::string::npos,
                          "prompt has clean - A: desc without brackets");
    }

    // 10. Structured instructions (object and array) and structured criteria
    {
        const Json structured_inst = {
            {"potential_duplicate",
             {{"name", "John Smith"}, {"location", "Oakland, California"}}},
            {"question", "Is the resume for the same person as potential_duplicate?"}};
        const Json req_json = {
            {"model", "qwen3.8-27b-t1.5"},
            {"state", "Resume text here"},
            {"questions",
             {{"dup_check",
               {{"type", "noul"},
                {"instructions", structured_inst},
                {"criteria",
                 {{"true", Json{{"status", "match"}, {"action", "merge"}}},
                  {"false", Json{{"status", "no_match"}, {"action", "create_new"}}}}}}}}}};
        const SystemOneRequest req = parse_systemone_request(req_json);
        failures += check(close_to(req.temperature, 1.5), "temperature parsed from model suffix");
        failures += check(req.questions.size() == 1, "structured question parsed");
        failures += check(req.questions[0].instructions.find("potential_duplicate") !=
                              std::string::npos,
                          "object instructions dumped in question");
        failures += check(req.questions[0].true_criteria.find("merge") != std::string::npos,
                          "object true criteria dumped");
        failures += check(req.questions[0].false_criteria.find("create_new") != std::string::npos,
                          "object false criteria dumped");
    }

    // 11. Validation Rejections (HTTP 422 Unprocessable Entity)
    {
        // Empty question ID
        failures += check_422_rejection(
            [] {
                parse_systemone_request(Json{
                    {"model", "jev-latest"},
                    {"state", "test"},
                    {"questions", {{"" /* empty id */, {{"type", "noul"}, {"instructions", "abc"}}}}}});
            },
            "empty question id 422");

        // Temperature non-positive
        failures += check_422_rejection(
            [] {
                parse_systemone_request(Json{
                    {"model", "jev-latest"},
                    {"state", "test"},
                    {"temperature", -0.5},
                    {"questions", {{"q1", {{"type", "noul"}, {"instructions", "abc"}}}}}});
            },
            "negative temperature 422");

        // Missing state
        failures += check_422_rejection(
            [] {
                parse_systemone_request(
                    Json{{"model", "jev-latest"}, {"questions", Json::object()}});
            },
            "missing state 422");

        // State wrong type
        failures += check_422_rejection(
            [] {
                parse_systemone_request(
                    Json{{"model", "jev-latest"}, {"state", 123}, {"questions", Json::object()}});
            },
            "state number 422");

        // Missing model
        failures += check_422_rejection(
            [] { parse_systemone_request(Json{{"state", "test"}, {"questions", Json::object()}}); },
            "missing model 422");

        // Empty model
        failures += check_422_rejection(
            [] {
                parse_systemone_request(
                    Json{{"model", ""}, {"state", "test"}, {"questions", Json::object()}});
            },
            "empty model 422");

        // Missing questions
        failures += check_422_rejection(
            [] { parse_systemone_request(Json{{"model", "jev-latest"}, {"state", "test"}}); },
            "missing questions 422");

        // Empty questions
        failures += check_422_rejection(
            [] {
                parse_systemone_request(Json{
                    {"model", "jev-latest"}, {"state", "test"}, {"questions", Json::object()}});
            },
            "empty questions 422");

        // Questions not an object
        failures += check_422_rejection(
            [] {
                parse_systemone_request(
                    Json{{"model", "jev-latest"}, {"state", "test"}, {"questions", Json::array()}});
            },
            "questions array 422");

        // Unknown question type
        failures += check_422_rejection(
            [] {
                parse_systemone_request(
                    Json{{"model", "jev-latest"},
                         {"state", "test"},
                         {"questions", {{"q1", {{"type", "magic"}, {"instructions", "abc"}}}}}});
            },
            "unknown question type 422");

        // Missing instructions
        failures += check_422_rejection(
            [] {
                parse_systemone_request(Json{{"model", "jev-latest"},
                                             {"state", "test"},
                                             {"questions", {{"q1", {{"type", "noul"}}}}}});
            },
            "missing instructions 422");

        // Choice missing criteria
        failures += check_422_rejection(
            [] {
                parse_systemone_request(
                    Json{{"model", "jev-latest"},
                         {"state", "test"},
                         {"questions", {{"q1", {{"type", "choice"}, {"instructions", "abc"}}}}}});
            },
            "choice missing criteria 422");

        // Choice empty criteria
        failures += check_422_rejection(
            [] {
                parse_systemone_request(Json{{"model", "jev-latest"},
                                             {"state", "test"},
                                             {"questions",
                                              {{"q1",
                                                {{"type", "choice"},
                                                 {"instructions", "abc"},
                                                 {"criteria", Json::object()}}}}}});
            },
            "choice empty criteria 422");

        failures += check_422_rejection(
            [] {
                Json criteria = Json::object();
                for (int i = 0; i < 63; ++i) { criteria["opt" + std::to_string(i)] = "d"; }
                parse_systemone_request(Json{
                    {"model", "jev-latest"},
                    {"state", "test"},
                    {"questions",
                     {{"q1",
                       {{"type", "choice"},
                        {"instructions", "abc"},
                        {"criteria", std::move(criteria)}}}}}});
            },
            "choice 63 options 422");

        // Score criteria not array
        failures += check_422_rejection(
            [] {
                parse_systemone_request(Json{
                    {"model", "jev-latest"},
                    {"state", "test"},
                    {"questions",
                     {{"q1",
                       {{"type", "score"}, {"instructions", "abc"}, {"criteria", "invalid"}}}}}});
            },
            "score criteria not array 422");

        // Score criteria with 1 level (< 2)
        failures += check_422_rejection(
            [] {
                parse_systemone_request(Json{{"model", "jev-latest"},
                                             {"state", "test"},
                                             {"questions",
                                              {{"q1",
                                                {{"type", "score"},
                                                 {"instructions", "abc"},
                                                 {"criteria", Json::array({"one"})}}}}}});
            },
            "score criteria 1 level 422");

        // Score criteria with 11 levels (> 10)
        failures += check_422_rejection(
            [] {
                Json levels = Json::array();
                for (int i = 0; i < 11; ++i) { levels.push_back("lvl" + std::to_string(i)); }
                parse_systemone_request(
                    Json{{"model", "jev-latest"},
                         {"state", "test"},
                         {"questions",
                          {{"q1",
                            {{"type", "score"}, {"instructions", "abc"}, {"criteria", levels}}}}}});
            },
            "score criteria 11 levels 422");
    }

    // 10. Response JSON Serialization
    {
        SystemOneResponse response;
        response.model         = "jev-1.13.0";
        response.input_tokens  = 318;
        response.output_tokens = 34;

        // Noul answer
        SystemOneAnswer noul_ans;
        noul_ans.id   = "is_urgent";
        noul_ans.type = SystemOneQuestionType::Noul;
        noul_ans.noul = 0.95;
        response.answers.push_back(std::move(noul_ans));

        // Choice answer
        SystemOneAnswer choice_ans;
        choice_ans.id                   = "department";
        choice_ans.type                 = SystemOneQuestionType::Choice;
        choice_ans.choice_winner        = "billing";
        choice_ans.choice_probabilities = {{"billing", 0.88}, {"technical", 0.12}, {"sales", 0.0}};
        choice_ans.choice_confidence    = 0.81;
        response.answers.push_back(std::move(choice_ans));

        // Score answer
        SystemOneAnswer score_ans;
        score_ans.id                  = "frustration";
        score_ans.type                = SystemOneQuestionType::Score;
        score_ans.score               = 1.05;
        score_ans.score_legend        = {{"0", "Calm"}, {"1", "Frustrated"}, {"2", "Very angry"}};
        score_ans.score_probabilities = {{"0", 0.0}, {"1", 0.95}, {"2", 0.05}};
        score_ans.score_confidence    = 0.92;
        response.answers.push_back(std::move(score_ans));

        const Json out_json = make_systemone_response_json(response);
        failures += check(out_json.at("model") == "jev-1.13.0", "response model");
        failures += check(out_json.at("usage").at("input_tokens") == 318, "usage input_tokens");
        failures += check(out_json.at("usage").at("output_tokens") == 34, "usage output_tokens");

        const auto& answers = out_json.at("answers");
        failures += check(answers.size() == 3, "3 answers in response");

        // Check Noul
        const auto& noul_obj = answers.at("is_urgent");
        failures += check(noul_obj.at("type") == "noul", "noul type");
        failures += check(close_to(noul_obj.at("noul").get<double>(), 0.95), "noul value");
        failures += check(!noul_obj.contains("confidence"), "noul has no confidence field");

        // Check Choice
        const auto& choice_obj = answers.at("department");
        failures += check(choice_obj.at("type") == "choice", "choice type");
        failures += check(choice_obj.at("choice") == "billing", "choice winner");
        failures +=
            check(close_to(choice_obj.at("confidence").get<double>(), 0.81), "choice confidence");
        failures +=
            check(close_to(choice_obj.at("probabilities").at("billing").get<double>(), 0.88),
                  "billing prob");
        failures +=
            check(close_to(choice_obj.at("probabilities").at("technical").get<double>(), 0.12),
                  "technical prob");
        failures += check(close_to(choice_obj.at("probabilities").at("sales").get<double>(), 0.0),
                          "sales prob");

        // Check Score
        const auto& score_obj = answers.at("frustration");
        failures += check(score_obj.at("type") == "score", "score type");
        failures +=
            check(close_to(score_obj.at("score").get<double>(), 1.05), "score expected value");
        failures +=
            check(close_to(score_obj.at("confidence").get<double>(), 0.92), "score confidence");
        failures += check(score_obj.at("legend").at("0") == "Calm", "legend 0");
        failures += check(score_obj.at("legend").at("1") == "Frustrated", "legend 1");
        failures += check(score_obj.at("legend").at("2") == "Very angry", "legend 2");
        failures +=
            check(close_to(score_obj.at("probabilities").at("1").get<double>(), 0.95), "prob 1");
    }

    // 11. Error serialization: write_typesafe_error
    {
        httplib::Response res;
        ApiError err;
        err.status  = 422;
        err.message = "criteria for choice cannot exceed 62 options";
        err.param   = "questions.department.criteria";
        err.type    = "invalid_request_error";
        write_typesafe_error(res, err);

        failures += check(res.status == 422, "write_typesafe_error status 422");
        const Json err_json = Json::parse(res.body);
        failures += check(err_json.at("error").at("message") ==
                              "criteria for choice cannot exceed 62 options",
                          "error message preserved");
        failures += check(err_json.at("error").at("param") == "questions.department.criteria",
                          "error param preserved");
    }

    if (failures == 0) {
        std::cout << "PASS: all typesafe schema tests succeeded\n";
    } else {
        std::cerr << "FAIL: " << failures << " typesafe schema test checks failed\n";
    }

    return failures == 0 ? 0 : 1;
}
