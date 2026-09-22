#pragma once

#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::serve {

constexpr std::size_t kMaximumSystemOneQuestions = 256;
// A–Z, a–z, and 0–9. Each mapped choice is one native token the model can actually emit.
constexpr std::size_t kMaximumSystemOneChoices   = 62;
constexpr std::size_t kMinimumScoreLevels        = 2;
constexpr std::size_t kMaximumScoreLevels        = 10;

enum class SystemOneQuestionType : std::uint8_t {
    Noul,
    Choice,
    Score,
};

struct SystemOneChoiceOption {
    std::string key;
    std::string description;
    std::string token;
};

struct SystemOneQuestion {
    std::string id;
    SystemOneQuestionType type = SystemOneQuestionType::Noul;
    std::string instructions;
    // Noul criteria
    std::string true_criteria;
    std::string false_criteria;
    // Choice criteria
    std::vector<SystemOneChoiceOption> choice_options;
    // Score criteria
    std::vector<std::string> score_levels;
    // Evaluated candidate tokens
    std::vector<std::string> candidates;
};

struct SystemOneRequest {
    std::string requested_model;
    std::string state_text;
    double temperature = 1.0;
    std::vector<SystemOneQuestion> questions;
};

struct SystemOneAnswer {
    std::string id;
    SystemOneQuestionType type = SystemOneQuestionType::Noul;
    // Noul
    double noul = 0.0;
    // Choice
    std::string choice_winner;
    std::vector<std::pair<std::string, double>> choice_probabilities;
    double choice_confidence = 0.0;
    // Score
    double score = 0.0;
    std::vector<std::pair<std::string, std::string>> score_legend;
    std::vector<std::pair<std::string, double>> score_probabilities;
    double score_confidence = 0.0;
};

struct SystemOneResponse {
    std::string model;
    std::vector<SystemOneAnswer> answers;
    std::int64_t input_tokens  = 0;
    std::int64_t output_tokens = 0;
};

[[noreturn]] void systemone_bad_request(std::string message, std::string param = {},
                                        std::string code = {});

std::string choice_token_for_index(std::size_t index);
// The first question bills its whole prompt, which holds the shared state. A later question
// bills only the tokens past a prefix-cache hit, so the state is not charged once per question.
std::int64_t systemone_billed_input_tokens(int prompt_tokens, std::uint32_t cached_tokens,
                                           bool first);
std::vector<double> softmax_probabilities(const std::vector<double>& logprobs);
double calculate_choice_confidence(const std::vector<double>& probabilities);
double calculate_expected_score(const std::vector<double>& probabilities);

SystemOneRequest parse_systemone_request(const nlohmann::ordered_json& body);
std::string build_question_prompt(const SystemOneQuestion& question);
nlohmann::ordered_json make_systemone_response_json(const SystemOneResponse& response);

} // namespace ninfer::serve
