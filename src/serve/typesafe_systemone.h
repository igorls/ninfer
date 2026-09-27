#pragma once

#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// TypeSafe System One (`POST /v1/systemone`) served from next-token probabilities. The request,
// answer and error shapes follow TypeSafe's published contract (openapi.json 0.2.0) and the
// behaviour of its hosted jev-1.13 service, so a client written for Jev runs unchanged.

using SystemOneJson = nlohmann::ordered_json;

// TypeSafe accepts 1 to 255 options per Choice and 1 to 10 levels per Score. It bounds the number
// of questions only by tokens; here every question is its own branch, so a count bound keeps one
// call from occupying the Engine indefinitely. The bound sits above what Jev's 64k-token budget
// admits (about 4,900 one-line questions).
constexpr std::size_t kMaximumSystemOneQuestions = 8192;
constexpr std::size_t kMaximumSystemOneChoices   = 255;
constexpr std::size_t kMaximumScoreLevels        = 10;
// Options up to this count are labelled A-Z, a-z, 0-9: one native token each. Larger sets use a
// letter and a digit (A0 ... Z9), whose log-probability is log P(letter) + log P(digit | letter).
constexpr std::size_t kSingleTokenChoiceLabels = 62;

// A rejection in TypeSafe's wire shape: the HTTP status Jev returns and FastAPI's `detail`, which
// is a list of validation errors (422), a rule message (400), or an `error_type` object.
class SystemOneError : public std::exception {
public:
    SystemOneError(int status, SystemOneJson detail);

    [[nodiscard]] int status() const noexcept { return status_; }
    [[nodiscard]] const SystemOneJson& detail() const noexcept { return detail_; }
    [[nodiscard]] SystemOneJson body() const { return SystemOneJson{{"detail", detail_}}; }
    [[nodiscard]] const char* what() const noexcept override { return message_.c_str(); }

private:
    int status_ = 400;
    SystemOneJson detail_;
    std::string message_;
};

// 400 with a plain message, the form Jev uses for rule violations such as too many choices.
[[noreturn]] void systemone_rule_error(std::string message);
// 400 {"error_type": "api_usage_error", "message": ...}.
[[noreturn]] void systemone_usage_error(std::string message);
// Engine, transport and authentication failures carried as ApiError, in TypeSafe's shape.
[[nodiscard]] SystemOneError systemone_error_from(const ApiError& error);
[[nodiscard]] SystemOneError systemone_missing_api_key();
[[nodiscard]] SystemOneError systemone_invalid_api_key();
[[nodiscard]] SystemOneError systemone_method_not_allowed();

enum class SystemOneQuestionType : std::uint8_t {
    Noul,
    Choice,
    Score,
};

struct SystemOneChoiceOption {
    std::string key;
    std::string description; // prompt text; empty when the option is undescribed
    std::string label;       // "A" (one token) or "A0" (letter token, then digit token)
};

struct SystemOneQuestion {
    std::string id;
    SystemOneQuestionType type = SystemOneQuestionType::Noul;
    std::string instructions; // prompt text; empty when not given
    // Noul
    std::string true_criteria;
    std::string false_criteria;
    // Choice
    std::vector<SystemOneChoiceOption> choice_options;
    bool two_token_labels = false;
    // Score: prompt text per level, and the levels exactly as sent for the answer's legend.
    std::vector<std::string> score_levels;
    SystemOneJson score_legend = SystemOneJson::array();
};

struct SystemOneRequest {
    std::string state_text;
    // NInfer extension: observations acquired through the shared product media route.
    std::vector<std::string> images;
    double temperature = 1.0;
    std::vector<SystemOneQuestion> questions;
};

// Parses the JSON body text; malformed or empty bodies fail with Jev's 422 `json_invalid`/`missing`.
[[nodiscard]] SystemOneJson parse_systemone_body(std::string_view text);
// Validates a request against TypeSafe's schema (422, every violation listed) and rules (400).
// `temperature` and `images` are the only accepted fields beyond TypeSafe's own.
[[nodiscard]] SystemOneRequest parse_systemone_request(const SystemOneJson& body);
// The model identity an answer reports: the served id, with `-t<T>` when the candidate logits are
// scaled. Requested names (`jev-latest`, the served id, anything else) never change execution.
[[nodiscard]] std::string systemone_executed_model(std::string_view served_model_id,
                                                   double temperature);

[[nodiscard]] std::string choice_label_for_index(std::size_t index, bool two_token_labels);
// The tokens read at the first answer position: the option labels, the distinct label letters for
// two-token labels, "Yes"/"No" for a Noul and the level digits for a Score.
[[nodiscard]] std::vector<std::string> first_token_candidates(const SystemOneQuestion& question);

[[nodiscard]] nlohmann::ordered_json build_systemone_messages(const SystemOneRequest& request);
[[nodiscard]] std::string build_question_prompt(const SystemOneQuestion& question);

// The first question bills its whole prompt, which holds the shared state. A later branch bills
// only the tokens past a prefix-cache hit, so the state is not charged once per question.
[[nodiscard]] std::int64_t systemone_billed_input_tokens(int prompt_tokens,
                                                         std::uint32_t cached_tokens, bool first);
[[nodiscard]] std::vector<double> softmax_probabilities(const std::vector<double>& logprobs,
                                                        double temperature = 1.0);
// Confidence is one minus the expected distance from the most probable answer, relative to the
// same distance for a uniform distribution: the 0/1 distance between options for a Choice and the
// level distance |i - k| for a Score. Both reproduce every confidence jev-1.13 returns.
[[nodiscard]] double calculate_choice_confidence(const std::vector<double>& probabilities);
[[nodiscard]] double calculate_score_confidence(const std::vector<double>& probabilities);
[[nodiscard]] double calculate_expected_score(const std::vector<double>& probabilities);

struct SystemOneAnswer {
    // Per option (Choice) or level (Score) in request order; {P(yes)} for a Noul.
    std::vector<double> probabilities;
};

struct SystemOneUsage {
    std::int64_t input_tokens   = 0;
    std::uint64_t vision_tokens = 0;
};

[[nodiscard]] nlohmann::ordered_json
make_systemone_response_json(const SystemOneRequest& request, const std::string& model,
                             const std::vector<SystemOneAnswer>& answers,
                             const SystemOneUsage& usage);

// TypeSafe's `GET /v1/models` entries: the served id and the jev aliases a TypeSafe SDK sends.
[[nodiscard]] nlohmann::ordered_json systemone_model_entries(std::string_view served_model_id,
                                                             std::int64_t loaded_unix_seconds);

} // namespace ninfer::serve
