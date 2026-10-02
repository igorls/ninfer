#pragma once

#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
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

// NInfer extension: how a question is decided beyond the single native readout. `Averaged` reads
// each question through three prompts (the native prompt, a lettered prompt with the option order
// rotated by one, and an evidence/criterion framing) and averages their renormalised
// distributions. A positive `reasoning_budget` escalates a question whose averaged distribution
// has normalised entropy above `escalate_entropy` to a bounded-thinking generation and mixes its
// answer in at `answer_weight`. Both apply to questions with at most kMaximumPolicyOptions
// options; other questions keep the native readout.
struct SystemOnePolicy {
    enum class Readout : std::uint8_t { Single, Averaged };
    Readout readout                = Readout::Single;
    std::uint32_t reasoning_budget = 0; // 0 disables escalation
    double escalate_entropy        = 0.66;
    double answer_weight           = 0.7;
    [[nodiscard]] bool active() const noexcept {
        return readout == Readout::Averaged || reasoning_budget != 0;
    }
};
constexpr std::size_t kMaximumPolicyOptions = 26;

struct SystemOneRequest {
    std::string state_text;
    // NInfer extension: observations acquired through the shared product media route.
    std::vector<std::string> images;
    double temperature = 1.0;
    std::vector<SystemOneQuestion> questions;
    SystemOnePolicy policy;
};

// Parses the JSON body text; malformed or empty bodies fail with Jev's 422 `json_invalid`/`missing`.
[[nodiscard]] SystemOneJson parse_systemone_body(std::string_view text);
// Validates a request against TypeSafe's schema (422, every violation listed) and rules (400).
// `temperature`, `images` and `policy` are the only accepted fields beyond TypeSafe's own; a
// request without `policy` takes `defaults`.
[[nodiscard]] SystemOneRequest parse_systemone_request(const SystemOneJson& body,
                                                       const SystemOnePolicy& defaults = {});
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

// Policy readouts. Options are listed in policy order: a Noul as `no`, `yes`; a Choice by key in
// request order; a Score by level index. Letter j names option (j + rotation) mod n.
struct SystemOneLetterPrompt {
    nlohmann::ordered_json messages;
    std::vector<std::string> letters; // candidate tokens in letter order
    std::vector<std::size_t> options; // option index of each letter
};
[[nodiscard]] std::size_t policy_option_count(const SystemOneQuestion& question);
// `framed` selects the evidence/criterion rendering; otherwise the lettered decision prompt.
[[nodiscard]] SystemOneLetterPrompt build_letter_prompt(const SystemOneRequest& request,
                                                        const SystemOneQuestion& question,
                                                        std::size_t rotation, bool framed);
// The native branch's distribution (P(Yes) first for a Noul) in policy option order.
[[nodiscard]] std::vector<double> native_to_option_order(const SystemOneQuestion& question,
                                                         std::vector<double> probabilities);
// Entropy divided by log(n); 0 for fewer than two options.
[[nodiscard]] double normalized_entropy(const std::vector<double>& probabilities);
[[nodiscard]] std::vector<double>
average_distributions(const std::vector<std::vector<double>>& members);
// The letter a bounded-reasoning answer names: exactly one declared letter, optionally followed
// by '.' or ')', surrounded by whitespace only.
[[nodiscard]] std::optional<std::size_t> parse_letter_answer(std::string_view text,
                                                             const std::vector<std::string>& letters);

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
    // Reported as `usage.policy` when a policy was active.
    bool policy                    = false;
    std::string readout            = "single";
    std::uint32_t escalated        = 0;
    std::uint32_t reasoning_tokens = 0;
};

[[nodiscard]] nlohmann::ordered_json
make_systemone_response_json(const SystemOneRequest& request, const std::string& model,
                             const std::vector<SystemOneAnswer>& answers,
                             const SystemOneUsage& usage);

// TypeSafe's `GET /v1/models` entries: the served id and the jev aliases a TypeSafe SDK sends.
[[nodiscard]] nlohmann::ordered_json systemone_model_entries(std::string_view served_model_id,
                                                             std::int64_t loaded_unix_seconds);

struct RequestLifetime;

// One in-process System One call: answers in question order, billed usage, and the first
// branch's lifetime so the HTTP response can outlive the handler's stack.
struct SystemOneExecution {
    std::vector<SystemOneAnswer> answers;
    SystemOneUsage usage;
    std::shared_ptr<RequestLifetime> lifetime;
};

} // namespace ninfer::serve
