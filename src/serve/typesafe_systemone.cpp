#include "serve/typesafe_systemone.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace ninfer::serve {
namespace {

using Json = SystemOneJson;

Json with(Json loc, Json item) {
    loc.push_back(std::move(item));
    return loc;
}

// TypeSafe content: a string, an object or an array (JSONContent in its schema).
bool is_content(const Json& value) {
    return value.is_string() || value.is_object() || value.is_array();
}

std::string content_text(const Json& value) {
    if (value.is_null()) { return {}; }
    return value.is_string() ? value.get<std::string>() : value.dump(2);
}

const Json* member(const Json& object, const char* name) {
    const auto found = object.find(name);
    return found == object.end() || found->is_null() ? nullptr : &*found;
}

// Validation errors in pydantic's shape, collected over the whole body like FastAPI reports them.
class SchemaErrors {
public:
    void add(const char* type, Json loc, const char* msg, const Json& input, Json ctx = nullptr) {
        Json error{{"type", type}, {"loc", std::move(loc)}, {"msg", msg}, {"input", input}};
        if (!ctx.is_null()) { error["ctx"] = std::move(ctx); }
        errors_.push_back(std::move(error));
    }

    void missing(const Json& loc, const Json& input) {
        add("missing", loc, "Field required", input);
    }

    // A JSONContent union reports one failed member each: string, object, array.
    void content(const Json& loc, const Json& input) {
        add("string_type", with(loc, "str"), "Input should be a valid string", input);
        add("dict_type", with(loc, "dict[any,any]"), "Input should be a valid dictionary", input);
        add("list_type", with(loc, "list[any]"), "Input should be a valid list", input);
    }

    void too_short(const Json& loc, const char* kind, std::size_t actual, const Json& input) {
        const std::string msg = std::string(kind) + " should have at least 1 item after " +
                                "validation, not " + std::to_string(actual);
        Json error{{"type", "too_short"},
                   {"loc", loc},
                   {"msg", msg},
                   {"input", input},
                   {"ctx", Json{{"field_type", kind}, {"min_length", 1}, {"actual_length", actual}}}};
        errors_.push_back(std::move(error));
    }

    void raise_if_any() const {
        if (!errors_.empty()) { throw SystemOneError(422, errors_); }
    }

private:
    Json errors_ = Json::array();
};

void validate_optional_content(SchemaErrors& errors, const Json& object, const char* name,
                               const Json& loc) {
    const Json* value = member(object, name);
    if (value != nullptr && !is_content(*value)) { errors.content(with(loc, name), *value); }
}

// Schema checks for one question. Returns false for an unknown type tag, which Jev rejects as a
// usage error rather than a validation error.
bool validate_question_schema(SchemaErrors& errors, const std::string& id, const Json& question) {
    const Json loc = Json::array({"body", "questions", id});
    if (!question.is_object()) {
        errors.add("model_attributes_type", loc,
                   "Input should be a valid dictionary or object to extract fields from", question);
        return true;
    }
    const Json* type = member(question, "type");
    if (type == nullptr) {
        errors.add("union_tag_not_found", loc, "Unable to extract tag using discriminator 'type'",
                   question, Json{{"discriminator", "'type'"}});
        return true;
    }
    if (!type->is_string()) { return false; }
    const std::string& tag = type->get_ref<const std::string&>();
    if (tag != "noul" && tag != "choice" && tag != "score") { return false; }

    const Json typed = with(loc, tag);
    validate_optional_content(errors, question, "instructions", typed);
    if (tag == "noul") {
        const Json* criteria = member(question, "criteria");
        if (criteria == nullptr) { return true; }
        if (!criteria->is_object()) {
            errors.add("model_attributes_type", with(typed, "criteria"),
                       "Input should be a valid dictionary or object to extract fields from",
                       *criteria);
            return true;
        }
        validate_optional_content(errors, *criteria, "true", with(typed, "criteria"));
        validate_optional_content(errors, *criteria, "false", with(typed, "criteria"));
        return true;
    }

    const Json criteria_loc = with(typed, "criteria");
    const auto found        = question.find("criteria");
    if (found == question.end()) {
        errors.missing(criteria_loc, question);
        return true;
    }
    const Json& criteria = *found;
    if (tag == "choice") {
        if (!criteria.is_object()) {
            errors.add("dict_type", criteria_loc, "Input should be a valid dictionary", criteria);
            return true;
        }
        for (auto option = criteria.begin(); option != criteria.end(); ++option) {
            if (!option->is_null() && !is_content(*option)) {
                errors.content(with(criteria_loc, option.key()), *option);
            }
        }
        return true;
    }
    if (!criteria.is_array()) {
        errors.add("list_type", criteria_loc, "Input should be a valid list", criteria);
        return true;
    }
    if (criteria.empty()) {
        errors.too_short(criteria_loc, "List", 0, criteria);
        return true;
    }
    for (std::size_t level = 0; level < criteria.size(); ++level) {
        if (!is_content(criteria[level])) { errors.content(with(criteria_loc, level), criteria[level]); }
    }
    return true;
}

// A `-t<T>` model-name suffix scales the candidate logits by 1/T when `temperature` is absent. Only
// a suffix that is entirely a number counts, so names such as "gpt-turbo" are left alone.
std::optional<double> temperature_suffix(const std::string& model) {
    const std::size_t position = model.rfind("-t");
    if (position == std::string::npos || position + 2 >= model.size()) { return std::nullopt; }
    const std::string text = model.substr(position + 2);
    std::size_t consumed   = 0;
    double value           = 0.0;
    try {
        value = std::stod(text, &consumed);
    } catch (const std::exception&) { return std::nullopt; }
    if (consumed != text.size()) { return std::nullopt; }
    return value;
}

bool single_printable_character(const std::string& key) {
    return key.size() == 1 && static_cast<unsigned char>(key[0]) > 32 &&
           static_cast<unsigned char>(key[0]) < 127;
}

SystemOneQuestion build_question(const std::string& id, const Json& value) {
    SystemOneQuestion question;
    question.id           = id;
    const std::string tag = value.at("type").get<std::string>();
    if (const Json* instructions = member(value, "instructions")) {
        question.instructions = content_text(*instructions);
    }

    if (tag == "noul") {
        question.type = SystemOneQuestionType::Noul;
        if (const Json* criteria = member(value, "criteria")) {
            if (const Json* yes = member(*criteria, "true")) {
                question.true_criteria = content_text(*yes);
            }
            if (const Json* no = member(*criteria, "false")) {
                question.false_criteria = content_text(*no);
            }
        }
        if (question.instructions.empty() && question.true_criteria.empty() &&
            question.false_criteria.empty()) {
            systemone_rule_error("Noul question must have criteria or instructions: " + id);
        }
        return question;
    }

    const Json& criteria = value.at("criteria");
    if (tag == "choice") {
        question.type = SystemOneQuestionType::Choice;
        if (criteria.empty()) {
            systemone_rule_error("Choice question must have at least one choice: " + id);
        }
        if (criteria.size() > kMaximumSystemOneChoices) {
            systemone_rule_error("Too many choices. Must have at most 255 choices.");
        }
        bool verbatim = true;
        for (auto option = criteria.begin(); option != criteria.end(); ++option) {
            verbatim = verbatim && single_printable_character(option.key());
        }
        question.two_token_labels = !verbatim && criteria.size() > kSingleTokenChoiceLabels;
        std::size_t index         = 0;
        for (auto option = criteria.begin(); option != criteria.end(); ++option, ++index) {
            question.choice_options.push_back(SystemOneChoiceOption{
                .key         = option.key(),
                .description = content_text(*option),
                .label = verbatim ? option.key()
                                  : choice_label_for_index(index, question.two_token_labels),
            });
        }
        return question;
    }

    question.type = SystemOneQuestionType::Score;
    if (criteria.size() > kMaximumScoreLevels) {
        systemone_rule_error("Too many score levels. Must have at most 10 levels.");
    }
    for (const Json& level : criteria) {
        question.score_levels.push_back(content_text(level));
        question.score_legend.push_back(level);
    }
    return question;
}

std::string iso_date(std::int64_t unix_seconds) {
    const auto days = std::chrono::floor<std::chrono::days>(
        std::chrono::sys_seconds{std::chrono::seconds{unix_seconds}});
    const std::chrono::year_month_day date{days};
    char text[16];
    std::snprintf(text, sizeof(text), "%04d-%02u-%02u", static_cast<int>(date.year()),
                  static_cast<unsigned>(date.month()), static_cast<unsigned>(date.day()));
    return text;
}

} // namespace

SystemOneError::SystemOneError(int status, SystemOneJson detail)
    : status_(status), detail_(std::move(detail)), message_(detail_.dump()) {}

void systemone_rule_error(std::string message) { throw SystemOneError(400, std::move(message)); }

void systemone_usage_error(std::string message) {
    throw SystemOneError(400, Json{{"error_type", "api_usage_error"}, {"message", std::move(message)}});
}

SystemOneError systemone_error_from(const ApiError& error) {
    if (error.code == "context_length_exceeded") {
        return SystemOneError(400, Json{{"error_type", "max_tokens_exceeded"}});
    }
    if (error.status == 401) { return systemone_invalid_api_key(); }
    const char* type = error.status == 429  ? "rate_limit_error"
                       : error.status >= 500 ? "api_error"
                                             : "api_usage_error";
    return SystemOneError(error.status, Json{{"error_type", type}, {"message", error.message}});
}

SystemOneError systemone_missing_api_key() {
    return SystemOneError(
        403, Json{{"error_type", "authentication_error"},
                  {"message", "Must supply an API key! Check your request and try again."}});
}

SystemOneError systemone_invalid_api_key() {
    return SystemOneError(401, Json{{"error_type", "authentication_error"},
                                    {"message", "Cannot authenticate with the server. Please check "
                                                "your API key and try again."}});
}

SystemOneError systemone_method_not_allowed() { return SystemOneError(405, "Method Not Allowed"); }

SystemOneJson parse_systemone_body(std::string_view text) {
    if (text.empty()) {
        throw SystemOneError(422, Json::array({Json{{"type", "missing"},
                                                    {"loc", Json::array({"body"})},
                                                    {"msg", "Field required"},
                                                    {"input", nullptr}}}));
    }
    try {
        return Json::parse(text);
    } catch (const Json::parse_error& error) {
        throw SystemOneError(
            422, Json::array({Json{{"type", "json_invalid"},
                                   {"loc", Json::array({"body", error.byte})},
                                   {"msg", "JSON decode error"},
                                   {"input", Json::object()},
                                   {"ctx", Json{{"error", error.what()}}}}}));
    }
}

SystemOneRequest parse_systemone_request(const SystemOneJson& body) {
    SchemaErrors errors;
    const Json root = Json::array({"body"});
    if (!body.is_object()) {
        errors.add("model_attributes_type", root,
                   "Input should be a valid dictionary or object to extract fields from", body);
        errors.raise_if_any();
    }

    const Json* state = member(body, "state");
    if (state == nullptr) {
        errors.missing(with(root, "state"), body);
    } else if (!is_content(*state)) {
        errors.content(with(root, "state"), *state);
    }
    const Json* model = member(body, "model");
    if (model == nullptr) {
        errors.missing(with(root, "model"), body);
    } else if (!model->is_string()) {
        errors.add("string_type", with(root, "model"), "Input should be a valid string", *model);
    }
    bool known_types     = true;
    const Json* questions = member(body, "questions");
    if (questions == nullptr) {
        errors.missing(with(root, "questions"), body);
    } else if (!questions->is_object()) {
        errors.add("dict_type", with(root, "questions"), "Input should be a valid dictionary",
                   *questions);
    } else if (questions->empty()) {
        errors.too_short(with(root, "questions"), "Dictionary", 0, *questions);
    } else {
        for (auto entry = questions->begin(); entry != questions->end(); ++entry) {
            known_types = validate_question_schema(errors, entry.key(), *entry) && known_types;
        }
    }

    // Jev answers fields outside its schema, and question types it does not know, with a generic
    // usage error. `temperature` and `images` are this server's extensions.
    bool known_fields = true;
    for (auto field = body.begin(); field != body.end(); ++field) {
        const std::string& key = field.key();
        known_fields = known_fields && (key == "state" || key == "model" || key == "questions" ||
                                        key == "temperature" || key == "images");
    }
    if (!known_fields || !known_types) { systemone_usage_error("Invalid request."); }
    errors.raise_if_any();

    const std::string& model_name = model->get_ref<const std::string&>();
    if (model_name.empty()) { systemone_usage_error("Unknown model: "); }

    SystemOneRequest request;
    request.state_text = content_text(*state);

    if (const Json* temperature = member(body, "temperature")) {
        if (!temperature->is_number()) {
            systemone_usage_error("temperature must be a finite positive number");
        }
        request.temperature = temperature->get<double>();
    } else if (const std::optional<double> suffix = temperature_suffix(model_name)) {
        request.temperature = *suffix;
    }
    if (!std::isfinite(request.temperature) || request.temperature <= 0.0) {
        systemone_usage_error("temperature must be a finite positive number");
    }

    if (const auto images = body.find("images"); images != body.end()) {
        if (!images->is_array()) {
            systemone_usage_error("images must be an array of image URLs or data URIs");
        }
        for (const Json& image : *images) {
            if (!image.is_string() || image.get_ref<const std::string&>().empty()) {
                systemone_usage_error("each image must be a nonempty URL or data URI");
            }
            request.images.push_back(image.get<std::string>());
        }
    }

    if (questions->size() > kMaximumSystemOneQuestions) {
        throw SystemOneError(400, Json{{"error_type", "max_tokens_exceeded"}});
    }
    for (auto entry = questions->begin(); entry != questions->end(); ++entry) {
        if (entry.key().empty()) { systemone_rule_error("Question key cannot be empty."); }
        request.questions.push_back(build_question(entry.key(), *entry));
    }
    return request;
}

std::string systemone_executed_model(std::string_view served_model_id, double temperature) {
    std::string model(served_model_id);
    if (temperature != 1.0) {
        char text[32];
        std::snprintf(text, sizeof(text), "%g", temperature);
        model += "-t";
        model += text;
    }
    return model;
}

std::string choice_label_for_index(std::size_t index, bool two_token_labels) {
    if (two_token_labels) {
        if (index >= 260) { return {}; }
        return std::string{static_cast<char>('A' + index / 10), static_cast<char>('0' + index % 10)};
    }
    if (index < 26) { return std::string(1, static_cast<char>('A' + index)); }
    if (index < 52) { return std::string(1, static_cast<char>('a' + (index - 26))); }
    if (index < kSingleTokenChoiceLabels) {
        return std::string(1, static_cast<char>('0' + (index - 52)));
    }
    return {};
}

std::vector<std::string> first_token_candidates(const SystemOneQuestion& question) {
    std::vector<std::string> candidates;
    switch (question.type) {
    case SystemOneQuestionType::Noul:
        candidates = {"Yes", "No"};
        break;
    case SystemOneQuestionType::Choice:
        for (const SystemOneChoiceOption& option : question.choice_options) {
            std::string first = question.two_token_labels ? option.label.substr(0, 1) : option.label;
            if (candidates.empty() || candidates.back() != first) {
                candidates.push_back(std::move(first));
            }
        }
        break;
    case SystemOneQuestionType::Score:
        for (std::size_t level = 0; level < question.score_levels.size(); ++level) {
            candidates.push_back(std::to_string(level));
        }
        break;
    }
    return candidates;
}

nlohmann::ordered_json build_systemone_messages(const SystemOneRequest& request) {
    Json messages = Json::array(
        {Json{{"role", "system"},
              {"content", "You are an evaluation assistant. State to evaluate:\n" +
                              request.state_text}}});
    if (!request.images.empty()) {
        Json content = Json::array();
        for (const auto& image : request.images) {
            content.push_back(Json{{"type", "image_url"}, {"image_url", Json{{"url", image}}}});
        }
        messages.push_back(Json{{"role", "user"}, {"content", std::move(content)}});
    }
    return messages;
}

std::string build_question_prompt(const SystemOneQuestion& question) {
    std::string prompt;
    switch (question.type) {
    case SystemOneQuestionType::Noul: {
        prompt += "Question: " +
                  (question.instructions.empty()
                       ? std::string("Based on the criteria below, is the answer Yes or No?")
                       : question.instructions) +
                  "\n";
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
        prompt += "Question: " +
                  (question.instructions.empty() ? std::string("Which option best fits the state?")
                                                 : question.instructions) +
                  "\n\nOptions:\n";
        for (const SystemOneChoiceOption& option : question.choice_options) {
            prompt += "- " + option.label;
            if (option.key != option.label) {
                prompt += ": [" + option.key + "]";
                if (!option.description.empty()) { prompt += " " + option.description; }
            } else if (!option.description.empty()) {
                prompt += ": " + option.description;
            }
            prompt += "\n";
        }
        if (question.two_token_labels) {
            prompt += "\nSelect the best option. Answer with only the option code:\n";
        } else if (question.choice_options.size() <= 26) {
            prompt += "\nSelect the best option. Answer with only the option letter:\n";
        } else {
            prompt += "\nSelect the best option. Answer with only the option letter or digit:\n";
        }
        break;
    }
    case SystemOneQuestionType::Score: {
        prompt += "Question: " +
                  (question.instructions.empty()
                       ? std::string("Which rating level best fits the state?")
                       : question.instructions) +
                  "\n\nRating levels:\n";
        for (std::size_t level = 0; level < question.score_levels.size(); ++level) {
            prompt += "- " + std::to_string(level) + ": " + question.score_levels[level] + "\n";
        }
        prompt +=
            "\nSelect the rating level that best applies. Answer with only the level number:\n";
        break;
    }
    }
    return prompt;
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

std::vector<double> softmax_probabilities(const std::vector<double>& logprobs,
                                          double temperature) {
    if (logprobs.empty()) { return {}; }
    if (logprobs.size() == 1) { return {1.0}; }
    double max_lp = -std::numeric_limits<double>::infinity();
    for (const double lp : logprobs) {
        if (std::isfinite(lp)) { max_lp = std::max(max_lp, lp); }
    }
    std::vector<double> probabilities(logprobs.size(), 0.0);
    double sum = 0.0;
    for (std::size_t i = 0; i < logprobs.size(); ++i) {
        if (std::isfinite(logprobs[i])) {
            // Center before temperature scaling: finite scores must never be mistaken for a
            // masked-value sentinel at low temperatures, or overflow before subtraction.
            probabilities[i] = std::exp((logprobs[i] - max_lp) / temperature);
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
    // Expected 0/1 distance from the top option is 1 - max; for a uniform distribution 1 - 1/n.
    const double max_p = *std::max_element(probabilities.begin(), probabilities.end());
    const double conf  = (static_cast<double>(n) * max_p - 1.0) / (static_cast<double>(n) - 1.0);
    return std::clamp(conf, 0.0, 1.0);
}

double calculate_score_confidence(const std::vector<double>& probabilities) {
    const std::size_t n = probabilities.size();
    if (n == 0) { return 0.0; }
    if (n == 1) { return 1.0; }
    const auto mode = static_cast<std::size_t>(
        std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin());
    double spread = 0.0;
    for (std::size_t level = 0; level < n; ++level) {
        const double distance = level > mode ? static_cast<double>(level - mode)
                                             : static_cast<double>(mode - level);
        spread += probabilities[level] * distance;
    }
    // A uniform distribution's smallest mean distance to one level, floor(n^2 / 4) / n.
    const double uniform = static_cast<double>((n * n) / 4) / static_cast<double>(n);
    return std::clamp(1.0 - spread / uniform, 0.0, 1.0);
}

double calculate_expected_score(const std::vector<double>& probabilities) {
    double score = 0.0;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        score += static_cast<double>(i) * probabilities[i];
    }
    return score;
}

nlohmann::ordered_json make_systemone_response_json(const SystemOneRequest& request,
                                                    const std::string& model,
                                                    const std::vector<SystemOneAnswer>& answers,
                                                    const SystemOneUsage& usage) {
    Json answer_map = Json::object();
    for (std::size_t q = 0; q < request.questions.size(); ++q) {
        const SystemOneQuestion& question = request.questions[q];
        const std::vector<double>& p      = answers.at(q).probabilities;
        switch (question.type) {
        case SystemOneQuestionType::Noul:
            answer_map[question.id] = Json{{"type", "noul"}, {"noul", p.empty() ? 0.0 : p[0]}};
            break;
        case SystemOneQuestionType::Choice: {
            const auto winner = static_cast<std::size_t>(std::max_element(p.begin(), p.end()) - p.begin());
            Json probabilities = Json::object();
            for (std::size_t i = 0; i < question.choice_options.size(); ++i) {
                probabilities[question.choice_options[i].key] = p.at(i);
            }
            answer_map[question.id] = Json{{"type", "choice"},
                                           {"choice", question.choice_options.at(winner).key},
                                           {"confidence", calculate_choice_confidence(p)},
                                           {"probabilities", std::move(probabilities)}};
            break;
        }
        case SystemOneQuestionType::Score: {
            Json legend        = Json::object();
            Json probabilities = Json::object();
            for (std::size_t level = 0; level < question.score_legend.size(); ++level) {
                legend[std::to_string(level)]        = question.score_legend[level];
                probabilities[std::to_string(level)] = p.at(level);
            }
            answer_map[question.id] = Json{{"type", "score"},
                                           {"score", calculate_expected_score(p)},
                                           {"confidence", calculate_score_confidence(p)},
                                           {"legend", std::move(legend)},
                                           {"probabilities", std::move(probabilities)}};
            break;
        }
        }
    }
    Json result{{"model", model},
                {"answers", std::move(answer_map)},
                {"usage", Json{{"input_tokens", usage.input_tokens}, {"output_tokens", 0}}}};
    if (usage.vision_tokens != 0) { result["usage"]["vision_tokens"] = usage.vision_tokens; }
    return result;
}

nlohmann::ordered_json systemone_model_entries(std::string_view served_model_id,
                                               std::int64_t loaded_unix_seconds) {
    const std::string served(served_model_id);
    const std::string date = iso_date(loaded_unix_seconds);
    Json models            = Json::array();
    models.push_back(Json{{"name", served},
                          {"description", "System One decisions from " + served +
                                              "'s next-token probabilities"},
                          {"release_date", date}});
    for (const char* alias : {"jev-latest", "jev-preview"}) {
        models.push_back(Json{{"name", alias},
                              {"description", "TypeSafe alias; answered by " + served},
                              {"release_date", date}});
    }
    return models;
}

} // namespace ninfer::serve
