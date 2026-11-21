#include "serve/http_server.h"

#include "serve/http_transport.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <utility>

// POST /v1/hidden_states: run one prefill pass on the resident model and return every prompt
// token's BF16 hidden row for 1..8 requested layers (`layer` or `layers`: -2 = post-prefix raw
// embedding, 0..N-1 = post-layer residual, -1 = final RMSNorm). Response is application/octet-stream:
//   [8-byte LE uint64 header_len][UTF-8 JSON metadata][seq_len * num_layers * hidden_size * 2 bytes LE BF16]

namespace ninfer::serve {
namespace {

using Json = nlohmann::ordered_json;

constexpr std::size_t kMaxHiddenTokenLayers = 32768ULL * ninfer::kMaximumHiddenStateTapLayers;

[[noreturn]] void hidden_bad_request(std::string message, std::string param,
                                     std::string code = {}) {
    ApiError error;
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

} // namespace

void HttpServer::handle_hidden_states(const httplib::Request& req, httplib::Response& res) {
    OpenAIChatRequest request;
    std::vector<std::int32_t> tap_layers;
    bool explicit_layers = false;
    try {
        const Json body = parse_json_body(req);
        if (!body.is_object()) { hidden_bad_request("request body must be an object", ""); }
        if (!body.contains("messages") || !body.at("messages").is_array() ||
            body.at("messages").empty()) {
            hidden_bad_request("messages must be a nonempty array", "messages");
        }
        const bool has_layer  = body.contains("layer") && !body.at("layer").is_null();
        const bool has_layers = body.contains("layers") && !body.at("layers").is_null();
        if (has_layer && has_layers) {
            hidden_bad_request("specify either layer or layers, not both", "layers");
        }
        auto parse_layer_id = [](const Json& val, const char* param) -> std::int32_t {
            if (!val.is_number_integer()) {
                hidden_bad_request("layer index must be an integer (-2=embedding, -1=final, >=0=block)", param);
            }
            if (val.is_number_unsigned()) {
                const auto uval = val.get<std::uint64_t>();
                if (uval > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
                    hidden_bad_request("layer index exceeds int32 range", param);
                }
                return static_cast<std::int32_t>(uval);
            }
            const auto parsed = val.get<std::int64_t>();
            if (parsed < ninfer::kHiddenStateTapEmbedding ||
                parsed > std::numeric_limits<std::int32_t>::max()) {
                hidden_bad_request("layer index must be >= -2 and <= INT32_MAX", param);
            }
            return static_cast<std::int32_t>(parsed);
        };
        if (has_layers) {
            const Json& arr = body.at("layers");
            if (!arr.is_array() || arr.empty()) {
                hidden_bad_request("layers must be a nonempty array of integers", "layers");
            }
            if (arr.size() > ninfer::kMaximumHiddenStateTapLayers) {
                hidden_bad_request("layers exceeds maximum tap count (8)", "layers");
            }
            for (const auto& entry : arr) {
                tap_layers.push_back(parse_layer_id(entry, "layers"));
            }
            explicit_layers = true;
        } else if (has_layer) {
            tap_layers.push_back(parse_layer_id(body.at("layer"), "layer"));
        } else {
            tap_layers.push_back(ninfer::kHiddenStateTapFinalNorm);
        }

        Json chat_body{{"messages", body.at("messages")},
                       {"max_tokens", 1},
                       {"temperature", 0},
                       {"stream", false},
                       {"prompt_cache_read_only", true}};
        if (body.contains("model") && body.at("model").is_string()) {
            chat_body["model"] = body.at("model");
        }
        if (body.contains("chat_template_kwargs")) {
            chat_body["chat_template_kwargs"] = body.at("chat_template_kwargs");
        } else {
            const bool enable_thinking =
                body.contains("enable_thinking") && body.at("enable_thinking").is_boolean()
                    ? body.at("enable_thinking").get<bool>()
                    : false;
            chat_body["chat_template_kwargs"] = Json{{"enable_thinking", enable_thinking}};
        }

        RequestLimits limits;
        limits.default_max_tokens = 1;
        limits.max_context        = options_.launch.engine.max_context;
        request                   = parse_chat_completion_request(chat_body, limits);
        request.model             = request.model.empty()
                                        ? served_model_names_.id
                                        : resolve_openai_model(request.model, served_model_names_);
        request.generation.capture_prompt_hidden_states = true;
        request.generation.hidden_states_layers         = tap_layers;
        request.generation.max_tokens                   = 1;
        request.generation.prompt_cache_read_only       = true;
    } catch (const ApiException& exception) {
        write_openai_error(res, exception.error());
        return;
    }

    const std::uint64_t req_id = ++request_seq_;
    RequestLogMetadata metadata{.http_request_id        = res.get_header_value("x-request-id"),
                                .model                  = request.model,
                                .stream                 = false,
                                .output_tokens_explicit = true};
    observations_.preparing(req_id, "ninfer_hidden_states", metadata);
    PreparedRequest prepared;
    try {
        ninfer::GenerationObservationOptions observation{
            .scheduling = scheduling_observer(req_id, metadata.http_request_id),
            .route_tag  = req_id,
        };
        prepared = service_->prepare(request.generation, GenerationConsumerMode::Aggregate,
                                     std::move(observation),
                                     [&req] { return client_disconnected(req); });
        if (static_cast<std::size_t>(std::max(0, prepared.prompt_tokens)) * tap_layers.size() >
            kMaxHiddenTokenLayers) {
            hidden_bad_request("prompt_tokens * num_layers exceeds hidden_states payload cap",
                               "layers");
        }
    } catch (const ApiException& exception) {
        record_request_rejected(make_request_rejection_log_context(
            req_id, "ninfer_hidden_states", request.generation, metadata, exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        record_request_rejected(make_request_rejection_log_context(
            req_id, "ninfer_hidden_states", request.generation, metadata, error));
        write_openai_error(res, error);
        return;
    }

    auto lifecycle = begin_request(make_request_log_context(
        req_id, "ninfer_hidden_states", request.generation, metadata, prepared, client_label(req)));

    GenerationOutcome outcome;
    try {
        outcome = service_->run(prepared, nullptr, [&req] { return client_disconnected(req); });
    } catch (const ApiException& exception) {
        lifecycle->failure(make_generation_request_failure(exception.error()));
        write_openai_error(res, exception.error());
        return;
    } catch (const std::exception& exception) {
        lifecycle->failure(
            make_internal_request_failure(RequestFailurePhase::Generation, exception.what()));
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = exception.what();
        write_openai_error(res, error);
        return;
    }

    const std::size_t seq_len     = static_cast<std::size_t>(outcome.prompt_tokens);
    const std::size_t num_layers  = tap_layers.size();
    const std::size_t hidden_size = static_cast<std::size_t>(outcome.prompt_hidden_size);
    const std::size_t payload_bytes =
        outcome.prompt_hidden_rows.size() * sizeof(std::uint16_t);
    if (hidden_size == 0 ||
        outcome.prompt_hidden_rows.size() != seq_len * num_layers * hidden_size) {
        lifecycle->failure(make_internal_request_failure(
            RequestFailurePhase::Generation,
            "hidden_states readout returned an unexpected tensor shape"));
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = "hidden_states readout returned an unexpected tensor shape";
        write_openai_error(res, error);
        return;
    }

    lifecycle->done(outcome);

    Json token_types = Json::array();
    for (const std::uint8_t t : outcome.prompt_token_types) {
        token_types.push_back(static_cast<int>(t));
    }
    Json token_ids       = Json::array();
    Json token_texts     = Json::array();
    Json token_byte_lens = Json::array();
    for (const ninfer::TokenId id : outcome.prompt_token_ids) {
        token_ids.push_back(static_cast<std::int64_t>(id));
        std::string bytes;
        try {
            bytes = service_->token_bytes(id);
        } catch (const std::out_of_range&) {}
        token_byte_lens.push_back(static_cast<std::uint64_t>(bytes.size()));
        // Replace any invalid UTF-8 byte fragment safely so nlohmann::json never throws on partial BPE bytes.
        token_texts.push_back(
            Json::parse(Json(bytes).dump(-1, ' ', false, Json::error_handler_t::replace))
                .get<std::string>());
    }
    Json layers_json = Json::array();
    for (const std::int32_t l : tap_layers) {
        layers_json.push_back(l);
    }
    Json legacy_layer = Json(nullptr);
    if (!explicit_layers && num_layers == 1 && tap_layers[0] >= 0) {
        legacy_layer = Json(tap_layers[0]);
    }
    Json header{
        {"model", request.model},
        {"seq_len", seq_len},
        {"num_layers", num_layers},
        {"hidden_size", hidden_size},
        {"dtype", "bf16"},
        {"layer", std::move(legacy_layer)},
        {"layers", std::move(layers_json)},
        {"token_types", std::move(token_types)},
        {"token_ids", std::move(token_ids)},
        {"token_texts", std::move(token_texts)},
        {"token_byte_lens", std::move(token_byte_lens)},
        {"prefill_ms", outcome.stats.timings.prefill_seconds * 1000.0},
        {"vision_ms", outcome.stats.timings.vision_seconds * 1000.0},
    };
    const std::string header_json = header.dump();
    const std::uint64_t header_len = static_cast<std::uint64_t>(header_json.size());

    std::string body_bytes;
    body_bytes.resize(8 + header_json.size() + payload_bytes);
    for (int i = 0; i < 8; ++i) {
        body_bytes[static_cast<std::size_t>(i)] =
            static_cast<char>((header_len >> (8 * i)) & 0xFFU);
    }
    std::memcpy(body_bytes.data() + 8, header_json.data(), header_json.size());
    std::memcpy(body_bytes.data() + 8 + header_json.size(), outcome.prompt_hidden_rows.data(),
                payload_bytes);

    res.status = 200;
    res.set_content(std::move(body_bytes), "application/octet-stream");
}

} // namespace ninfer::serve
