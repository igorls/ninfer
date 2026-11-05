#include "product/config/schema.h"

#include "product/speculative_options.h"

#include <algorithm>
#include <iterator>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ninfer::product {
namespace {

using namespace config_detail;

struct Argument {
    Field field = Field::Count;
    std::string value;
    int position     = 0;
    bool clear_stops = false;
    bool json_object = false;
};

struct CommandLine {
    bool help                = false;
    int end_options_position = 0;
    std::string error;
    std::filesystem::path config_path;
    std::vector<std::string> argv;
    std::vector<Argument> arguments;
    std::vector<CommandLineArgument> extra_arguments;
};

CommandLine read_arguments(int argc, char** argv, EntryPoint entry,
                           std::span<const CommandLineOption> extra_options) {
    CommandLine result;
    for (int i = 0; i < argc; ++i) result.argv.emplace_back(argv[i]);
    bool positional_only = false;
    bool model_seen      = false;
    bool config_seen     = false;
    std::string error;
    const auto fail = [&](std::string message) {
        if (error.empty()) error = std::move(message);
    };
    for (int i = 1; i < argc; ++i) {
        std::string_view token(argv[i]);
        const int position = i;
        if (!positional_only && token == "--") {
            positional_only             = true;
            result.end_options_position = i;
            continue;
        }
        if (positional_only || !token.starts_with('-')) {
            if (model_seen) fail("only one model artifact path may be specified");
            model_seen = true;
            result.arguments.push_back({Field::ModelPath, std::string(token), position});
            continue;
        }
        const auto equal     = token.find('=');
        const auto flag      = token.substr(0, equal);
        const bool has_value = equal != token.npos;
        const auto value     = [&]() -> std::string {
            if (has_value) return std::string(token.substr(equal + 1));
            if (i + 1 >= argc) {
                fail(std::string(flag) + " needs a value");
                return {};
            }
            return argv[++i];
        };
        if (flag == "--help" || flag == "-h") {
            if (has_value)
                fail("--help does not take a value");
            else
                result.help = true;
            continue;
        }
        if (flag == "--config") {
            if (config_seen) fail("--config may be specified only once");
            config_seen        = true;
            result.config_path = value();
            if (result.config_path.empty()) fail("--config needs a nonempty path");
            continue;
        }
        if ((flag == "--clear-stops" || flag == "--json-object") && entry == EntryPoint::Cli) {
            if (has_value) fail(std::string(flag) + " does not take a value");
            result.arguments.push_back({.position    = position,
                                        .clear_stops = flag == "--clear-stops",
                                        .json_object = flag == "--json-object"});
            continue;
        }
        const auto catalog = fields();
        const auto spec = std::find_if(catalog.begin(), catalog.end(), [&](const auto& candidate) {
            return applies_to(candidate, entry) &&
                   (candidate.flag == flag ||
                    (!candidate.negative_flag.empty() && candidate.negative_flag == flag));
        });
        if (spec == catalog.end()) {
            const auto extra =
                std::find_if(extra_options.begin(), extra_options.end(),
                             [&](const auto& option) { return option.flag == flag; });
            if (extra == extra_options.end())
                fail("unknown argument: " + std::string(flag));
            else if (extra->takes_value)
                result.extra_arguments.push_back({std::string(flag), value()});
            else if (has_value)
                fail(std::string(flag) + " does not take a value");
            else
                result.extra_arguments.push_back({std::string(flag), {}});
            continue;
        }
        std::string argument;
        if (!spec->negative_flag.empty()) {
            if (flag == spec->negative_flag && has_value)
                fail(std::string(flag) + " does not take a value");
            argument = has_value ? value() : flag == spec->negative_flag ? "false" : "true";
        } else
            argument = value();
        if (spec->field == Field::ApiKey) {
            if (has_value)
                result.argv[position] = "--api-key=<redacted>";
            else if (i != position)
                result.argv[i] = "<redacted>";
        }
        if (spec->field == Field::ConstraintKind && argument != "none")
            fail("--constraint accepts none; select a constraint with its dedicated option");
        result.arguments.push_back({spec->field, std::move(argument), position});
    }
    result.error = std::move(error);
    return result;
}

bool list_field(Field field) {
    return field == Field::StopIds || field == Field::Stop || field == Field::ReasoningStop ||
           field == Field::Choices || field == Field::ModelAliases;
}

void select_constraint(StartupPatch& patch, std::string kind, ConfigSource source) {
    auto& selected = assignment(patch, Field::ConstraintKind);
    if (selected && std::get<std::string>(selected->value) != kind)
        field_error(field_spec(Field::ConstraintKind), source, {},
                    "select only one output constraint");
    selected = Assignment{std::move(kind), source};
}

StartupPatch command_patch(const CommandLine& command) {
    StartupPatch patch;
    for (const auto& argument : command.arguments) {
        const ConfigSource source{.kind     = ConfigSourceKind::CommandLine,
                                  .argument = argument.position};
        if (argument.clear_stops) {
            assignment(patch, Field::StopIds) = Assignment{std::vector<std::uint32_t>{}, source};
            assignment(patch, Field::Stop)    = Assignment{std::vector<std::string>{}, source};
            assignment(patch, Field::ReasoningStop) =
                Assignment{std::vector<std::string>{}, source};
            continue;
        }
        if (argument.json_object) {
            select_constraint(patch, "json_object", source);
            continue;
        }
        const auto& spec = field_spec(argument.field);
        FieldValue value;
        try {
            value = parse_field_value(argument.field, RawValue{argument.value}, true);
        } catch (const std::invalid_argument& error) {
            field_error(spec, source, {}, error.what());
        }
        switch (argument.field) {
        case Field::GrammarFile:
            select_constraint(patch, "grammar", source);
            break;
        case Field::JsonSchemaFile:
            select_constraint(patch, "json_schema", source);
            break;
        case Field::Regex:
            select_constraint(patch, "regex", source);
            break;
        case Field::Choices:
            select_constraint(patch, "choice", source);
            break;
        case Field::ConstraintKind:
            select_constraint(patch, "none", source);
            continue;
        default:
            break;
        }
        auto& destination = assignment(patch, argument.field);
        if (list_field(argument.field) && destination) {
            if (argument.field == Field::StopIds) {
                auto& out         = std::get<std::vector<std::uint32_t>>(destination->value);
                const auto& added = std::get<std::vector<std::uint32_t>>(value);
                out.insert(out.end(), added.begin(), added.end());
            } else {
                auto& out   = std::get<std::vector<std::string>>(destination->value);
                auto& added = std::get<std::vector<std::string>>(value);
                out.insert(out.end(), std::make_move_iterator(added.begin()),
                           std::make_move_iterator(added.end()));
            }
        } else
            destination = Assignment{std::move(value), source};
    }
    return patch;
}

constexpr Field kConstraintFields[] = {Field::ConstraintKind, Field::GrammarFile,
                                       Field::JsonSchemaFile, Field::Regex, Field::Choices};

void check_layer(const StartupPatch& patch, EntryPoint entry, const std::filesystem::path& path) {
    if (entry == EntryPoint::Serve && assignment(patch, Field::ApiKey) &&
        assignment(patch, Field::ApiKeyFile))
        field_error(field_spec(Field::ApiKeyFile), assignment(patch, Field::ApiKeyFile)->source,
                    path, "select api_key or api_key_file within one source");

    const auto fail = [&](Field field, std::string_view message) {
        field_error(field_spec(field), assignment(patch, field)->source, path, message);
    };
    if (const auto& backend = assignment(patch, Field::SpecBackend);
        backend && std::get<std::string>(backend->value) == "none") {
        if (const auto& drafts = assignment(patch, Field::DraftTokens);
            drafts && std::get<std::uint64_t>(drafts->value) != 0)
            fail(Field::DraftTokens, "disabled speculative backend requires zero drafts");
        if (const auto& head = assignment(patch, Field::ProposalHead);
            head && std::get<std::string>(head->value) != "full")
            fail(Field::ProposalHead, "disabled speculative backend requires the full head");
    }
    if (entry != EntryPoint::Cli) return;
    if (assignment(patch, Field::Prompt) && assignment(patch, Field::Messages))
        fail(Field::Messages, "select only one of prompt or messages in each source");
    const auto& effort   = assignment(patch, Field::ReasoningEffort);
    const auto& thinking = assignment(patch, Field::Thinking);
    if (effort && thinking && !std::holds_alternative<DefaultValue>(effort->value) &&
        !std::holds_alternative<DefaultValue>(thinking->value) &&
        std::get<bool>(thinking->value) != (std::get<std::string>(effort->value) != "none"))
        fail(Field::ReasoningEffort, "reasoning effort conflicts with thinking in the same source");
    const auto& selection  = assignment(patch, Field::ConstraintKind);
    const std::string kind = selection ? std::get<std::string>(selection->value) : "none";
    const Field expected   = kind == "grammar"       ? Field::GrammarFile
                             : kind == "json_schema" ? Field::JsonSchemaFile
                             : kind == "regex"       ? Field::Regex
                             : kind == "choice"      ? Field::Choices
                                                     : Field::Count;
    for (Field field : kConstraintFields) {
        if (field != Field::ConstraintKind && assignment(patch, field) && field != expected)
            fail(field, "field does not belong to the selected constraint kind");
    }
    if (expected != Field::Count && !assignment(patch, expected))
        fail(Field::ConstraintKind,
             "selected constraint requires " + std::string(field_spec(expected).key));
}

void merge_layer(StartupPatch& merged, StartupPatch layer, EntryPoint entry) {
    if (assignment(layer, Field::ApiKey)) assignment(merged, Field::ApiKeyFile).reset();
    if (assignment(layer, Field::ApiKeyFile)) assignment(merged, Field::ApiKey).reset();
    if (const auto& backend = assignment(layer, Field::SpecBackend);
        backend && std::get<std::string>(backend->value) == "none") {
        assignment(merged, Field::DraftTokens)  = Assignment{std::uint64_t{0}, backend->source};
        assignment(merged, Field::ProposalHead) = Assignment{std::string("full"), backend->source};
    }
    if (entry == EntryPoint::Cli) {
        if (assignment(layer, Field::Prompt) || assignment(layer, Field::Messages)) {
            assignment(merged, Field::Prompt).reset();
            assignment(merged, Field::Messages).reset();
        }
        if (assignment(layer, Field::ConstraintKind)) {
            for (Field field : kConstraintFields) assignment(merged, field).reset();
        }
        if (const auto& effort = assignment(layer, Field::ReasoningEffort)) {
            if (!std::holds_alternative<DefaultValue>(effort->value))
                assignment(layer, Field::Thinking) =
                    Assignment{std::get<std::string>(effort->value) != "none", effort->source};
            else if (!assignment(layer, Field::Thinking))
                assignment(layer, Field::Thinking) = Assignment{DefaultValue{}, effort->source};
        } else if (const auto& thinking = assignment(layer, Field::Thinking))
            assignment(layer, Field::ReasoningEffort) =
                Assignment{DefaultValue{}, thinking->source};
    }
    for (std::size_t i = 0; i < layer.size(); ++i)
        if (layer[i]) merged[i] = std::move(layer[i]);
}

ReasoningEffort reasoning_effort(std::string_view name) {
    if (name == "none") return ReasoningEffort::None;
    if (name == "minimal") return ReasoningEffort::Minimal;
    if (name == "low") return ReasoningEffort::Low;
    if (name == "medium") return ReasoningEffort::Medium;
    if (name == "high") return ReasoningEffort::High;
    if (name == "xhigh") return ReasoningEffort::XHigh;
    return ReasoningEffort::Max;
}

KvCacheStorage kv_storage(std::string_view name) {
    if (name == "bf16") return KvCacheStorage::BFloat16;
    if (name == "int8") return KvCacheStorage::Int8Group64;
    if (name == "fp8") return KvCacheStorage::Fp8E4M3Row256;
    if (name == "nvfp4") return KvCacheStorage::Nvfp4Group16;
    return KvCacheStorage::Fp8KeyNvfp4Value;
}

ConstraintKind constraint_kind(std::string_view name) {
    if (name == "none") return ConstraintKind::None;
    if (name == "grammar") return ConstraintKind::Grammar;
    if (name == "json_object") return ConstraintKind::JsonObject;
    if (name == "json_schema") return ConstraintKind::JsonSchema;
    if (name == "regex") return ConstraintKind::Regex;
    return ConstraintKind::Choice;
}

struct ParsedOptions {
    ParsedLaunchOptions common;
    CliOptions cli;
    ServeOptions serve;
};

ParsedOptions parse(int argc, char** argv, EntryPoint entry,
                    std::span<const CommandLineOption> extra_options = {}) {
    ParsedOptions parsed;
    parsed.common.launch = default_launch_options(entry);
    auto command         = read_arguments(argc, argv, entry, extra_options);
    if (command.help) {
        parsed.common.help_requested = true;
        parsed.cli.help_requested    = true;
        parsed.serve.help_requested  = true;
        return parsed;
    }
    if (!command.error.empty()) throw std::invalid_argument(command.error);
    StartupPatch merged;
    FileConfiguration file;
    if (!command.config_path.empty()) {
        command.config_path = std::filesystem::absolute(command.config_path).lexically_normal();
        file                = read_config_file(command.config_path);
        check_layer(file.fields, entry, command.config_path);
        merge_layer(merged, std::move(file.fields), entry);
    }
    auto cli = command_patch(command);
    check_layer(cli, entry, command.config_path);
    merge_layer(merged, std::move(cli), entry);

    parsed.common.extra_arguments = std::move(command.extra_arguments);
    auto& launch                  = parsed.common.launch;
    auto& engine                  = launch.engine;
    auto& generation              = launch.generation;
    auto& sampling                = generation.sampling;
    if (entry == EntryPoint::Bench) {
        const auto& limit = assignment(merged, Field::MaxTokens);
        if (limit && std::holds_alternative<std::uint64_t>(limit->value) &&
            std::get<std::uint64_t>(limit->value) == 1) {
            engine.context_cache.enabled           = false;
            engine.context_cache.extra_state_slots = 0;
        }
    }
    launch.provenance.config_path = command.config_path;
    launch.provenance.argv        = std::move(command.argv);
    engine.deployment             = std::move(file.deployment);
    launch.provenance.fields      = std::move(file.deployment_sources);
    bool kv_follows_context       = true;
    for (const auto& spec : fields()) {
        const auto& setting = assignment(merged, spec.field);
        if (!setting || !applies_to(spec, entry)) continue;
        launch.provenance.fields.emplace(spec.key, setting->source);
        const auto& value = setting->value;
        if (std::holds_alternative<DefaultValue>(value)) {
            if (spec.field == Field::PreserveThinking) generation.preserve_thinking.reset();
            continue;
        }
        const auto u64     = [&] { return std::get<std::uint64_t>(value); };
        const auto u32     = [&] { return static_cast<std::uint32_t>(u64()); };
        const auto real    = [&] { return static_cast<float>(std::get<double>(value)); };
        const auto boolean = [&] { return std::get<bool>(value); };
        const auto string  = [&]() -> const std::string& { return std::get<std::string>(value); };
        const auto path    = [&] {
            if (string().empty()) return std::filesystem::path{};
            auto result = std::filesystem::path(string());
            if (setting->source.kind == ConfigSourceKind::File && result.is_relative())
                result = command.config_path.parent_path() / result;
            return std::filesystem::absolute(result).lexically_normal();
        };
        switch (spec.field) {
        case Field::ModelPath:
            engine.artifact_path = path();
            break;
        case Field::ChatTemplate:
            engine.chat_template_path = path();
            break;
        case Field::Vision:
            engine.enable_vision = boolean();
            break;
        case Field::Device:
            engine.device = static_cast<int>(u64());
            break;
        case Field::MaxContext:
            engine.max_context = u32();
            break;
        case Field::KvDtype:
            engine.kv_cache = kv_storage(string());
            break;
        case Field::KvCapacity:
            if (const auto* capacity = std::get_if<std::uint64_t>(&value)) {
                engine.kv_capacity =
                    KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(*capacity));
                kv_follows_context = false;
            } else if (string() == "auto") {
                engine.kv_capacity = KvCapacityPolicy::automatic();
                kv_follows_context = false;
            }
            break;
        case Field::PrefillChunk:
            engine.prefill_chunk = u32();
            break;
        case Field::GpuTiming:
            engine.observability.gpu_timing = boolean();
            break;
        case Field::CudaGraph:
            engine.use_cuda_graph = boolean();
            break;
        case Field::MaxConcurrency:
            engine.max_concurrency = u32();
            break;
        case Field::MaxPendingRequests:
            engine.max_pending_requests = u32();
            break;
        case Field::PendingTimeout:
            engine.pending_timeout_ms = u32();
            break;
        case Field::PrefixReuse:
            engine.context_cache.enabled = boolean();
            break;
        case Field::ExtraStateSlots:
            engine.context_cache.extra_state_slots = u32();
            break;
        case Field::HostCapacity:
            engine.context_cache.host_capacity_bytes = u64();
            break;
        case Field::CostPresets:
            engine.context_cost.preset_path = path();
            break;
        case Field::SpecBackend:
            engine.speculative.backend = parse_speculative_backend(string());
            break;
        case Field::DraftTokens:
            engine.speculative.draft_tokens = u32();
            break;
        case Field::ProposalHead:
            engine.speculative.proposal_head =
                string() == "optimized" ? ProposalHead::Optimized : ProposalHead::Full;
            break;
        case Field::MediaCache:
            engine.media_cache_bytes = u64();
            break;
        case Field::MediaLive:
            engine.media_live_bytes = u64();
            break;
        case Field::MediaThreads:
            engine.media_preprocess_threads = u32();
            break;
        case Field::MaxTokens:
            generation.max_tokens = u32();
            break;
        case Field::Thinking:
            generation.enable_thinking = boolean();
            break;
        case Field::PreserveThinking:
            generation.preserve_thinking = boolean();
            break;
        case Field::ThinkingBudget:
            generation.thinking_budget = u32();
            break;
        case Field::Temperature:
            sampling.temperature = real();
            break;
        case Field::TopK:
            sampling.top_k = static_cast<std::int32_t>(u64());
            break;
        case Field::TopP:
            sampling.top_p = real();
            break;
        case Field::MinP:
            sampling.min_p = real();
            break;
        case Field::PresencePenalty:
            sampling.presence_penalty = real();
            break;
        case Field::FrequencyPenalty:
            sampling.frequency_penalty = real();
            break;
        case Field::Seed:
            if (std::holds_alternative<std::string>(value))
                sampling.seed.reset();
            else
                sampling.seed = u64();
            break;
        case Field::ServerHost:
            parsed.serve.host = string();
            break;
        case Field::ServerPort:
            parsed.serve.port = static_cast<int>(u64());
            break;
        case Field::ApiKey:
            parsed.serve.api_key = string();
            break;
        case Field::ModelId:
            if (!string().empty()) parsed.serve.model_id_override = string();
            break;
        case Field::Cors:
            parsed.serve.enable_cors = boolean();
            break;
        case Field::MaxRequestSize:
            parsed.serve.max_request_bytes = u64();
            break;
        case Field::ResponseRecords:
            parsed.serve.response_store_max_records = u64();
            break;
        case Field::ResponseSize:
            parsed.serve.response_store_max_bytes = u64();
            break;
        case Field::LogLevel:
            launch.logging.level = parse_log_level(string());
            break;
        case Field::StatsInterval:
            launch.logging.stats_interval_ms = u32();
            break;
        case Field::RequestJsonl:
            launch.logging.request_jsonl = path();
            break;
        case Field::Prompt:
            parsed.cli.prompt = string();
            break;
        case Field::Messages:
            parsed.cli.messages_path = path();
            break;
        case Field::ReasoningEffort:
            parsed.cli.reasoning_effort = reasoning_effort(string());
            break;
        case Field::RawOutput:
            parsed.cli.raw_output = boolean();
            break;
        case Field::PrintTokenIds:
            parsed.cli.print_token_ids = boolean();
            break;
        case Field::StopIds:
            for (auto token : std::get<std::vector<std::uint32_t>>(value))
                parsed.cli.stop_token_ids.push_back(static_cast<TokenId>(token));
            break;
        case Field::Stop:
        case Field::ReasoningStop:
            for (const auto& stop : std::get<std::vector<std::string>>(value))
                parsed.cli.stop_strings.push_back({.text    = stop,
                                                   .channel = spec.field == Field::Stop
                                                                  ? OutputChannel::Content
                                                                  : OutputChannel::Reasoning});
            break;
        case Field::ConstraintKind:
            parsed.cli.constraint.kind = constraint_kind(string());
            break;
        case Field::GrammarFile:
        case Field::JsonSchemaFile:
            parsed.cli.constraint.path = path();
            break;
        case Field::Regex:
            parsed.cli.constraint.regex = string();
            break;
        case Field::Choices:
            parsed.cli.constraint.choices = std::get<std::vector<std::string>>(value);
            break;
        case Field::DesktopReserve:
            engine.desktop_reserve_bytes = u64();
            break;
        case Field::SlackFloor:
            engine.min_slack_floor_bytes = u64();
            break;
        case Field::ClampConcurrency:
            engine.clamp_concurrency_to_pool = boolean();
            break;
        case Field::RepetitionPenalty:
            sampling.repetition_penalty = real();
            break;
        case Field::ApiKeyFile:
            parsed.serve.api_key_file = path();
            break;
        case Field::ModelAliases:
            parsed.serve.model_aliases = std::get<std::vector<std::string>>(value);
            break;
        case Field::ExpertStats:
            parsed.serve.expert_stats = boolean();
            break;
        case Field::ExpertTrace:
            parsed.serve.expert_trace_directory = path();
            break;
        case Field::ExpertTraceMax:
            parsed.serve.expert_trace_max_bytes = u64();
            break;
        case Field::SystemOneReadout:
            parsed.serve.systemone_policy.readout = string() == "single"
                                                        ? SystemOnePolicy::Readout::Single
                                                        : SystemOnePolicy::Readout::Averaged;
            break;
        case Field::SystemOneReasoningBudget:
            parsed.serve.systemone_policy.reasoning_budget = u32();
            break;
        case Field::SystemOneEntropy:
            parsed.serve.systemone_policy.escalate_entropy = std::get<double>(value);
            break;
        case Field::RerankExact:
            parsed.serve.rerank_weight_exact = std::get<double>(value);
            break;
        case Field::RerankSubstitute:
            parsed.serve.rerank_weight_substitute = std::get<double>(value);
            break;
        case Field::RerankComplement:
            parsed.serve.rerank_weight_complement = std::get<double>(value);
            break;
        case Field::RerankIrrelevant:
            parsed.serve.rerank_weight_irrelevant = std::get<double>(value);
            break;
        case Field::RerankMaxDocuments:
            parsed.serve.rerank_max_documents = u64();
            break;
        case Field::RerankModelId:
            parsed.serve.rerank_model_id = string();
            break;
        case Field::Count:
            break;
        }
    }
    const auto fail = [&](Field field, std::string_view message) {
        const auto& setting = assignment(merged, field);
        field_error(field_spec(field), setting ? setting->source : ConfigSource{},
                    command.config_path, message);
    };
    if (engine.artifact_path.empty())
        throw std::invalid_argument("model.path or a model artifact argument is required");
    if (kv_follows_context)
        engine.kv_capacity = KvCapacityPolicy::explicit_capacity(engine.max_context);
    if (engine.kv_capacity.mode == KvCapacityMode::Explicit &&
        engine.kv_capacity.explicit_tokens < engine.max_context)
        fail(Field::KvCapacity, "must be at least engine.max_context");
    try {
        validate_speculative_cli_options(engine.speculative);
    } catch (const std::invalid_argument& error) {
        fail(assignment(merged, Field::DraftTokens)    ? Field::DraftTokens
             : assignment(merged, Field::ProposalHead) ? Field::ProposalHead
                                                       : Field::SpecBackend,
             error.what());
    }
    if (entry == EntryPoint::Serve) {
        auto& serve = parsed.serve;
        if (!serve.api_key_file.empty()) {
            std::ifstream input(serve.api_key_file, std::ios::binary);
            if (!input) fail(Field::ApiKeyFile, "cannot read the API key file");
            std::ostringstream buffer;
            buffer << input.rdbuf();
            if (input.bad()) fail(Field::ApiKeyFile, "cannot read the API key file");
            serve.api_key = buffer.str();
            while (!serve.api_key.empty() &&
                   std::string_view("\r\n \t").find(serve.api_key.back()) != std::string_view::npos)
                serve.api_key.pop_back();
            if (serve.api_key.empty()) fail(Field::ApiKeyFile, "API key file is empty");
        }
        if (!serve.expert_trace_directory.empty()) serve.expert_stats = true;
        std::vector<std::string> aliases;
        for (const auto& alias : serve.model_aliases) {
            if (alias == "default")
                fail(Field::ModelAliases, "default is accepted by every server");
            if (std::find(aliases.begin(), aliases.end(), alias) == aliases.end())
                aliases.push_back(alias);
        }
        serve.model_aliases = std::move(aliases);
        // The fork drops closed-turn reasoning unless a request/startup override selects it.
        if (!generation.preserve_thinking && !assignment(merged, Field::PreserveThinking))
            generation.preserve_thinking = false;
        engine.min_slack_floor_bytes =
            engine.kv_capacity.mode == KvCapacityMode::Automatic
                ? std::max(kDefaultKvCapacitySlackFloorBytes, engine.min_slack_floor_bytes)
                : 0;
    }
    if (entry == EntryPoint::Cli) {
        if (parsed.cli.prompt.empty() && parsed.cli.messages_path.empty())
            throw std::invalid_argument(
                "select cli.prompt or cli.messages (or --prompt/--messages)");
    }
    if (entry != EntryPoint::Serve && generation.enable_thinking == false &&
        generation.thinking_budget)
        fail(Field::ThinkingBudget, "requires thinking to be enabled");
    return parsed;
}

} // namespace

LaunchOptions default_launch_options(EntryPoint entry) {
    LaunchOptions options;
    if (entry != EntryPoint::Serve) {
        if (entry == EntryPoint::Cli) {
            options.engine.context_cache.enabled           = false;
            options.engine.context_cache.extra_state_slots = 0;
        }
        options.engine.context_cache.host_capacity_bytes = 0;
        options.generation.sampling.seed                 = 0;
    } else {
        options.generation.max_tokens        = kDefaultServeMaxTokens;
        options.generation.preserve_thinking = false;
    }
    return options;
}

std::uint64_t random_sampling_seed() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    return rng();
}

CliOptions parse_cli_options(int argc, char** argv) {
    auto parsed       = parse(argc, argv, EntryPoint::Cli);
    parsed.cli.launch = std::move(parsed.common.launch);
    return std::move(parsed.cli);
}

ServeOptions parse_serve_options(int argc, char** argv) {
    auto parsed         = parse(argc, argv, EntryPoint::Serve);
    parsed.serve.launch = std::move(parsed.common.launch);
    return std::move(parsed.serve);
}

ParsedLaunchOptions parse_launch_options(int argc, char** argv, EntryPoint entry,
                                         std::span<const CommandLineOption> extra_options) {
    return parse(argc, argv, entry, extra_options).common;
}

namespace {
const FieldSpec& serve_argument_spec(std::string_view key) {
    for (const auto& spec : fields())
        if (spec.key == key && applies_to(spec, EntryPoint::Serve) && !spec.flag.empty())
            return spec;
    throw std::invalid_argument("unknown serve argument key: " + std::string(key));
}

CommandLine child_arguments(std::span<const std::string> args) {
    std::vector<std::string> arguments{"ninfer-serve"};
    arguments.insert(arguments.end(), args.begin(), args.end());
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    return read_arguments(static_cast<int>(argv.size()), argv.data(), EntryPoint::Serve, {});
}
} // namespace

bool has_serve_argument(std::span<const std::string> args, std::string_view key) {
    const auto field   = serve_argument_spec(key).field;
    const auto command = child_arguments(args);
    return std::any_of(command.arguments.begin(), command.arguments.end(),
                       [&](const auto& argument) { return argument.field == field; });
}

std::vector<std::string> remove_serve_argument(std::vector<std::string> args,
                                               std::string_view key) {
    const auto& spec   = serve_argument_spec(key);
    const auto field   = spec.field;
    const auto command = child_arguments(args);
    for (auto i = command.arguments.rbegin(); i != command.arguments.rend(); ++i) {
        if (i->field != field) continue;
        const auto position = static_cast<std::size_t>(i->position - 1);
        const auto count    = !spec.negative_flag.empty() ||
                                   args[position].find('=') != std::string::npos ||
                                   position + 1 == args.size()
                                  ? std::size_t{1}
                                  : std::size_t{2};
        args.erase(args.begin() + static_cast<std::ptrdiff_t>(position),
                   args.begin() + static_cast<std::ptrdiff_t>(position + count));
    }
    return args;
}

std::vector<std::string> set_serve_argument(std::vector<std::string> args, std::string_view key,
                                            std::string value) {
    const auto& spec = serve_argument_spec(key);
    if (!spec.negative_flag.empty())
        throw std::invalid_argument("child edit requires a scalar argument");
    const auto command = child_arguments(args);
    for (auto i = command.arguments.rbegin(); i != command.arguments.rend(); ++i) {
        if (i->field != spec.field) continue;
        const auto position = static_cast<std::size_t>(i->position - 1);
        if (args[position].find('=') != std::string::npos)
            args[position] = std::string(spec.flag) + "=" + value;
        else if (position + 1 < args.size())
            args[position + 1] = std::move(value);
        else
            args.push_back(std::move(value));
        return args;
    }
    const auto position = command.end_options_position
                              ? static_cast<std::size_t>(command.end_options_position - 1)
                              : args.size();
    args.insert(args.begin() + static_cast<std::ptrdiff_t>(position),
                {std::string(spec.flag), std::move(value)});
    return args;
}

std::string startup_usage_text(std::string_view executable, EntryPoint entry) {
    std::ostringstream out;
    out << "usage: " << executable << " [--config FILE] [model.ninfer] [options]\n\n"
        << "Command-line values override TOML settings. See ninfer.example.toml.\n"
        << "Configuration is logged automatically before startup.\n";
    std::string_view previous;
    for (const auto& spec : fields()) {
        if (!applies_to(spec, entry) || spec.flag.empty()) continue;
        const auto group = spec.key.substr(0, spec.key.find('.'));
        if (group != previous) {
            out << '\n' << group << ":\n";
            previous = group;
        }
        out << "  " << spec.flag;
        if (!spec.negative_flag.empty()) out << " / " << spec.negative_flag;
        if (!spec.argument.empty()) out << ' ' << spec.argument;
        out << "\n      " << spec.description << '\n';
    }
    if (entry == EntryPoint::Cli)
        out << "  --json-object\n      Constrain output to a JSON object.\n";
    out << "\nprocess:\n  --config FILE\n  --help, -h\n";
    if (entry == EntryPoint::Cli)
        out << "  --clear-stops\n\nAnswer content uses stdout; reasoning and "
               "diagnostics use stderr.\n";
    out << "\nCapacities: bytes or B/KiB/MiB/GiB; durations: milliseconds or ms/s.\n"
        << "Boolean options also accept --flag=true|false.\n"
        << "Default prefill chunk: Qwen4Exp/CPU expert offload 4096, otherwise 1024.\n"
        << "Context cache defaults: CLI 0 extra slots / 0B Host; serve one extra slot per lane,\n"
        << "RTX 5090 Host 8GiB + 8 StateImages, Spark Host 0B.\n"
        << "Environment: NINFER_VRAM_BUDGET_MIB, NINFER_CUDA_SYNC.\n";
    return out.str();
}

} // namespace ninfer::product
