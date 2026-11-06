#include "evaluation.h"
#include "options.h"
#include "report.h"
#include "product/logging/configuration.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

// Node paths must be absolute for the platform that parses the launch file.
#ifdef _WIN32
#    define NODE_ROOT "C:"
#else
#    define NODE_ROOT ""
#endif

namespace {

int require(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

int check_partition(std::size_t tokens, std::uint32_t context, std::uint32_t stride) {
    const auto windows   = ninfer::perplexity::plan_windows(tokens, context, stride);
    int failures         = 0;
    std::size_t frontier = 1;
    for (const auto& window : windows) {
        failures += require(window.input_begin < window.input_end, "window input is nonempty");
        failures +=
            require(window.input_end - window.input_begin <= context, "window respects context");
        failures += require(window.target_begin == frontier, "target ranges are contiguous");
        failures +=
            require(window.target_end <= window.input_end, "targets are inside input window");
        failures += require(window.first_target == window.target_begin - window.input_begin,
                            "local target matches global range");
        frontier = window.target_end;
    }
    failures += require(frontier == tokens, "targets cover every token after the first");
    return failures;
}

struct Files {
    std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        ("ninfer-ppl-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

    Files() { std::filesystem::create_directories(root); }

    ~Files() { std::filesystem::remove_all(root); }
};

ninfer::perplexity::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (auto& argument : arguments) argv.push_back(argument.data());
    return ninfer::perplexity::parse_options(static_cast<int>(argv.size()), argv.data());
}

int check_options(const Files& files) {
    using namespace ninfer;
    using product::EntryPoint;
    int failures        = 0;
    const auto defaults = parse({"ninfer-perplexity", "model.ninfer", "--text", "text.txt"});
    failures +=
        require(defaults.context == 4096 && defaults.stride == 2048 &&
                    defaults.launch.engine.max_context == 4096 &&
                    defaults.launch.engine.purpose == EnginePurpose::CausalScoring &&
                    defaults.launch.engine.kv_cache == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0,
                "PPL defaults changed");
    const auto window = parse({"ninfer-perplexity", "model.ninfer", "--text", "text.txt",
                               "--context", "6144", "--stride", "1024"});
    failures += require(window.launch.engine.max_context == 6144,
                        "implicit Engine capacity did not follow the scoring window");
    const auto config = files.root / "launch.toml";
    {
        std::ofstream file(config);
        file << R"(
[model]
path = "weights/model.ninfer"
vision = true
[engine]
max_context = 8192
prefill_chunk = 256
context_capacity = "2GiB"
kv_dtype = "bf16"
max_concurrency = 8
[speculative]
backend = "none"
draft_tokens = 3
[generation]
thinking = false
thinking_budget = 4
[server]
port = 9999
[distributed]
control_host = "127.0.0.1"
[distributed.parallel]
tp = 2
[[distributed.nodes]]
id = 0
worker = ")" NODE_ROOT R"(/ninfer-worker"
artifact = ")" NODE_ROOT R"(/weights/model.ninfer"
[[distributed.nodes]]
id = 1
ssh = "user@spark"
worker = ")" NODE_ROOT R"(/ninfer-worker"
artifact = ")" NODE_ROOT R"(/weights/model.ninfer"
[[distributed.ranks]]
rank = 0
node = 0
device = 0
[[distributed.ranks]]
rank = 1
node = 1
device = 0
)";
        if (!file) throw std::runtime_error("cannot write PPL configuration fixture");
    }
    const auto configured =
        parse({"ninfer-perplexity", "--config", config.string(), "--text", "text.txt"});
    const auto& e = configured.launch.engine;
    failures +=
        require(e.max_context == 8192 && configured.context == 4096 && e.prefill_chunk == 256 &&
                    e.context_capacity.explicit_bytes == (2ULL << 30) &&
                    e.kv_cache == KvCacheStorage::BFloat16 &&
                    e.artifact_path == files.root / "weights/model.ninfer",
                "public config changed the scoring protocol or lost startup settings");
    failures +=
        require(e.deployment && e.deployment->ranks.size() == 2 && e.deployment->parallel.tp == 2 &&
                    e.deployment->nodes[1].ssh == "user@spark",
                "PPL dropped deployment");
    failures += require(!e.enable_vision && e.max_concurrency == 1 &&
                            e.speculative.backend == SpeculativeBackend::None &&
                            e.speculative.draft_tokens == 0 &&
                            !configured.launch.generation.thinking_budget,
                        "generation-only configuration reached PPL");
    const auto overridden =
        parse({"ninfer-perplexity", "--config", config.string(), "--text", "text.txt",
               "--max-context", "16384", "--kv-dtype", "fp8", "--prefill-chunk", "128"});
    failures += require(overridden.launch.engine.max_context == 16384 &&
                            overridden.launch.engine.prefill_chunk == 128 &&
                            overridden.launch.engine.kv_cache == defaults.launch.engine.kv_cache,
                        "CLI did not override PPL startup config");
    const auto display = product::format_configuration(
        product::configuration_view(configured.launch, EntryPoint::Perplexity));
    failures += require(display.find("configuration  perplexity") != std::string::npos &&
                            display.find("distributed") != std::string::npos &&
                            display.find("[file:") != std::string::npos &&
                            display.find("speculative") == std::string::npos &&
                            display.find("thinking_budget") == std::string::npos,
                        "PPL configuration display lost scope or source");
    const auto help = ninfer::perplexity::usage_text();
    failures += require(help.find("--context-capacity") != std::string::npos &&
                            help.find("--cuda-graph") == std::string::npos,
                        "PPL help lists inapplicable startup options");
    failures +=
        require(parse({"ninfer-perplexity", "--config", "absent.toml", "--help"}).help_requested,
                "PPL help tried to load configuration");
    for (const auto& extra : std::vector<std::vector<std::string>>{{"--corpus", "manifest.json"},
                                                                   {"--quick"},
                                                                   {"--stride", "4096"},
                                                                   {"--context", "4294967296"},
                                                                   {"--max-context", "2048"},
                                                                   {"--vision"}}) {
        std::vector<std::string> args{"ninfer-perplexity", "model.ninfer", "--text", "text.txt"};
        args.insert(args.end(), extra.begin(), extra.end());
        bool rejected = false;
        try {
            (void)parse(std::move(args));
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += require(rejected, "invalid PPL options were accepted");
    }
    return failures;
}

int check_report(const Files& files) {
    using namespace ninfer;
    using namespace ninfer::perplexity;
    auto options   = parse({"ninfer-perplexity", "model.ninfer", "--text", "text.txt"});
    options.output = files.root / "result";
    RunDescription run;
    run.configuration =
        product::configuration_view(options.launch, product::EntryPoint::Perplexity);
    run.engine.ranks         = {{.rank = 0, .node = 0, .device = 2, .hardware = "first"},
                                {.rank = 1, .node = 1, .device = 3, .hardware = "second"}};
    run.engine.max_context   = 4096;
    run.engine.prefill_chunk = 256;
    run.engine.kv_cache      = options.launch.engine.kv_cache;
    run.startup.ranks.resize(2);
    run.initial.ranks.resize(2);
    for (std::uint32_t rank = 0; rank < 2; ++rank) {
        run.initial.ranks[rank].rank                                    = rank;
        run.initial.ranks[rank].memory.allocations.device_weights_bytes = (rank + 1) * 1024;
    }
    EvaluationResult result;
    result.scoring_seconds = 2;
    for (const auto& [domain, logprobs] : std::vector<std::pair<std::string, std::vector<float>>>{
             {"a", {-1.0F, -2.0F}}, {"b", {-3.0F}}}) {
        StreamResult stream{.id = domain, .domain = domain, .input_tokens = logprobs.size() + 1};
        stream.score.add(logprobs);
        stream.windows.push_back(
            {plan_windows(stream.input_tokens, 4096, 2048).front(), stream.score, 1});
        result.overall.add(stream.score);
        result.domains[domain].add(stream.score);
        result.streams.push_back(std::move(stream));
    }
    const CorpusSelection corpus{.corpus_id = "fixture", .mode = "custom", .source = "text.txt"};
    const auto report = make_report(options, run, corpus, result);
    int failures      = 0;
    failures += require(report.at("startup").at("ranks").size() == 2 &&
                            report["startup"]["ranks"][1]["node"] == 1 &&
                            report["startup"]["ranks"][1]["device"] == 3 &&
                            report["startup"]["total_allocations"]["device_weights_bytes"] == 3072,
                        "PPL report lost actual rank resources");
    failures +=
        require(report["execution"]["prefill_chunk_tokens"] == 256 &&
                    report["overall"]["scored_tokens"] == 3 &&
                    report["overall"]["total_nll"] == 6.0 && report["overall"]["mean_nll"] == 2.0 &&
                    report["streams"][0]["windows"][0]["target_end"] == 3 &&
                    report["timing"]["scored_tokens_per_second"] == 1.5,
                "PPL report changed score aggregation or effective execution settings");
    const auto directory = prepare_output_directory(options, run, corpus);
    std::ostringstream summary;
    write_report(directory, report, summary);
    std::ifstream file(directory / "report.json");
    failures += require(nlohmann::json::parse(file) == report &&
                            !std::filesystem::exists(directory / "report.json.tmp") &&
                            summary.str().find("ranks: 2") != std::string::npos,
                        "PPL report publication differs from the result");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += check_partition(2, 4096, 2048);
    failures += check_partition(4096, 4096, 2048);
    failures += check_partition(4097, 4096, 2048);
    failures += check_partition(12001, 4096, 2048);

    const std::array<ninfer::perplexity::WindowPlan, 3> expected{{
        {.input_begin = 0, .input_end = 6, .target_begin = 1, .target_end = 6, .first_target = 1},
        {.input_begin = 2, .input_end = 8, .target_begin = 6, .target_end = 8, .first_target = 4},
        {.input_begin = 4, .input_end = 10, .target_begin = 8, .target_end = 10, .first_target = 4},
    }};
    const auto planned = ninfer::perplexity::plan_windows(10, 6, 2);
    failures += require(planned.size() == expected.size(), "window count matches the protocol");
    for (std::size_t index = 0; index < std::min(planned.size(), expected.size()); ++index) {
        const auto& actual = planned[index];
        const auto& wanted = expected[index];
        failures += require(actual.input_begin == wanted.input_begin &&
                                actual.input_end == wanted.input_end &&
                                actual.target_begin == wanted.target_begin &&
                                actual.target_end == wanted.target_end &&
                                actual.first_target == wanted.first_target,
                            "window boundaries match the protocol");
    }

    const std::vector<float> first{-1.0F, -2.0F};
    const std::vector<float> second{-3.0F};
    ninfer::perplexity::ScoreAggregate a;
    ninfer::perplexity::ScoreAggregate b;
    a.add(first);
    b.add(second);
    a.add(b);
    failures += require(a.scored_tokens == 3, "aggregate counts target tokens");
    failures += require(std::abs(a.total_nll - 6.0) < 1e-12, "aggregate accumulates FP64 NLL");
    failures += require(std::abs(a.mean_nll() - 2.0) < 1e-12, "aggregate computes mean NLL");
    failures += require(std::abs(a.ppl() - std::exp(2.0)) < 1e-12, "aggregate computes perplexity");

    // Top-K divergence: P = (0.7, 0.2 | tail 0.1), Q = (0.5, 0.3 | tail 0.2).
    const std::vector<float> p{std::log(0.7F), std::log(0.2F)};
    const std::vector<float> q{std::log(0.5F), std::log(0.3F)};
    const double expected_kl = 0.7 * std::log(0.7 / 0.5) + 0.2 * std::log(0.2 / 0.3) +
                               0.1 * std::log(0.1 / 0.2);
    failures += require(std::abs(ninfer::perplexity::top_k_kl_divergence(p, q) - expected_kl) < 1e-6,
                        "top-K divergence includes the tail bucket");
    failures += require(ninfer::perplexity::top_k_kl_divergence(p, p) < 1e-9,
                        "top-K divergence of equal distributions is zero");
    // Q with (numerically) no tail while P keeps 0.1 of its mass there stays finite and large.
    const std::vector<float> saturated{std::log(0.999999F), std::log(1.0e-7F)};
    const double large = ninfer::perplexity::top_k_kl_divergence(p, saturated);
    failures += require(std::isfinite(large) && large > 1.0, "top-K divergence floors the tail");

    ninfer::perplexity::DistributionAggregate d;
    for (const float value : {4.0F, 1.0F, 3.0F, 2.0F}) { d.add(value, value < 2.5F, -1.0F, -0.5F); }
    failures += require(d.positions() == 4, "distribution aggregate counts positions");
    failures += require(std::abs(d.mean_kl() - 2.5) < 1e-12, "distribution aggregate mean");
    failures += require(d.kl_quantile(0.5) == 2.0 && d.kl_quantile(0.99) == 4.0 &&
                            d.kl_quantile(0.0) == 1.0,
                        "distribution aggregate nearest-rank quantiles");
    failures += require(std::abs(d.top1_agreement() - 0.5) < 1e-12, "top-1 agreement rate");
    failures += require(std::abs(d.mean_evaluated_nll() - 1.0) < 1e-12 &&
                            std::abs(d.mean_reference_nll() - 0.5) < 1e-12,
                        "distribution aggregate NLL means");
    try {
        const Files files;
        failures += check_options(files);
        failures += check_report(files);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        ++failures;
    }

    std::cout << (failures == 0 ? "OK" : "FAIL") << " perplexity_evaluation\n";
    return failures == 0 ? 0 : 1;
}
