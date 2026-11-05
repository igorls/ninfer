#include "product/config/options.h"
#include "product/config/quantity.h"
#include "product/logging/configuration.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
namespace fs = std::filesystem;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class F>
void rejects(F&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid startup configuration was accepted");
}

ninfer::product::ServeOptions parse(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& value : args) argv.push_back(value.data());
    return ninfer::product::parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

struct Files {
    fs::path root = fs::temp_directory_path() /
                    ("ninfer-fork-startup-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

    Files() { fs::create_directories(root); }

    ~Files() {
        std::error_code error;
        fs::remove_all(root, error);
    }

    fs::path write(const char* name, const std::string& text) {
        auto path = root / name;
        std::ofstream file(path, std::ios::binary);
        file << text;
        if (!file) throw std::runtime_error("cannot write test fixture");
        return path;
    }
};

void quantities() {
    using ninfer::product::parse_capacity_quantity;
    using ninfer::product::parse_duration_quantity;
    // Each expected value is an independent exact integer, including the smallest
    // integral byte in GiB notation and the largest uint64 budget.
    require(parse_capacity_quantity("0.000000000931322574615478515625GiB") == 1,
            "thirty-place exact fraction changed");
    require(parse_capacity_quantity("17179869183.999999999068677425384521484375GiB") == UINT64_MAX,
            "maximum exact byte budget rounded or overflowed");
    require(parse_capacity_quantity("18446744073709551615B") == UINT64_MAX,
            "maximum byte count changed");
    require(parse_capacity_quantity("0.125 MiB") == 131072 &&
                parse_duration_quantity("0.001s") == 1 &&
                parse_duration_quantity("4294967.295s") == UINT32_MAX,
            "capacity/duration units changed");
    for (const char* text : {"0.1B", "0.000000000931322574615478515624GiB", "17179869184GiB",
                             "18446744073709551616B", "-1B", "1.2.3GiB"})
        rejects([&] { (void)parse_capacity_quantity(text); });
    rejects([&] { (void)parse_duration_quantity("0.0001s"); });
    rejects([&] { (void)parse_duration_quantity("4294967.296s"); });
}

void fork_options() {
    const auto options = parse({"ninfer-serve",
                                "model.ninfer",
                                "--desktop-reserve",
                                "1700MiB",
                                "--context-slack-floor",
                                "2GiB",
                                "--context-capacity",
                                "auto",
                                "--clamp-concurrency-to-pool",
                                "--model-alias",
                                "router",
                                "--model-alias",
                                "router",
                                "--model-alias",
                                "worker",
                                "--expert-trace",
                                "trace",
                                "--expert-trace-max-size",
                                "1.5GiB",
                                "--systemone-readout",
                                "averaged",
                                "--systemone-reasoning-budget",
                                "71",
                                "--systemone-escalate-entropy",
                                "0.4",
                                "--rerank-max-documents",
                                "31",
                                "--rerank-weight-exact",
                                "0.5",
                                "--rerank-weight-substitute",
                                "-0.2",
                                "--rerank-weight-complement",
                                "0.1",
                                "--rerank-weight-irrelevant",
                                "-1",
                                "--rerank-model-id",
                                "local-rerank",
                                "--repetition-penalty",
                                "1.1"});
    const auto& engine = options.launch.engine;
    require(engine.desktop_reserve_bytes == 1700ULL * 1048576 &&
                engine.min_slack_floor_bytes == 2ULL * 1073741824 &&
                engine.clamp_concurrency_to_pool,
            "desktop sizing controls were lost");
    require(options.model_aliases == std::vector<std::string>{"router", "worker"},
            "model aliases lost order or deduplication");
    require(options.expert_stats && options.expert_trace_max_bytes == 1610612736,
            "trace no longer enables bounded expert observation");
    require(options.systemone_policy.active() && options.systemone_policy.reasoning_budget == 71 &&
                options.systemone_policy.escalate_entropy == 0.4 &&
                options.rerank_max_documents == 31 && options.rerank_model_id == "local-rerank" &&
                options.rerank_weight_exact == 0.5 && options.rerank_weight_substitute == -0.2 &&
                options.rerank_weight_complement == 0.1 && options.rerank_weight_irrelevant == -1,
            "System One/rerank defaults changed");
    require(options.launch.generation.sampling.repetition_penalty == 1.1F,
            "startup repetition penalty was lost");
    const auto zero = parse(
        {"ninfer-serve", "model.ninfer", "--desktop-reserve", "0B", "--context-capacity", "8GiB", "--context-slack-floor", "4GiB"});
    require(zero.launch.engine.desktop_reserve_bytes == 0 &&
                zero.launch.engine.min_slack_floor_bytes == 0 &&
                zero.launch.generation.preserve_thinking == false,
            "explicit pool, zero reserve or closed-turn reasoning policy changed");
    for (const auto& suffix :
         std::vector<std::vector<std::string>>{{"--model-alias", "default"},
                                               {"--repetition-penalty", "0"},
                                               {"--repetition-penalty", "1e-100"},
                                               {"--rerank-weight-exact", "nan"},
                                               {"--rerank-max-documents", "8193"},
                                               {"--systemone-escalate-entropy", "1.1"},
                                               {"--expert-trace-max-size", "0B"},
                                               {"--desktop-reserve-gib", "8"},
                                               {"--lm-head-draft"},
                                               {"--greedy"}}) {
        std::vector<std::string> args{"ninfer-serve", "model.ninfer"};
        args.insert(args.end(), suffix.begin(), suffix.end());
        rejects([&] { (void)parse(args); });
    }
}

void files_and_sources() {
    Files files;
    files.write("key.txt", "private-file-key\r\n\t ");
    const auto config = files.write(
        "startup.toml",
        "model.path='model.ninfer'\nserver.api_key_file='key.txt'\n"
        "server.model_aliases=['file-alias']\nengine.desktop_reserve='0.5GiB'\n"
        "expert_observation.trace_directory='trace'\nexpert_observation.max_size='1GiB'\n"
        "systemone.readout='averaged'\nrerank.weight_exact=0.2\n");
    const auto options =
        parse({"ninfer-serve", "--config", config.string(), "--model-alias", "cli-alias"});
    require(options.api_key == "private-file-key" &&
                options.api_key_file == fs::absolute(files.root / "key.txt").lexically_normal() &&
                options.expert_trace_directory ==
                    fs::absolute(files.root / "trace").lexically_normal() &&
                options.model_aliases == std::vector<std::string>{"cli-alias"} &&
                options.launch.engine.desktop_reserve_bytes == 536870912,
            "file-relative paths, exact units or list replacement changed");
    const auto view  = ninfer::product::configuration_view(options);
    const auto shown = ninfer::product::format_configuration(view);
    require(shown.find("private-file-key") == std::string::npos &&
                view.argv == options.launch.provenance.argv,
            "startup configuration exposed the key contents");
    const auto override_key =
        parse({"ninfer-serve", "--config", config.string(), "--api-key", "inline-key"});
    require(override_key.api_key == "inline-key" && override_key.api_key_file.empty(),
            "higher-priority key did not replace key file");
    const auto missing = files.write(
        "missing.toml", "model.path='model.ninfer'\nserver.api_key_file='absent.txt'\n");
    const auto bypass =
        parse({"ninfer-serve", "--config", missing.string(), "--api-key", "override"});
    require(bypass.api_key == "override", "overridden key file was opened");
    rejects([&] {
        (void)parse({"ninfer-serve", "--config", config.string(), "--api-key", "x",
                     "--api-key-file", "key.txt"});
    });
    auto inactive = ninfer::product::parse_cli_options;
    std::vector<std::string> args{"ninfer", "--config", missing.string(), "--prompt", "test"};
    std::vector<char*> argv;
    for (auto& value : args) argv.push_back(value.data());
    (void)inactive(static_cast<int>(argv.size()), argv.data());
    const auto malformed =
        files.write("bad.toml", "model.path='model.ninfer'\nserver.model_aliases=['default']\n");
    rejects([&] { (void)parse({"ninfer-serve", "--config", malformed.string()}); });
}

void child_arguments() {
    using namespace ninfer::product;
    auto args = set_serve_argument({"model.ninfer", "--api-key", "--port"}, "server.port", "9001");
    auto options = parse([&] {
        std::vector<std::string> full{"ninfer-serve"};
        full.insert(full.end(), args.begin(), args.end());
        return full;
    }());
    require(options.api_key == "--port" && options.port == 9001,
            "child edit confused a literal key value with an option");
    require(has_serve_argument(args, "server.api_key") && has_serve_argument(args, "server.port"),
            "child argument inspection lost an explicit setting");
    args = set_serve_argument({"model.ninfer", "--port", "8000", "--port=8001"}, "server.port",
                              "9002");
    require(args.back() == "--port=9002", "child edit did not update the winning duplicate scalar");
    args = set_serve_argument({"--", "--model.ninfer"}, "server.port", "9003");
    require(args == std::vector<std::string>{"--port", "9003", "--", "--model.ninfer"},
            "child edit inserted an option after the end-of-options marker");
    args = remove_serve_argument({"model.ninfer", "--api-key", "--desktop-reserve",
                                  "--desktop-reserve", "4GiB", "--desktop-reserve=5GiB"},
                                 "engine.desktop_reserve");
    require(args == std::vector<std::string>{"model.ninfer", "--api-key", "--desktop-reserve"},
            "reserve removal changed a literal key value or left a duplicate override");
}
} // namespace

int main() {
    try {
        quantities();
        fork_options();
        files_and_sources();
        child_arguments();
        std::cout << "ok\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
