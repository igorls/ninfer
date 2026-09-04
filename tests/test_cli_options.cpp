#include "options.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

ninfer::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const ninfer::cli::Options configured =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "37"});
    failures += check(configured.thinking_budget == 37,
                      "--thinking-budget did not preserve its positive value");
    failures +=
        check(ninfer::cli::usage_text("ninfer-cli").find("--thinking-budget") != std::string::npos,
              "CLI help omits --thinking-budget");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "0"});
                      }),
                      "zero --thinking-budget was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--thinking-budget", "8", "--no-thinking"});
                      }),
                      "--thinking-budget was accepted with --no-thinking");
    const ninfer::cli::Options with_effort =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--thinking-budget", "8",
               "--reasoning-effort", "medium"});
    failures += check(with_effort.thinking_budget == 8 && with_effort.reasoning_effort,
                      "thinking budget did not coexist with reasoning effort");
    const ninfer::cli::Options nvfp4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == ninfer::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ninfer::cli::Options k8v4 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == ninfer::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    const std::string help = ninfer::cli::usage_text("ninfer-cli");
    failures +=
        check(help.find("nvfp4") != std::string::npos && help.find("k8v4") != std::string::npos,
              "CLI help omits a production KV storage mode");
    failures += check(help.find("--output-head-fp8") != std::string::npos,
                      "CLI help omits --output-head-fp8");
    failures += check(help.find("--output-head-dtype") != std::string::npos,
                      "CLI help omits --output-head-dtype");

    const ninfer::cli::Options default_cli =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello"});
    failures += check(!default_cli.quantize_output_head_fp8,
                      "CLI output head FP8 quantization is unexpectedly enabled by default");

    const ninfer::cli::Options cli_head_fp8 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--output-head-fp8"});
    failures += check(cli_head_fp8.quantize_output_head_fp8,
                      "--output-head-fp8 did not enable output head FP8 quantization");

    const ninfer::cli::Options cli_no_head_fp8 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--output-head-fp8", "--no-output-head-fp8"});
    failures += check(!cli_no_head_fp8.quantize_output_head_fp8,
                      "--no-output-head-fp8 did not disable output head FP8 quantization");

    const ninfer::cli::Options cli_head_dtype_fp8 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--output-head-dtype", "fp8"});
    failures += check(cli_head_dtype_fp8.quantize_output_head_fp8,
                      "--output-head-dtype fp8 did not enable output head FP8 quantization");

    const ninfer::cli::Options cli_head_dtype_bf16 =
        parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--output-head-fp8", "--output-head-dtype", "bf16"});
    failures += check(!cli_head_dtype_bf16.quantize_output_head_fp8,
                      "--output-head-dtype bf16 did not disable output head FP8 quantization");

    failures += check(rejects([] {
                          (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello",
                                       "--output-head-dtype", "invalid"});
                      }),
                      "--output-head-dtype invalid did not reject");

    failures +=
        check(rejects([] {
                  (void)parse({"ninfer-cli", "model.ninfer", "--prompt", "hello", "--top-k", "21"});
              }),
              "CLI accepted top_k beyond the executable candidate domain");
    return failures == 0 ? 0 : 1;
}
