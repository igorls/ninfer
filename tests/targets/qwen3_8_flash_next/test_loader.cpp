#include "artifact/reader.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/loader.h"
#include "targets/qwen3_8_flash_next/impl/ple_index.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"
#include "tools/reference/qwen3_8_flash_next/options.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer::targets::qwen3_8_flash_next::detail;
using LoadedModel = StandaloneLoadedModel;

int test_identity_validation() {
    using ninfer::artifact::ArtifactError;
    using ninfer::artifact::ArtifactIdentity;

    // 1. Valid exact identity
    ArtifactIdentity valid_id{
        .model_id   = "qwen3.8-flash-next",
        .weights_id = "mixed-nvfp4-fp8-ple-int4",
    };
    try {
        validate_identity(valid_id);
    } catch (const std::exception& ex) {
        std::cerr << "Unexpected failure validating exact identity: " << ex.what() << "\n";
        return 1;
    }

    // 2. Invalid model_id
    ArtifactIdentity bad_model_id{
        .model_id   = "qwen3.6-27b",
        .weights_id = "mixed-nvfp4-fp8-ple-int4",
    };
    bool caught_bad_model = false;
    try {
        validate_identity(bad_model_id);
    } catch (const ArtifactError&) { caught_bad_model = true; }
    if (!caught_bad_model) {
        std::cerr << "Failed to reject invalid model_id\n";
        return 1;
    }

    // 3. Invalid weights_id
    ArtifactIdentity bad_weights_id{
        .model_id   = "qwen3.8-flash-next",
        .weights_id = "groupwise-int",
    };
    bool caught_bad_weights = false;
    try {
        validate_identity(bad_weights_id);
    } catch (const ArtifactError&) { caught_bad_weights = true; }
    if (!caught_bad_weights) {
        std::cerr << "Failed to reject invalid weights_id\n";
        return 1;
    }

    // 4. Empty identity
    ArtifactIdentity empty_id{"", ""};
    bool caught_empty = false;
    try {
        validate_identity(empty_id);
    } catch (const ArtifactError&) { caught_empty = true; }
    if (!caught_empty) {
        std::cerr << "Failed to reject empty identity\n";
        return 1;
    }

    std::cout << "PASS: test_identity_validation\n";
    return 0;
}

int test_ple_metadata_observable_behavior() {
    // Verify observable runtime PLE metadata properties and index computation
    if (kPleIndexMetadata.multipliers.size() != 3 || kPleIndexMetadata.multipliers[0] <= 0 ||
        kPleIndexMetadata.multipliers[1] <= 0 || kPleIndexMetadata.multipliers[2] <= 0) {
        std::cerr << "PLE multipliers metadata invalid\n";
        return 1;
    }

    if (kPleIndexMetadata.head_offsets.size() != 16 ||
        kPleIndexMetadata.head_vocab_sizes.size() != 16) {
        std::cerr << "PLE head metadata size mismatch\n";
        return 1;
    }

    // Monotonicity and positive vocabulary size checks
    for (std::size_t i = 0; i < 16; ++i) {
        if (kPleIndexMetadata.head_vocab_sizes[i] <= 0) {
            std::cerr << "PLE head vocab size non-positive at " << i << "\n";
            return 1;
        }
        if (i > 0 && kPleIndexMetadata.head_offsets[i] <= kPleIndexMetadata.head_offsets[i - 1]) {
            std::cerr << "PLE head offsets not strictly monotonic at " << i << "\n";
            return 1;
        }
    }

    // Verify index generation for token
    PleTokenHistory history;
    const auto indices0 = ple_indices(kPleIndexMetadata, history, 42);
    for (std::size_t i = 0; i < 16; ++i) {
        const auto offset = kPleIndexMetadata.head_offsets[i];
        const auto vocab  = kPleIndexMetadata.head_vocab_sizes[i];
        if (indices0[i] < offset || indices0[i] >= offset + vocab) {
            std::cerr << "PLE index out of head range at " << i << ": " << indices0[i] << "\n";
            return 1;
        }
    }

    std::cout << "PASS: test_ple_metadata_observable_behavior\n";
    return 0;
}

int test_preflight_memory_accounting() {
    FlashNextRuntimeConfig cfg{
        .max_concurrency     = 4,
        .max_context         = 4096,
        .state_slot_capacity = 8,
    };

    const auto curve = flash_next_capacity_curve(cfg);
    if (curve.main_page_tokens != 256) {
        std::cerr << "Main page tokens mismatch: expected 256 got " << curve.main_page_tokens
                  << "\n";
        return 1;
    }

    // 4096 tokens / 256 = 16 groups * 4 concurrency = 64 maximum groups
    if (curve.maximum_main_page_groups != 64) {
        std::cerr << "Maximum main page groups expected 64, got " << curve.maximum_main_page_groups
                  << "\n";
        return 1;
    }

    const auto plan = finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups);
    if (plan.resolved_tokens != 64 * 256) {
        std::cerr << "Resolved tokens mismatch: expected 16384 got " << plan.resolved_tokens
                  << "\n";
        return 1;
    }

    // Check exact affine memory stride calculation
    if (curve.bytes_per_additional_main_page_group != kPhysicalStrideBytesPerGroup) {
        std::cerr << "Bytes per additional group mismatch: expected "
                  << kPhysicalStrideBytesPerGroup << " got "
                  << curve.bytes_per_additional_main_page_group << "\n";
        return 1;
    }
    if (kPhysicalStrideBytesPerGroup != 6'488'064ULL) {
        std::cerr << "kPhysicalStrideBytesPerGroup constant mismatch: expected 6488064 got "
                  << kPhysicalStrideBytesPerGroup << "\n";
        return 1;
    }

    // Out of bounds page group rejection
    try {
        (void)finalize_flash_next_runtime_plan(cfg, curve.minimum_main_page_groups - 1);
        std::cerr << "Failed to reject under-minimum main page groups\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    try {
        (void)finalize_flash_next_runtime_plan(cfg, curve.maximum_main_page_groups + 1);
        std::cerr << "Failed to reject over-maximum main page groups\n";
        return 1;
    } catch (const std::invalid_argument&) {}

    std::cout << "PASS: test_preflight_memory_accounting\n";
    return 0;
}

int test_options_parser_validation() {
    // 1. Valid invocation path without materialization
    const std::vector<std::string_view> valid_args = {
        "--model", "model.ninfer", "--preflight", "--max-context", "2048", "--max-concurrency", "2",
    };
    try {
        const auto opts = parse_reference_tool_options(valid_args);
        if (opts.model_path != "model.ninfer" || opts.mode != "preflight" ||
            opts.max_context != 2048 || opts.max_concurrency != 2 || opts.state_slots != 4) {
            std::cerr << "Parsed options mismatch on valid args\n";
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Unexpected failure parsing valid args: " << ex.what() << "\n";
        return 1;
    }

    const std::vector<std::string_view> full_args = {
        "--model", "model.ninfer", "--materialize-full", "--max-context", "4096",
    };
    try {
        const auto opts = parse_reference_tool_options(full_args);
        if (opts.model_path != "model.ninfer" || opts.mode != "materialize-full" ||
            opts.max_context != 4096) {
            std::cerr << "Parsed options mismatch on materialize-full args\n";
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Unexpected failure parsing materialize-full args: " << ex.what() << "\n";
        return 1;
    }

    const std::vector<std::string_view> vision_args = {
        "--model", "model.ninfer", "--materialize-vision", "--max-context", "4096",
    };
    try {
        const auto opts = parse_reference_tool_options(vision_args);
        if (opts.model_path != "model.ninfer" || opts.mode != "materialize-vision" ||
            opts.max_context != 4096) {
            std::cerr << "Parsed options mismatch on materialize-vision args\n";
            return 1;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Unexpected failure parsing materialize-vision args: " << ex.what() << "\n";
        return 1;
    }

    // Helper lambda to test that invalid arguments throw std::invalid_argument
    const auto assert_throws = [](std::initializer_list<std::string_view> args,
                                  std::string_view label) -> bool {
        try {
            std::vector<std::string_view> vec(args);
            (void)parse_reference_tool_options(vec);
            std::cerr << "Failed to reject " << label << "\n";
            return false;
        } catch (const std::invalid_argument&) { return true; }
    };

    // 2. Negative unsigned input
    if (!assert_throws({"--model", "m.ninfer", "--max-context", "-1"}, "negative max-context"))
        return 1;
    if (!assert_throws({"--model", "m.ninfer", "--max-concurrency", "-2"},
                       "negative max-concurrency"))
        return 1;
    if (!assert_throws({"--model", "m.ninfer", "--page-groups", "-10"}, "negative page-groups"))
        return 1;
    if (!assert_throws({"--model", "m.ninfer", "--state-slots", "-4"}, "negative state-slots"))
        return 1;

    // 3. Trailing junk
    if (!assert_throws({"--model", "m.ninfer", "--max-context", "4096abc"},
                       "trailing junk in context"))
        return 1;
    if (!assert_throws({"--model", "m.ninfer", "--token-id", "100xyz"},
                       "trailing junk in token-id"))
        return 1;

    // 4. Integer overflow
    if (!assert_throws({"--model", "m.ninfer", "--max-context", "999999999999999999999"},
                       "overflow in context"))
        return 1;

    // 5. Out-of-range concurrency
    if (!assert_throws({"--model", "m.ninfer", "--max-concurrency", "0"}, "concurrency 0"))
        return 1;
    if (!assert_throws({"--model", "m.ninfer", "--max-concurrency", "9"}, "concurrency 9"))
        return 1;

    // 6. Out-of-range token ID [0, 248320)
    if (!assert_throws({"--model", "m.ninfer", "--token-id", "-1"}, "negative token-id")) return 1;
    if (!assert_throws({"--model", "m.ninfer", "--token-id", "248320"}, "token-id == vocab_size"))
        return 1;
    if (!assert_throws({"--model", "m.ninfer", "--token-id", "300000"}, "token-id > vocab_size"))
        return 1;

    // 7. Missing value
    if (!assert_throws({"--model"}, "missing model path value")) return 1;
    if (!assert_throws({"--model", "m.ninfer", "--max-context"}, "missing context value")) return 1;

    // 8. Unknown option
    if (!assert_throws({"--model", "m.ninfer", "--invalid-option"}, "unknown option")) return 1;

    // 9. Missing required model path
    if (!assert_throws({"--preflight"}, "missing model path")) return 1;

    std::cout << "PASS: test_options_parser_validation\n";
    return 0;
}

int test_real_artifact_preflight_if_available() {
    const char* env_path = std::getenv("NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS");
    std::filesystem::path path =
        env_path != nullptr && *env_path != '\0'
            ? std::filesystem::path(env_path)
            : std::filesystem::path(
                  "" /* real-artifact cases run only when NINFER_WEIGHTS is set explicitly */);

    if (!std::filesystem::is_regular_file(path)) {
        std::cout << "SKIP: Real artifact preflight (artifact not present at " << path << ")\n";
        return 0;
    }

    FlashNextRuntimeConfig cfg{
        .max_concurrency     = 1,
        .max_context         = 4096,
        .state_slot_capacity = 2,
    };

    const auto report = preflight_text_file(path, cfg);

    // Exact delivered contract verification
    if (report.identity.model_id != kExpectedModelId ||
        report.identity.weights_id != kExpectedWeightsId) {
        std::cerr << "Report identity mismatch: " << report.identity.model_id << "/"
                  << report.identity.weights_id << "\n";
        return 1;
    }
    if (report.file_bytes != 113'298'397'952ULL) {
        std::cerr << "Preflight file_bytes mismatch: expected 113298397952 got "
                  << report.file_bytes << "\n";
        return 1;
    }
    if (report.planned_device_weights_bytes != 75'172'951'040ULL) {
        std::cerr << "Preflight planned_device_weights_bytes mismatch: expected 75172951040 got "
                  << report.planned_device_weights_bytes << "\n";
        return 1;
    }
    if (report.planned_device_tensors_count != 1067) {
        std::cerr << "Preflight planned_device_tensors_count mismatch: expected 1067 got "
                  << report.planned_device_tensors_count << "\n";
        return 1;
    }
    if (report.planned_retained_resources_count != 6) {
        std::cerr << "Preflight planned_retained_resources_count mismatch: expected 6 got "
                  << report.planned_retained_resources_count << "\n";
        return 1;
    }
    if (report.planned_mapped_tensors_count != 131) { // 128 shards + 3 embedding metadata tensors
        std::cerr << "Preflight planned_mapped_tensors_count mismatch: expected 131 got "
                  << report.planned_mapped_tensors_count << "\n";
        return 1;
    }
    if (report.runtime_plan.total_device_bytes != 509'714'432ULL) {
        std::cerr << "Preflight runtime total_device_bytes mismatch: expected 509714432 got "
                  << report.runtime_plan.total_device_bytes << "\n";
        return 1;
    }
    if (report.runtime_plan.attention_kv_bytes != 100'663'296ULL ||
        report.runtime_plan.indexer_block_keys_bytes != 3'145'728ULL ||
        report.runtime_plan.recurrent_state_bytes != 231'312'384ULL ||
        report.runtime_plan.workspace_bytes != 174'084'096ULL) {
        std::cerr << "Preflight runtime plan sub-allocations mismatch\n";
        return 1;
    }

    std::cout << "PASS: test_real_artifact_preflight\n";
    return 0;
}

int test_real_artifact_full_binding_if_available() {
    const char* env_path = std::getenv("NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS");
    const std::filesystem::path path =
        env_path != nullptr && *env_path != '\0'
            ? std::filesystem::path(env_path)
            : std::filesystem::path(
                  "" /* real-artifact cases run only when NINFER_WEIGHTS is set explicitly */);

    if (!std::filesystem::is_regular_file(path)) {
        std::cout << "SKIP: Real artifact full binding (artifact not present at " << path << ")\n";
        return 0;
    }

    const ninfer::artifact::Reader reader(path);
    validate_identity(reader.identity());
    ninfer::artifact::Binder binder(reader);
    const auto full_plan = bind_artifact(binder, LoadFeatures{.vision = true, .mtp = true});

    std::uint64_t tensor_bytes = 0;
    for (const auto& object : full_plan.materialization.device_objects) {
        tensor_bytes += object.bytes;
    }

    if (tensor_bytes != 81'285'103'328ULL ||
        full_plan.materialization.device_capacity_bytes != 81'285'117'440ULL ||
        full_plan.materialization.device_objects.size() != 1'429 ||
        full_plan.materialization.host_objects.size() != 6 ||
        full_plan.materialization.mapped_tensor_objects.size() != 131) {
        std::cerr << "Full artifact binding inventory mismatch\n";
        return 1;
    }

    std::cout << "PASS: test_real_artifact_full_binding\n";
    return 0;
}

int test_real_artifact_text_and_vision_plan_if_available() {
    const char* env_path = std::getenv("NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS");
    const std::filesystem::path path =
        env_path != nullptr && *env_path != '\0'
            ? std::filesystem::path(env_path)
            : std::filesystem::path(
                  "" /* real-artifact cases run only when NINFER_WEIGHTS is set explicitly */);

    if (!std::filesystem::is_regular_file(path)) {
        std::cout << "SKIP: Real artifact text+vision plan (artifact not present at " << path
                  << ")\n";
        return 0;
    }

    const ninfer::artifact::Reader reader(path);
    validate_identity(reader.identity());
    ninfer::artifact::Binder binder(reader);
    const auto tv_plan = bind_artifact(binder, LoadFeatures{.vision = true, .mtp = false});

    std::uint64_t tensor_bytes = 0;
    for (const auto& object : tv_plan.materialization.device_objects) {
        tensor_bytes += object.bytes;
    }

    if (tv_plan.materialization.device_objects.size() != 1'400) {
        std::cerr << "Text+Vision device_objects count mismatch: expected 1400, got "
                  << tv_plan.materialization.device_objects.size() << "\n";
        return 1;
    }
    if (tv_plan.materialization.host_objects.size() != 6) {
        std::cerr << "Text+Vision host_objects count mismatch: expected 6, got "
                  << tv_plan.materialization.host_objects.size() << "\n";
        return 1;
    }
    if (tv_plan.materialization.mapped_tensor_objects.size() != 131) {
        std::cerr << "Text+Vision mapped_tensor_objects count mismatch: expected 131, got "
                  << tv_plan.materialization.mapped_tensor_objects.size() << "\n";
        return 1;
    }

    constexpr std::uint64_t kExpectedTvTensorBytes = 76'070'801'632ULL;
    constexpr std::uint64_t kExpectedTvArenaBytes  = 76'070'815'744ULL;
    if (tensor_bytes != kExpectedTvTensorBytes ||
        tv_plan.materialization.device_capacity_bytes != kExpectedTvArenaBytes) {
        std::cerr << "Text+Vision tensor bytes mismatch: expected " << kExpectedTvTensorBytes
                  << " bytes (" << kExpectedTvArenaBytes << " arena), got " << tensor_bytes
                  << " bytes (" << tv_plan.materialization.device_capacity_bytes << " arena)\n";
        return 1;
    }

    std::cout << "PASS: test_real_artifact_text_and_vision_plan\n";
    return 0;
}

} // namespace

int main() {
    if (test_identity_validation() != 0) return 1;
    if (test_ple_metadata_observable_behavior() != 0) return 1;
    if (test_preflight_memory_accounting() != 0) return 1;
    if (test_options_parser_validation() != 0) return 1;
    if (test_real_artifact_preflight_if_available() != 0) return 1;
    if (test_real_artifact_text_and_vision_plan_if_available() != 0) return 1;
    if (test_real_artifact_full_binding_if_available() != 0) return 1;

    std::cout << "OK Flash-Next Native Loader Tests\n";
    return 0;
}
