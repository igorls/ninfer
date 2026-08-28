#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/loader.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"
#include "targets/qwen3_8_flash_next/impl/runtime_state.h"
#include "targets/qwen3_8_flash_next/impl/text_executor.h"
#include "options.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer::targets::qwen3_8_flash_next::detail;

float bf16_to_float(std::uint16_t val) {
    const std::uint32_t bits = static_cast<std::uint32_t>(val) << 16U;
    float f                  = 0.0f;
    std::memcpy(&f, &bits, sizeof(float));
    return f;
}

std::uint64_t fnv1a_64(const void* data, std::size_t size) {
    const auto* bytes             = static_cast<const std::uint8_t*>(data);
    std::uint64_t hash            = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t prime = 0x100000001b3ULL;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= prime;
    }
    return hash;
}

int run_preflight(const ReferenceToolOptions& opts) {
    FlashNextRuntimeConfig config{
        .max_concurrency     = opts.max_concurrency,
        .max_context         = opts.max_context,
        .state_slot_capacity = opts.state_slots,
    };

    const auto report = preflight_text_file(opts.model_path, config, opts.page_groups);

    if (opts.json_output) {
        std::cout
            << "{\n"
            << "  \"model_id\": \"" << report.identity.model_id << "\",\n"
            << "  \"weights_id\": \"" << report.identity.weights_id << "\",\n"
            << "  \"file_bytes\": " << report.file_bytes << ",\n"
            << "  \"planned_device_weights_bytes\": " << report.planned_device_weights_bytes
            << ",\n"
            << "  \"planned_device_tensors\": " << report.planned_device_tensors_count << ",\n"
            << "  \"planned_retained_resources\": " << report.planned_retained_resources_count
            << ",\n"
            << "  \"planned_mapped_tensors\": " << report.planned_mapped_tensors_count << ",\n"
            << "  \"runtime\": {\n"
            << "    \"max_concurrency\": " << report.runtime_plan.config.max_concurrency << ",\n"
            << "    \"max_context\": " << report.runtime_plan.config.max_context << ",\n"
            << "    \"main_page_groups\": " << report.runtime_plan.main_page_groups << ",\n"
            << "    \"resolved_tokens\": " << report.runtime_plan.resolved_tokens << ",\n"
            << "    \"attention_physical_pages\": " << report.runtime_plan.attention_physical_pages
            << ",\n"
            << "    \"indexer_physical_pages\": " << report.runtime_plan.indexer_physical_pages
            << ",\n"
            << "    \"attention_logical_pages\": " << report.runtime_plan.attention_logical_pages
            << ",\n"
            << "    \"indexer_logical_pages\": " << report.runtime_plan.indexer_logical_pages
            << ",\n"
            << "    \"state_slots\": " << report.runtime_plan.state_slots << ",\n"
            << "    \"attention_kv_bytes\": " << report.runtime_plan.attention_kv_bytes << ",\n"
            << "    \"indexer_block_keys_bytes\": " << report.runtime_plan.indexer_block_keys_bytes
            << ",\n"
            << "    \"recurrent_state_bytes\": " << report.runtime_plan.recurrent_state_bytes
            << ",\n"
            << "    \"workspace_bytes\": " << report.runtime_plan.workspace_bytes << ",\n"
            << "    \"total_device_bytes\": " << report.runtime_plan.total_device_bytes << "\n"
            << "  }\n"
            << "}\n";
    } else {
        std::cout
            << "=== Qwen3.8-Flash-Next Preflight Report ===\n"
            << "Identity:                     " << report.identity.model_id << " / "
            << report.identity.weights_id << "\n"
            << "Artifact File Size:           " << (report.file_bytes / (1024.0 * 1024.0 * 1024.0))
            << " GiB (" << report.file_bytes << " bytes)\n"
            << "Planned Device Weights:       "
            << (report.planned_device_weights_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB ("
            << report.planned_device_weights_bytes << " bytes)\n"
            << "Planned Device Tensors:       " << report.planned_device_tensors_count << "\n"
            << "Planned Retained Resources:   " << report.planned_retained_resources_count << "\n"
            << "Planned Mapped PLE Shards:    " << report.planned_mapped_tensors_count << "\n\n"
            << "--- Runtime Capacity Plan ---\n"
            << "Max Concurrency:              " << report.runtime_plan.config.max_concurrency
            << "\n"
            << "Max Context Tokens:           " << report.runtime_plan.config.max_context << "\n"
            << "Main Page Groups (256 tok):   " << report.runtime_plan.main_page_groups << "\n"
            << "Resolved Capacity Tokens:     " << report.runtime_plan.resolved_tokens << "\n"
            << "Attention Physical Pages:     " << report.runtime_plan.attention_physical_pages
            << " (64 tok/page)\n"
            << "Indexer Physical Pages:       " << report.runtime_plan.indexer_physical_pages
            << " (256 tok/page)\n"
            << "Attention Logical Pages:      " << report.runtime_plan.attention_logical_pages
            << "\n"
            << "Indexer Logical Pages:        " << report.runtime_plan.indexer_logical_pages << "\n"
            << "State Slots:                  " << report.runtime_plan.state_slots << "\n"
            << "Attention KV Size:            "
            << (report.runtime_plan.attention_kv_bytes / (1024.0 * 1024.0)) << " MiB\n"
            << "Indexer Block Keys:           "
            << (report.runtime_plan.indexer_block_keys_bytes / (1024.0 * 1024.0)) << " MiB\n"
            << "Recurrent State Size:         "
            << (report.runtime_plan.recurrent_state_bytes / (1024.0 * 1024.0)) << " MiB\n"
            << "Runtime Workspace Size:       "
            << (report.runtime_plan.workspace_bytes / (1024.0 * 1024.0)) << " MiB\n"
            << "Total Runtime Device Memory:  "
            << (report.runtime_plan.total_device_bytes / (1024.0 * 1024.0)) << " MiB ("
            << report.runtime_plan.total_device_bytes << " bytes)\n"
            << "Preflight Status:             OK\n";
    }
    return 0;
}

int run_execute_token(const ReferenceToolOptions& opts) {
    int device_count = 0;
    const auto err   = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "Error: No usable CUDA device available for execute-token mode\n";
        return 1;
    }

    ninfer::DeviceContext device(0);

    FlashNextRuntimeConfig config{
        .max_concurrency     = opts.max_concurrency,
        .max_context         = opts.max_context,
        .state_slot_capacity = opts.state_slots,
    };

    const auto curve = flash_next_capacity_curve(config);
    const std::uint32_t resolved_groups =
        opts.page_groups == 0 ? curve.maximum_main_page_groups : opts.page_groups;
    auto runtime_plan = finalize_flash_next_runtime_plan(config, resolved_groups);

    if (!opts.json_output) {
        std::cout << "Loading and materializing model artifact: " << opts.model_path << " ...\n";
    } else {
        std::cerr << "Loading and materializing model artifact: " << opts.model_path << " ...\n";
    }
    auto model = LoadedModel::load_from_file(opts.model_path, device);

    if (!opts.json_output) {
        std::cout << "Allocating runtime buffers ("
                  << (runtime_plan.total_device_bytes / (1024.0 * 1024.0)) << " MiB) ...\n";
    }
    FlashNextRuntimeAllocation alloc(runtime_plan);
    alloc.initialize(device.stream);

    FlashNextTextExecutor executor(model.text_view(), model.ple_metadata(), device, alloc);

    auto lane = executor.allocate_lane();

    LaneStepRequest req{
        .handle          = lane,
        .token_id        = opts.token_id,
        .token_index     = 0,
        .mrope_positions = {0, 0, 0},
    };

    std::array<LaneStepRequest, 1> reqs{req};
    if (!opts.json_output) {
        std::cout << "Executing token round for token_id=" << opts.token_id
                  << " at position=0 ...\n";
    }
    auto round = executor.execute_round(reqs);
    device.synchronize();

    constexpr std::size_t kVocabSize = 248'320;
    std::vector<std::uint16_t> host_logits_bf16(kVocabSize);
    CUDA_CHECK(cudaMemcpy(host_logits_bf16.data(), round.logits().data,
                          kVocabSize * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));

    std::int32_t best_token = 0;
    float best_logit        = -1e30f;
    double logit_sum        = 0.0;

    for (std::size_t i = 0; i < kVocabSize; ++i) {
        const float val = bf16_to_float(host_logits_bf16[i]);
        logit_sum += val;
        if (val > best_logit) {
            best_logit = val;
            best_token = static_cast<std::int32_t>(i);
        }
    }

    const std::uint64_t checksum =
        fnv1a_64(host_logits_bf16.data(), host_logits_bf16.size() * sizeof(std::uint16_t));

    if (opts.do_commit) {
        std::vector<LaneCommitDecision> decisions = {{.accept = true}};
        round.commit(decisions);
        if (!opts.json_output) { std::cout << "Transaction: committed (frontier advanced to 1)\n"; }
    } else {
        round.abort();
        if (!opts.json_output) { std::cout << "Transaction: aborted (frontier remains 0)\n"; }
    }

    executor.release_lane(lane);

    if (opts.json_output) {
        std::cout << "{\n"
                  << "  \"token_id\": " << opts.token_id << ",\n"
                  << "  \"vocab_size\": " << kVocabSize << ",\n"
                  << "  \"argmax_token\": " << best_token << ",\n"
                  << "  \"argmax_logit\": " << best_logit << ",\n"
                  << "  \"logits_sum\": " << logit_sum << ",\n"
                  << "  \"logits_checksum\": \"" << std::hex << checksum << std::dec << "\",\n"
                  << "  \"committed\": " << (opts.do_commit ? "true" : "false") << ",\n"
                  << "  \"status\": \"OK\"\n"
                  << "}\n";
    } else {
        std::cout << "\n=== Execution Results ===\n"
                  << "Token ID:                     " << opts.token_id << "\n"
                  << "Vocab Size:                   " << kVocabSize << "\n"
                  << "Argmax Token:                 " << best_token << "\n"
                  << "Argmax Logit Value:           " << std::fixed << std::setprecision(4)
                  << best_logit << "\n"
                  << "Logits Sum:                   " << logit_sum << "\n"
                  << "Logits FNV-1a Checksum:       0x" << std::hex << checksum << std::dec << "\n"
                  << "Execution Status:             OK\n";
    }

    return 0;
}

int run_materialize_full(const ReferenceToolOptions& opts) {
    int device_count = 0;
    const auto err   = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "Error: No usable CUDA device available for materialize-full mode\n";
        return 1;
    }

    std::size_t free_before  = 0;
    std::size_t total_before = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_before));

    ninfer::DeviceContext device(0);
    const ninfer::artifact::Reader reader(opts.model_path);
    validate_identity(reader.identity());

    ninfer::artifact::Binder binder(reader);
    const auto load_plan = bind_artifact(binder, LoadFeatures{.vision = true, .mtp = true});

    const std::uint64_t file_bytes = reader.file_bytes();
    const std::uint64_t planned_device_weights_bytes =
        load_plan.materialization.device_capacity_bytes;
    const std::size_t planned_device_tensors     = load_plan.materialization.device_objects.size();
    const std::size_t planned_retained_resources = load_plan.materialization.host_objects.size();
    const std::size_t planned_mapped_tensors =
        load_plan.materialization.mapped_tensor_objects.size();

    std::uint64_t planned_device_tensor_bytes = 0;
    for (const auto& object : load_plan.materialization.device_objects) {
        planned_device_tensor_bytes += object.bytes;
    }

    constexpr std::uint64_t kExpectedFullDeviceTensorBytes = 81'285'103'328ULL;
    constexpr std::uint64_t kExpectedFullDeviceArenaBytes  = 81'285'117'440ULL;
    constexpr std::size_t kExpectedFullDeviceTensors       = 1'429;
    constexpr std::size_t kExpectedRetainedResources       = 6;
    constexpr std::size_t kExpectedMappedTensors           = 131;

    if (planned_device_tensor_bytes != kExpectedFullDeviceTensorBytes ||
        planned_device_weights_bytes != kExpectedFullDeviceArenaBytes ||
        planned_device_tensors != kExpectedFullDeviceTensors ||
        planned_retained_resources != kExpectedRetainedResources ||
        planned_mapped_tensors != kExpectedMappedTensors) {
        throw std::runtime_error(
            "Full inventory validation failed: got tensor_bytes=" +
            std::to_string(planned_device_tensor_bytes) +
            " arena_bytes=" + std::to_string(planned_device_weights_bytes) +
            " device_tensors=" + std::to_string(planned_device_tensors) +
            " retained_resources=" + std::to_string(planned_retained_resources) +
            " mapped_tensors=" + std::to_string(planned_mapped_tensors));
    }

    if (!opts.json_output) {
        std::cout << "Materializing full model artifact (Text + MTP + Vision: "
                  << (planned_device_weights_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB) ...\n";
    } else {
        std::cerr << "Materializing full model artifact: " << opts.model_path << " ...\n";
    }
    auto materialized = ninfer::artifact::materialize(reader, load_plan.materialization, device);
    device.synchronize();

    FlashNextRuntimeConfig config{
        .max_concurrency     = opts.max_concurrency,
        .max_context         = opts.max_context,
        .state_slot_capacity = opts.state_slots,
    };
    const auto curve = flash_next_capacity_curve(config);
    const std::uint32_t resolved_groups =
        opts.page_groups == 0 ? curve.maximum_main_page_groups : opts.page_groups;
    auto runtime_plan = finalize_flash_next_runtime_plan(config, resolved_groups);

    if (!opts.json_output) {
        std::cout << "Allocating text runtime buffers ("
                  << (runtime_plan.total_device_bytes / (1024.0 * 1024.0)) << " MiB) ...\n";
    }
    FlashNextRuntimeAllocation allocation(runtime_plan);
    allocation.initialize(device.stream);
    device.synchronize();

    std::size_t free_resident  = 0;
    std::size_t total_resident = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_resident, &total_resident));
    (void)materialized;

    if (opts.json_output) {
        std::cout << "{\n"
                  << "  \"mode\": \"materialize-full\",\n"
                  << "  \"model_id\": \"" << reader.identity().model_id << "\",\n"
                  << "  \"weights_id\": \"" << reader.identity().weights_id << "\",\n"
                  << "  \"file_bytes\": " << file_bytes << ",\n"
                  << "  \"planned_device_tensor_bytes\": " << planned_device_tensor_bytes << ",\n"
                  << "  \"planned_device_weights_bytes\": " << planned_device_weights_bytes << ",\n"
                  << "  \"planned_device_tensors\": " << planned_device_tensors << ",\n"
                  << "  \"planned_retained_resources\": " << planned_retained_resources << ",\n"
                  << "  \"planned_mapped_tensors\": " << planned_mapped_tensors << ",\n"
                  << "  \"text_runtime_device_bytes\": " << runtime_plan.total_device_bytes << ",\n"
                  << "  \"cuda_free_bytes_before\": " << free_before << ",\n"
                  << "  \"cuda_total_bytes_before\": " << total_before << ",\n"
                  << "  \"cuda_free_bytes_resident\": " << free_resident << ",\n"
                  << "  \"cuda_total_bytes_resident\": " << total_resident << ",\n"
                  << "  \"status\": \"OK\"\n"
                  << "}\n";
    } else {
        std::cout << "\n=== Full Artifact GPU Residency Report ===\n"
                  << "Identity:                     " << reader.identity().model_id << " / "
                  << reader.identity().weights_id << "\n"
                  << "Artifact File Size:           " << (file_bytes / (1024.0 * 1024.0 * 1024.0))
                  << " GiB (" << file_bytes << " bytes)\n"
                  << "Full Device Tensor Payloads:  "
                  << (planned_device_tensor_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB ("
                  << planned_device_tensor_bytes << " bytes)\n"
                  << "Full Device Weights Arena:    "
                  << (planned_device_weights_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB ("
                  << planned_device_weights_bytes << " bytes)\n"
                  << "Full Device Tensors:          " << planned_device_tensors << "\n"
                  << "Retained Resources:           " << planned_retained_resources << "\n"
                  << "Mapped PLE Tensors:           " << planned_mapped_tensors << "\n"
                  << "Text Runtime Device Memory:   "
                  << (runtime_plan.total_device_bytes / (1024.0 * 1024.0)) << " MiB ("
                  << runtime_plan.total_device_bytes << " bytes)\n"
                  << "CUDA Free (Before Load):      " << (free_before / (1024.0 * 1024.0))
                  << " MiB / " << (total_before / (1024.0 * 1024.0)) << " MiB\n"
                  << "CUDA Free (Fully Resident):   " << (free_resident / (1024.0 * 1024.0))
                  << " MiB / " << (total_resident / (1024.0 * 1024.0)) << " MiB\n"
                  << "Residency Status:             OK\n";
    }

    return 0;
}

int run_materialize_vision(const ReferenceToolOptions& opts) {
    int device_count = 0;
    const auto err   = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "Error: No usable CUDA device available for materialize-vision mode\n";
        return 1;
    }

    std::size_t free_before  = 0;
    std::size_t total_before = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_before));

    ninfer::DeviceContext device(0);

    if (!opts.json_output) {
        std::cout << "Loading and materializing Text + Vision model: " << opts.model_path
                  << " ...\n";
    } else {
        std::cerr << "Loading and materializing Text + Vision model: " << opts.model_path
                  << " ...\n";
    }

    auto model = LoadedModel::load_from_file(opts.model_path, device,
                                             LoadFeatures{.vision = true, .mtp = false});
    device.synchronize();

    if (!model.has_vision()) {
        throw std::logic_error(
            "LoadedModel reports has_vision() is false after requesting vision load");
    }

    const auto& vision = model.vision_view();
    const auto& text   = model.text_view();

    if (text.weights_arena == nullptr) {
        throw std::logic_error("Text weights_arena pointer is null");
    }
    const auto arena_begin      = reinterpret_cast<std::uintptr_t>(text.weights_arena->base());
    const auto arena_end        = arena_begin + text.weights_arena->capacity();
    const auto in_weights_arena = [arena_begin, arena_end](const void* data, std::size_t bytes) {
        const auto begin = reinterpret_cast<std::uintptr_t>(data);
        return data != nullptr && begin >= arena_begin && begin <= arena_end &&
               bytes <= arena_end - begin;
    };
    const auto valid_bf16_weight = [&in_weights_arena](const ninfer::Weight& weight,
                                                       std::int32_t rows, std::int32_t columns) {
        return weight.qtype == ninfer::QType::BF16_CTRL &&
               in_weights_arena(weight.payload, weight.payload_bytes) && weight.ndim == 2 &&
               weight.shape[0] == rows && weight.shape[1] == columns && weight.shape[2] == 1 &&
               weight.shape[3] == 1;
    };
    const auto valid_bf16_tensor = [&in_weights_arena](const ninfer::Tensor& tensor,
                                                       std::initializer_list<std::int32_t> shape) {
        if (tensor.dtype != ninfer::DType::BF16 || !tensor.is_contiguous() ||
            !in_weights_arena(tensor.data, tensor.bytes()) || shape.size() > 4) {
            return false;
        }
        std::array<std::int32_t, 4> expected{1, 1, 1, 1};
        std::copy(shape.begin(), shape.end(), expected.begin());
        return std::equal(expected.begin(), expected.end(), std::begin(tensor.ne));
    };

    if (!valid_bf16_weight(text.token_embedding, 248'320, 2'560)) {
        throw std::logic_error("Text token_embedding is not BF16 storage in the model arena");
    }

    if (!valid_bf16_weight(vision.patch_embedding, 1'152, 1'536)) {
        throw std::logic_error("Vision patch_embedding invalid");
    }
    if (!valid_bf16_tensor(vision.patch_embedding_bias, {1'152})) {
        throw std::logic_error("Vision patch_embedding_bias invalid");
    }
    if (!valid_bf16_tensor(vision.position_embedding, {2'304, 1'152})) {
        throw std::logic_error("Vision position_embedding invalid");
    }

    for (std::size_t l = 0; l < vision.layers.size(); ++l) {
        const auto& layer = vision.layers[l];
        if (!valid_bf16_weight(layer.qkv, 3'456, 1'152) ||
            !valid_bf16_tensor(layer.qkv_bias, {3'456}) ||
            !valid_bf16_weight(layer.output, 1'152, 1'152) ||
            !valid_bf16_tensor(layer.output_bias, {1'152}) ||
            !valid_bf16_weight(layer.fc1, 4'304, 1'152) ||
            !valid_bf16_tensor(layer.fc1_bias, {4'304}) ||
            !valid_bf16_weight(layer.fc2, 1'152, 4'304) ||
            !valid_bf16_tensor(layer.fc2_bias, {1'152}) ||
            !valid_bf16_tensor(layer.norm1_weight, {1'152}) ||
            !valid_bf16_tensor(layer.norm1_bias, {1'152}) ||
            !valid_bf16_tensor(layer.norm2_weight, {1'152}) ||
            !valid_bf16_tensor(layer.norm2_bias, {1'152})) {
            throw std::logic_error("Vision layer " + std::to_string(l) + " tensor view invalid");
        }
    }

    if (!valid_bf16_weight(vision.merger_fc1, 4'608, 4'608) ||
        !valid_bf16_tensor(vision.merger_fc1_bias, {4'608}) ||
        !valid_bf16_weight(vision.merger_fc2, 2'560, 4'608) ||
        !valid_bf16_tensor(vision.merger_fc2_bias, {2'560}) ||
        !valid_bf16_tensor(vision.merger_norm_weight, {1'152}) ||
        !valid_bf16_tensor(vision.merger_norm_bias, {1'152})) {
        throw std::logic_error("Vision merger tensor view invalid");
    }

    std::size_t free_resident  = 0;
    std::size_t total_resident = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_resident, &total_resident));

    constexpr std::uint64_t kExpectedDeviceWeightsBytes = 76'070'801'632ULL;
    constexpr std::uint64_t kExpectedDeviceArenaBytes   = 76'070'815'744ULL;
    constexpr std::size_t kExpectedDeviceTensors        = 1400;
    constexpr std::size_t kExpectedRetainedResources    = 6;

    const auto& stats              = model.stats();
    const auto device_tensor_count = stats.tensor_count - stats.mapped_tensor_count;
    if (stats.h2d_bytes != kExpectedDeviceWeightsBytes ||
        stats.device_capacity_bytes != kExpectedDeviceArenaBytes ||
        text.weights_arena->capacity() != kExpectedDeviceArenaBytes ||
        device_tensor_count != kExpectedDeviceTensors ||
        stats.resource_count != kExpectedRetainedResources ||
        model.frontend_resources().size() != kExpectedRetainedResources) {
        throw std::logic_error("Text + Vision materialization statistics violate the exact plan");
    }

    if (opts.json_output) {
        std::cout << "{\n"
                  << "  \"mode\": \"materialize-vision\",\n"
                  << "  \"model_id\": \"" << kExpectedModelId << "\",\n"
                  << "  \"weights_id\": \"" << kExpectedWeightsId << "\",\n"
                  << "  \"planned_device_tensor_bytes\": " << kExpectedDeviceWeightsBytes << ",\n"
                  << "  \"planned_device_arena_bytes\": " << kExpectedDeviceArenaBytes << ",\n"
                  << "  \"planned_device_tensors\": " << kExpectedDeviceTensors << ",\n"
                  << "  \"planned_retained_resources\": " << kExpectedRetainedResources << ",\n"
                  << "  \"cuda_free_bytes_before\": " << free_before << ",\n"
                  << "  \"cuda_total_bytes_before\": " << total_before << ",\n"
                  << "  \"cuda_free_bytes_resident\": " << free_resident << ",\n"
                  << "  \"cuda_total_bytes_resident\": " << total_resident << ",\n"
                  << "  \"status\": \"OK\"\n"
                  << "}\n";
    } else {
        std::cout << "\n=== Text + Vision GPU Materialization Report ===\n"
                  << "Identity:                     " << kExpectedIdentity << "\n"
                  << "Device Tensor Payloads:       "
                  << (kExpectedDeviceWeightsBytes / (1024.0 * 1024.0 * 1024.0)) << " GiB ("
                  << kExpectedDeviceWeightsBytes << " bytes)\n"
                  << "Device Weights Arena:         "
                  << (kExpectedDeviceArenaBytes / (1024.0 * 1024.0 * 1024.0)) << " GiB ("
                  << kExpectedDeviceArenaBytes << " bytes)\n"
                  << "Device Tensors:               " << kExpectedDeviceTensors
                  << " (1067 text + 333 vision)\n"
                  << "Retained Resources:           " << kExpectedRetainedResources << "\n"
                  << "CUDA Free (Before Load):      " << (free_before / (1024.0 * 1024.0))
                  << " MiB / " << (total_before / (1024.0 * 1024.0)) << " MiB\n"
                  << "CUDA Free (Fully Resident):   " << (free_resident / (1024.0 * 1024.0))
                  << " MiB / " << (total_resident / (1024.0 * 1024.0)) << " MiB\n"
                  << "Materialization Status:       OK\n";
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto opts = parse_reference_tool_options(argc, argv);
        if (opts.mode == "preflight") {
            return run_preflight(opts);
        } else if (opts.mode == "execute-token") {
            return run_execute_token(opts);
        } else if (opts.mode == "materialize-full") {
            return run_materialize_full(opts);
        } else if (opts.mode == "materialize-vision") {
            return run_materialize_vision(opts);
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
