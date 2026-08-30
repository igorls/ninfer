#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "targets/qwen3_8_flash_next/impl/frontend.h"
#include "targets/qwen3_8_flash_next/impl/load/bindings.h"
#include "targets/qwen3_8_flash_next/impl/load/loader.h"
#include "targets/qwen3_8_flash_next/impl/runtime_plan.h"
#include "targets/qwen3_8_flash_next/impl/runtime_state.h"
#include "targets/qwen3_8_flash_next/impl/text_decode.h"
#include "targets/qwen3_8_flash_next/impl/text_executor.h"
#include "targets/qwen3_8_flash_next/impl/vision_adapter.h"
#include "targets/qwen3_8_flash_next/impl/program_impl.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/targets/qwen3_6/frontend.h"
#include "ninfer/targets/qwen3_6/frontend_resources.h"
#include "ninfer/targets/qwen3_6/prepared_prompt.h"
#include "ninfer/targets/qwen3_6/vision_control.h"
#include <ninfer/targets/qwen3_vision/vision.h>
#include <ninfer/targets/qwen3_8_flash_next/package.h>
#include <ninfer/targets/qwen3_8_flash_next/runtime.h>
#include "runtime/engine/context_cost.h"
#include "options.h"

#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                                           \
    do {                                                                                           \
        cudaError_t err__ = (call);                                                                \
        if (err__ != cudaSuccess) {                                                                \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__));     \
        }                                                                                          \
    } while (0)
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <bit>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::targets::qwen3_8_flash_next;
using namespace ninfer::targets::qwen3_8_flash_next::detail;
using LoadedModel = StandaloneLoadedModel;

const char* dtype_to_string(DType dt) {
    switch (dt) {
        case DType::BF16: return "BF16";
        case DType::FP32: return "FP32";
        case DType::I32:  return "I32";
        case DType::I64:  return "I64";
        case DType::FP16: return "FP16";
        case DType::U8:   return "U8";
        default:          return "UNKNOWN";
    }
}

struct TensorDumpRecord {
    std::string name;
    std::string dtype;
    std::vector<std::int32_t> shape;
    std::string file;
    std::size_t bytes = 0;
};

struct PositionDumpRecord {
    std::int32_t position = 0;
    std::int32_t token_id = 0;
    std::array<std::int32_t, 3> mrope_position{0, 0, 0};
    std::vector<TensorDumpRecord> tensors;
};

class StateDumper {
public:
    explicit StateDumper(std::string dump_dir) : dump_dir_(std::move(dump_dir)) {
        std::filesystem::create_directories(dump_dir_);
    }

    FlashNextDecodeStateSink make_sink(std::int32_t position, std::int32_t token_id,
                                       std::array<std::int32_t, 3> mrope_pos) {
        char pos_buf[32];
        std::snprintf(pos_buf, sizeof(pos_buf), "pos%04d", position);
        const std::string pos_str = pos_buf;
        const std::filesystem::path pos_dir = std::filesystem::path(dump_dir_) / pos_str;
        std::filesystem::create_directories(pos_dir);

        const std::size_t pos_idx = positions_.size();
        auto& pos_rec = positions_.emplace_back();
        pos_rec.position = position;
        pos_rec.token_id = token_id;
        pos_rec.mrope_position = mrope_pos;

        return FlashNextDecodeStateSink{
            .on_state = [this, pos_idx, pos_str, pos_dir](std::string_view name,
                                                          const Tensor& tensor) {
                const std::string filename = std::string(name) + ".bin";
                const std::filesystem::path file_path = pos_dir / filename;

                std::vector<std::uint8_t> host_bytes(tensor.bytes());
                CUDA_CHECK(cudaMemcpy(host_bytes.data(), tensor.data, tensor.bytes(),
                                      cudaMemcpyDeviceToHost));

                std::ofstream ofs(file_path, std::ios::binary);
                if (!ofs) {
                    throw std::runtime_error("Failed to open file for writing: " +
                                             file_path.string());
                }
                ofs.write(reinterpret_cast<const char*>(host_bytes.data()), host_bytes.size());

                TensorDumpRecord rec;
                rec.name  = std::string(name);
                rec.dtype = dtype_to_string(tensor.dtype);
                if (tensor.ne[3] > 1) {
                    rec.shape = {tensor.ne[0], tensor.ne[1], tensor.ne[2], tensor.ne[3]};
                } else if (tensor.ne[2] > 1) {
                    rec.shape = {tensor.ne[0], tensor.ne[1], tensor.ne[2]};
                } else if (tensor.ne[1] > 1 || tensor.ne[0] > 1) {
                    rec.shape = {tensor.ne[0], tensor.ne[1]};
                } else {
                    rec.shape = {tensor.ne[0]};
                }
                rec.file  = pos_str + "/" + filename;
                rec.bytes = tensor.bytes();
                positions_[pos_idx].tensors.push_back(std::move(rec));
            },
        };
    }

    FlashNextDecodeStateSink
    make_chunk_sink(std::int32_t first_position, std::span<const std::int32_t> token_ids,
                    std::span<const std::array<std::int32_t, 3>> mrope_positions) {
        const std::size_t count     = token_ids.size();
        const std::size_t first_idx = positions_.size();
        for (std::size_t i = 0; i < count; ++i) {
            char pos_buf[32];
            std::snprintf(pos_buf, sizeof(pos_buf), "pos%04d",
                          first_position + static_cast<std::int32_t>(i));
            const std::string pos_str           = pos_buf;
            const std::filesystem::path pos_dir = std::filesystem::path(dump_dir_) / pos_str;
            std::filesystem::create_directories(pos_dir);

            auto& pos_rec          = positions_.emplace_back();
            pos_rec.position       = first_position + static_cast<std::int32_t>(i);
            pos_rec.token_id       = token_ids[i];
            pos_rec.mrope_position = mrope_positions[i];
        }

        return FlashNextDecodeStateSink{
            .on_state = [this, first_idx, count](std::string_view name, const Tensor& tensor) {
                std::vector<std::uint8_t> host_bytes(tensor.bytes());
                CUDA_CHECK(cudaMemcpy(host_bytes.data(), tensor.data, tensor.bytes(),
                                      cudaMemcpyDeviceToHost));

                const std::string filename = std::string(name) + ".bin";
                if (tensor.ne[1] == static_cast<std::int32_t>(count)) {
                    const std::size_t col_bytes = tensor.bytes() / count;
                    for (std::size_t i = 0; i < count; ++i) {
                        auto& pos_rec = positions_[first_idx + i];
                        char pos_buf[32];
                        std::snprintf(pos_buf, sizeof(pos_buf), "pos%04d", pos_rec.position);
                        const std::string pos_str = pos_buf;
                        const std::filesystem::path pos_dir =
                            std::filesystem::path(dump_dir_) / pos_str;
                        const std::filesystem::path file_path = pos_dir / filename;

                        std::ofstream ofs(file_path, std::ios::binary);
                        if (!ofs) {
                            throw std::runtime_error("Failed to open file for writing: " +
                                                     file_path.string());
                        }
                        ofs.write(
                            reinterpret_cast<const char*>(host_bytes.data() + i * col_bytes),
                            col_bytes);

                        TensorDumpRecord rec;
                        rec.name  = std::string(name);
                        rec.dtype = dtype_to_string(tensor.dtype);
                        rec.shape = {tensor.ne[0], 1};
                        rec.file  = pos_str + "/" + filename;
                        rec.bytes = col_bytes;
                        pos_rec.tensors.push_back(std::move(rec));
                    }
                } else {
                    // Single-token tensor (e.g. final_hidden, logits on last token)
                    const std::size_t last_idx = first_idx + count - 1;
                    auto& pos_rec              = positions_[last_idx];
                    char pos_buf[32];
                    std::snprintf(pos_buf, sizeof(pos_buf), "pos%04d", pos_rec.position);
                    const std::string pos_str = pos_buf;
                    const std::filesystem::path pos_dir =
                        std::filesystem::path(dump_dir_) / pos_str;
                    const std::filesystem::path file_path = pos_dir / filename;

                    std::ofstream ofs(file_path, std::ios::binary);
                    if (!ofs) {
                        throw std::runtime_error("Failed to open file for writing: " +
                                                 file_path.string());
                    }
                    ofs.write(reinterpret_cast<const char*>(host_bytes.data()), host_bytes.size());

                    TensorDumpRecord rec;
                    rec.name  = std::string(name);
                    rec.dtype = dtype_to_string(tensor.dtype);
                    rec.shape = {tensor.ne[0], 1};
                    rec.file  = pos_str + "/" + filename;
                    rec.bytes = tensor.bytes();
                    pos_rec.tensors.push_back(std::move(rec));
                }
            },
        };
    }

    void write_manifest() const {
        const std::filesystem::path manifest_path =
            std::filesystem::path(dump_dir_) / "manifest.json";
        std::ofstream ofs(manifest_path);
        if (!ofs) {
            throw std::runtime_error("Failed to open manifest for writing: " +
                                     manifest_path.string());
        }
        ofs << "{\n  \"positions\": [\n";
        for (std::size_t p = 0; p < positions_.size(); ++p) {
            const auto& pos = positions_[p];
            ofs << "    {\n";
            ofs << "      \"position\": " << pos.position << ",\n";
            ofs << "      \"token_id\": " << pos.token_id << ",\n";
            ofs << "      \"mrope_position\": [" << pos.mrope_position[0] << ", "
                << pos.mrope_position[1] << ", " << pos.mrope_position[2] << "],\n";
            ofs << "      \"tensors\": [\n";
            for (std::size_t t = 0; t < pos.tensors.size(); ++t) {
                const auto& item = pos.tensors[t];
                ofs << "        {\n";
                ofs << "          \"name\": \"" << item.name << "\",\n";
                ofs << "          \"dtype\": \"" << item.dtype << "\",\n";
                ofs << "          \"shape\": [";
                for (std::size_t s = 0; s < item.shape.size(); ++s) {
                    ofs << item.shape[s] << (s + 1 < item.shape.size() ? ", " : "");
                }
                ofs << "],\n";
                ofs << "          \"file\": \"" << item.file << "\",\n";
                ofs << "          \"bytes\": " << item.bytes << "\n";
                ofs << "        }" << (t + 1 < pos.tensors.size() ? "," : "") << "\n";
            }
            ofs << "      ]\n";
            ofs << "    }" << (p + 1 < positions_.size() ? "," : "") << "\n";
        }
        ofs << "  ]\n}\n";
    }

private:
    std::string dump_dir_;
    std::vector<PositionDumpRecord> positions_;
};

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

std::vector<std::uint8_t> make_test_bmp(int width, int height, int pattern) {
    const int row_stride           = ((width * 3 + 3) / 4) * 4;
    const std::uint32_t image_size = static_cast<std::uint32_t>(row_stride * height);
    const std::uint32_t file_size  = 54 + image_size;
    std::vector<std::uint8_t> bmp(file_size, 0);

    // Bitmap File Header
    bmp[0]                                     = 'B';
    bmp[1]                                     = 'M';
    *reinterpret_cast<std::uint32_t*>(&bmp[2])  = file_size;
    *reinterpret_cast<std::uint32_t*>(&bmp[10]) = 54; // pixel offset

    // DIB Header (BITMAPINFOHEADER)
    *reinterpret_cast<std::uint32_t*>(&bmp[14]) = 40; // header size
    *reinterpret_cast<std::int32_t*>(&bmp[18])  = width;
    *reinterpret_cast<std::int32_t*>(&bmp[22])  = height;
    *reinterpret_cast<std::uint16_t*>(&bmp[26]) = 1;  // color planes
    *reinterpret_cast<std::uint16_t*>(&bmp[28]) = 24; // bits per pixel
    *reinterpret_cast<std::uint32_t*>(&bmp[30]) = 0;  // BI_RGB (uncompressed)
    *reinterpret_cast<std::uint32_t*>(&bmp[34]) = image_size;
    *reinterpret_cast<std::int32_t*>(&bmp[38])  = 2835; // 72 DPI
    *reinterpret_cast<std::int32_t*>(&bmp[42])  = 2835;

    // Fill pixels (BGR order)
    for (int y = 0; y < height; ++y) {
        auto* row = &bmp[54 + y * row_stride];
        for (int x = 0; x < width; ++x) {
            if (pattern == 1) {
                row[x * 3 + 0] = static_cast<std::uint8_t>((x * 255) / width);
                row[x * 3 + 1] = static_cast<std::uint8_t>((y * 255) / height);
                row[x * 3 + 2] = static_cast<std::uint8_t>(((x + y) * 255) / (width + height));
            } else {
                const bool check = ((x / 32) + (y / 32)) % 2 == 0;
                row[x * 3 + 0]   = check ? 240 : 15;
                row[x * 3 + 1]   = check ? 30 : 220;
                row[x * 3 + 2]   = check ? 180 : 40;
            }
        }
    }
    return bmp;
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
            << "Planned Mapped Tensors:       " << report.planned_mapped_tensors_count << "\n"
            << "Max Concurrency:              " << report.runtime_plan.config.max_concurrency << "\n"
            << "Max Context Tokens:           " << report.runtime_plan.config.max_context << "\n"
            << "Page Groups:                  " << report.runtime_plan.main_page_groups << "\n"
            << "State Slots:                  " << report.runtime_plan.state_slots << "\n"
            << "Total Device Allocation:      "
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
        throw std::runtime_error("No CUDA device available for execution");
    }

    DeviceContext device(0);

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

    std::unique_ptr<StateDumper> dumper;
    std::optional<FlashNextDecodeStateSink> sink;
    if (!opts.dump_states.empty()) {
        dumper = std::make_unique<StateDumper>(opts.dump_states);
        sink   = dumper->make_sink(0, opts.token_id, {0, 0, 0});
    }

    std::array<LaneStepRequest, 1> reqs{req};
    if (!opts.json_output) {
        std::cout << "Executing token round for token_id=" << opts.token_id
                  << " at position=0 ...\n";
    }
    auto round = executor.execute_round(reqs, sink ? &*sink : nullptr);
    device.synchronize();

    if (dumper) {
        dumper->write_manifest();
        if (!opts.json_output) {
            std::cout << "Dumped state tensors to " << opts.dump_states << "\n";
        }
    }

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
        throw std::runtime_error("No CUDA device available for full materialization probe");
    }

    std::size_t free_before  = 0;
    std::size_t total_before = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_before));

    DeviceContext device(0);

    constexpr std::string_view kExpectedModelId   = "qwen3.8-flash-next";
    constexpr std::string_view kExpectedWeightsId = "groupwise-nvfp4-dense-bf16-router-bf16";
    constexpr std::string_view kExpectedIdentity  = "qwen3.8-flash-next/groupwise-nvfp4";

    if (!opts.json_output) {
        std::cout << "Loading and fully materializing (Text + Vision + MTP): " << opts.model_path
                  << " ...\n";
    }

    auto model = LoadedModel::load_from_file(opts.model_path, device,
                                             LoadFeatures{.vision = true, .mtp = true});
    device.synchronize();

    if (!model.has_vision()) {
        throw std::logic_error(
            "LoadedModel reports has_vision() is false after requesting full load");
    }

    const auto& text = model.text_view();
    if (text.weights_arena == nullptr) {
        throw std::logic_error("Text weights_arena pointer is null");
    }

    std::size_t free_resident  = 0;
    std::size_t total_resident = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_resident, &total_resident));

    constexpr std::uint64_t kExpectedDeviceWeightsBytes = 81'272'758'784ULL;
    constexpr std::uint64_t kExpectedDeviceArenaBytes   = 81'272'774'656ULL;
    constexpr std::size_t kExpectedDeviceTensors        = 1435;
    constexpr std::size_t kExpectedRetainedResources    = 6;

    const auto& stats              = model.stats();
    const auto device_tensor_count = stats.tensor_count - stats.mapped_tensor_count;
    const auto& fe                 = model.frontend_resources();
    if (stats.h2d_bytes != kExpectedDeviceWeightsBytes ||
        stats.device_capacity_bytes != kExpectedDeviceArenaBytes ||
        text.weights_arena->capacity() != kExpectedDeviceArenaBytes ||
        device_tensor_count != kExpectedDeviceTensors ||
        stats.resource_count != kExpectedRetainedResources ||
        fe.tokenizer_json.empty() || fe.tokenizer_config_json.empty() ||
        fe.chat_template_jinja.empty() || fe.generation_config_json.empty() ||
        fe.preprocessor_config_json.empty() || fe.video_preprocessor_config_json.empty()) {
        throw std::logic_error("Materialization statistics violate the exact plan");
    }

    if (opts.json_output) {
        std::cout << "{\n"
                  << "  \"mode\": \"materialize-full\",\n"
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
        std::cout << "\n=== Full GPU Residency Materialization Report ===\n"
                  << "Identity:                     " << kExpectedIdentity << "\n"
                  << "Device Tensor Payloads:       "
                  << (kExpectedDeviceWeightsBytes / (1024.0 * 1024.0 * 1024.0)) << " GiB ("
                  << kExpectedDeviceWeightsBytes << " bytes)\n"
                  << "Device Weights Arena:         "
                  << (kExpectedDeviceArenaBytes / (1024.0 * 1024.0 * 1024.0)) << " GiB ("
                  << kExpectedDeviceArenaBytes << " bytes)\n"
                  << "Device Tensors:               " << kExpectedDeviceTensors
                  << " (1067 text + 333 vision + 35 mtp)\n"
                  << "Retained Resources:           " << kExpectedRetainedResources << "\n"
                  << "CUDA Free (Before Load):      " << (free_before / (1024.0 * 1024.0))
                  << " MiB / " << (total_before / (1024.0 * 1024.0)) << " MiB\n"
                  << "CUDA Free (Fully Resident):   " << (free_resident / (1024.0 * 1024.0))
                  << " MiB / " << (total_resident / (1024.0 * 1024.0)) << " MiB\n"
                  << "Materialization Status:       OK\n";
    }
    return 0;
}

int run_materialize_vision(const ReferenceToolOptions& opts) {
    int device_count = 0;
    const auto err   = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        throw std::runtime_error("No CUDA device available for vision materialization probe");
    }

    std::size_t free_before  = 0;
    std::size_t total_before = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_before));

    DeviceContext device(0);

    constexpr std::string_view kExpectedModelId   = "qwen3.8-flash-next";
    constexpr std::string_view kExpectedWeightsId = "groupwise-nvfp4-dense-bf16-router-bf16";
    constexpr std::string_view kExpectedIdentity  = "qwen3.8-flash-next/groupwise-nvfp4";

    if (!opts.json_output) {
        std::cout << "Loading and materializing (Text + Vision): " << opts.model_path << " ...\n";
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
    const auto& fe                 = model.frontend_resources();
    if (stats.h2d_bytes != kExpectedDeviceWeightsBytes ||
        stats.device_capacity_bytes != kExpectedDeviceArenaBytes ||
        text.weights_arena->capacity() != kExpectedDeviceArenaBytes ||
        device_tensor_count != kExpectedDeviceTensors ||
        stats.resource_count != kExpectedRetainedResources ||
        fe.tokenizer_json.empty() || fe.tokenizer_config_json.empty() ||
        fe.chat_template_jinja.empty() || fe.generation_config_json.empty() ||
        fe.preprocessor_config_json.empty() || fe.video_preprocessor_config_json.empty()) {
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

int run_chat_diagnostic(const ReferenceToolOptions& opts) {
    ninfer::DeviceContext device(0);

    // 1. Load model with Text features
    auto loaded = LoadedModel::load_from_file(
        opts.model_path, device, LoadFeatures{.vision = false, .mtp = false});

    // 2. Build Frontend
    auto frontend = make_frontend(
        loaded,
        ninfer::targets::qwen3_6::FrontendOptions{
            .vision_enabled = false,
            .max_context    = opts.max_context,
        });

    // 3. Prepare prompt
    std::vector<ninfer::ChatMessage> messages;
    if (!opts.system_prompt.empty()) {
        ninfer::ChatMessage sys_msg;
        sys_msg.role = ninfer::ChatRole::System;
        sys_msg.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text,
            .text = opts.system_prompt,
        });
        messages.push_back(std::move(sys_msg));
    }
    const std::string prompt_text = opts.prompt.empty()
                                        ? "Hello! Tell me who you are in one concise sentence."
                                        : opts.prompt;
    ninfer::ChatMessage user_msg;
    user_msg.role = ninfer::ChatRole::User;
    user_msg.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text,
        .text = prompt_text,
    });
    messages.push_back(std::move(user_msg));

    ninfer::PromptInput input{
        .messages = std::move(messages),
    };
    if (opts.reasoning_effort == "low") {
        input.options.reasoning_effort = ninfer::ReasoningEffort::Low;
    } else if (opts.reasoning_effort == "high" || opts.reasoning_effort == "xhigh") {
        input.options.reasoning_effort = ninfer::ReasoningEffort::XHigh;
    } else if (opts.reasoning_effort == "none") {
        input.options.enable_thinking = false;
    } else {
        input.options.reasoning_effort = ninfer::ReasoningEffort::Medium;
    }

    auto prepared = frontend.prepare(std::move(input));
    const auto& prepared_data =
        ninfer::targets::qwen3_6::PreparedPromptAccess::view(prepared);
    const auto& prompt_tokens = prepared_data.token_ids;
    if (prompt_tokens.empty()) {
        throw std::runtime_error("Prepared prompt contains zero tokens");
    }
    const auto pos0 = prepared_data.position_axis(0);
    const auto pos1 = prepared_data.position_axis(1);
    const auto pos2 = prepared_data.position_axis(2);

    // 4. Runtime & Text Executor
    FlashNextRuntimeConfig config{
        .max_concurrency     = 1,
        .max_context         = opts.max_context,
        .state_slot_capacity = opts.state_slots,
        .prefill_chunk       = opts.prefill_chunk > 0 ? opts.prefill_chunk : 1024,
        .use_cuda_graph      = opts.use_cuda_graph,
    };
    const auto curve = flash_next_capacity_curve(config);
    const std::uint32_t resolved_groups =
        opts.page_groups == 0 ? curve.maximum_main_page_groups : opts.page_groups;
    const auto plan = finalize_flash_next_runtime_plan(config, resolved_groups);
    FlashNextRuntimeAllocation alloc(plan);
    alloc.initialize(device.stream);
    FlashNextTextExecutor executor(loaded.text_view(), kPleIndexMetadata, device, alloc);
    auto lane = executor.allocate_lane();

    // 5. Sampling Setup
    constexpr std::int32_t kSemanticTokenDomain = 248'077;
    const std::size_t ws_bytes =
        ninfer::ops::sampling_workspace_capacity_bytes(kSemanticTokenDomain, 1, 1);
    ninfer::WorkspaceArena workspace(std::max<std::size_t>(256, ws_bytes));

    ninfer::ops::SamplingConfig sampling_cfg{
        .temperature = opts.temperature,
        .top_k       = opts.top_k,
        .top_p       = opts.top_p,
        .seed        = opts.seed,
    };

    ninfer::DeviceBuffer device_configs(sizeof(ninfer::ops::SamplingConfig));
    device_configs.copy_from_host(&sampling_cfg, sizeof(ninfer::ops::SamplingConfig));

    ninfer::DeviceBuffer device_positions(sizeof(std::int32_t));
    ninfer::DeviceBuffer device_out(sizeof(std::int32_t));

    ninfer::Tensor out_tensor(device_out.p, ninfer::DType::I32, {1});
    ninfer::Tensor positions_tensor(device_positions.p, ninfer::DType::I32, {1});

    // 6. Output Session
    ninfer::StopPolicy stop_policy;
    stop_policy.token_ids = {248'046, 248'044};
    auto output_session   = frontend.make_output_session(
        prepared, stop_policy, ninfer::OutputOptions{});

    // 7. Prefill Prompt
    std::unique_ptr<StateDumper> dumper;
    if (!opts.dump_states.empty()) {
        dumper = std::make_unique<StateDumper>(opts.dump_states);
    }

    const std::size_t prompt_len = prompt_tokens.size();
    PendingRound round;

    auto run_prefill = [&]() {
    if (opts.prefill_chunk == 0) {
        for (std::size_t i = 0; i + 1 < prompt_len; ++i) {
            LaneStepRequest req{
                .handle          = lane,
                .token_id        = prompt_tokens[i],
                .token_index     = static_cast<std::int32_t>(i),
                .mrope_positions = {pos0[i], pos1[i], pos2[i]},
            };
            std::optional<FlashNextDecodeStateSink> sink;
            if (dumper) {
                sink = dumper->make_sink(static_cast<std::int32_t>(i), prompt_tokens[i],
                                         {pos0[i], pos1[i], pos2[i]});
            }
            auto r = executor.execute_round(std::span<const LaneStepRequest>(&req, 1),
                                            sink ? &*sink : nullptr);
            std::vector<LaneCommitDecision> decision = {{.accept = true}};
            r.commit(decision);
        }

        LaneStepRequest last_prompt_req{
            .handle          = lane,
            .token_id        = prompt_tokens[prompt_len - 1],
            .token_index     = static_cast<std::int32_t>(prompt_len - 1),
            .mrope_positions = {pos0[prompt_len - 1], pos1[prompt_len - 1], pos2[prompt_len - 1]},
        };
        std::optional<FlashNextDecodeStateSink> last_sink;
        if (dumper) {
            last_sink = dumper->make_sink(static_cast<std::int32_t>(prompt_len - 1),
                                          prompt_tokens[prompt_len - 1],
                                          {pos0[prompt_len - 1], pos1[prompt_len - 1],
                                           pos2[prompt_len - 1]});
        }
        round = executor.execute_round(std::span<const LaneStepRequest>(&last_prompt_req, 1),
                                       last_sink ? &*last_sink : nullptr);
    } else {
        const std::uint32_t chunk_cfg = opts.prefill_chunk;
        std::uint32_t start_i = 0;
        while (start_i < prompt_len) {
            const std::uint32_t current_chunk =
                std::min<std::uint32_t>(chunk_cfg, static_cast<std::uint32_t>(prompt_len - start_i));
            const std::uint32_t end_i = start_i + current_chunk;

            std::span<const std::int32_t> chunk_token_ids(prompt_tokens.data() + start_i, current_chunk);
            std::vector<std::array<std::int32_t, 3>> chunk_positions(current_chunk);
            for (std::uint32_t c = 0; c < current_chunk; ++c) {
                chunk_positions[c] = {pos0[start_i + c], pos1[start_i + c], pos2[start_i + c]};
            }

            std::optional<FlashNextDecodeStateSink> chunk_sink;
            if (dumper) {
                chunk_sink = dumper->make_chunk_sink(static_cast<std::int32_t>(start_i),
                                                     chunk_token_ids, chunk_positions);
            }

            if (end_i < prompt_len) {
                auto chunk_round = executor.execute_prefill_chunk(
                    lane, chunk_token_ids, chunk_positions, static_cast<std::int32_t>(start_i),
                    chunk_sink ? &*chunk_sink : nullptr);
                std::vector<LaneCommitDecision> decision = {{.accept = true}};
                chunk_round.commit(decision);
            } else {
                round = executor.execute_prefill_chunk(
                    lane, chunk_token_ids, chunk_positions, static_cast<std::int32_t>(start_i),
                    chunk_sink ? &*chunk_sink : nullptr);
            }
            start_i = end_i;
        }
    }

    };
    run_prefill();
    if (opts.repeat_prefill > 1) {
        auto grab = [&](std::vector<std::uint16_t>& out) {
            out.resize(248'320);
            CUDA_CHECK(cudaStreamSynchronize(device.stream));
            CUDA_CHECK(cudaMemcpy(out.data(), round.logits().data, out.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost));
        };
        auto bf = [](std::uint16_t v) { return std::bit_cast<float>(static_cast<std::uint32_t>(v) << 16U); };
        std::vector<std::uint16_t> first;
        grab(first);
        std::uint32_t differing = 0;
        float worst = 0.0F;
        for (std::uint32_t r = 1; r < opts.repeat_prefill; ++r) {
            std::vector<LaneCommitDecision> d = {{.accept = true}};
            round.commit(d);
            executor.release_lane(lane);
            lane = executor.allocate_lane();
            run_prefill();
            std::vector<std::uint16_t> cur;
            grab(cur);
            if (cur != first) {
                float m = 0.0F;
                for (std::size_t i = 0; i < cur.size(); ++i) { m = std::max(m, std::abs(bf(cur[i]) - bf(first[i]))); }
                worst = std::max(worst, m);
                ++differing;
                std::cout << "repeat-prefill " << r << ": DIFFERS from the first run, max|d|=" << m << "\n";
            }
        }
        std::cout << "repeat-prefill: " << differing << " of " << (opts.repeat_prefill - 1)
                  << " repetitions differ from the first (worst max|d| " << worst << ")\n";
    }

    if (dumper) {
        dumper->write_manifest();
        if (!opts.json_output) {
            std::cout << "Dumped prompt state tensors to " << opts.dump_states << "\n";
        }
    }

    // 8. Generation Loop
    std::uint32_t generated_tokens   = 0;
    std::int32_t current_pos         = pos0[prompt_len - 1];
    std::int32_t current_token_index = static_cast<std::int32_t>(prompt_len - 1);
    std::string accumulated_reasoning;
    std::string accumulated_content;

    if (!opts.json_output) {
        std::cout << "\n=== Qwen3.8-Flash-Next Chat Diagnostic ===\n"
                  << "Prompt tokens: " << prompt_len << "\n"
                  << "Temperature:   " << opts.temperature << " (greedy: "
                  << (opts.temperature <= 0.0f ? "true" : "false") << ")\n"
                  << "Top-k:         " << opts.top_k << "\n"
                  << "Top-p:         " << opts.top_p << "\n"
                  << "Seed:          " << opts.seed << "\n"
                  << "Max tokens:    " << opts.max_tokens << "\n"
                  << "Effort:        " << opts.reasoning_effort << "\n\n"
                  << "--- Output ---\n";
    }

    if (!opts.dump_gen_logits.empty()) { std::filesystem::create_directories(opts.dump_gen_logits); }
    while (true) {
        if (!opts.dump_gen_logits.empty()) {
            // Per-round logits without a state sink, so graph replay stays in effect.
            device.synchronize();
            std::vector<std::uint16_t> host_logits(248'320);
            CUDA_CHECK(cudaMemcpy(host_logits.data(), round.logits().data,
                                  host_logits.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost));
            char name[32];
            std::snprintf(name, sizeof(name), "gen_%03u.bin", generated_tokens);
            std::ofstream out(std::filesystem::path(opts.dump_gen_logits) / name, std::ios::binary);
            out.write(reinterpret_cast<const char*>(host_logits.data()),
                      static_cast<std::streamsize>(host_logits.size() * sizeof(std::uint16_t)));
        }
        device_positions.copy_from_host(&current_pos, sizeof(std::int32_t));
        ninfer::ops::sample(
            round.logits(), out_tensor, kSemanticTokenDomain,
            static_cast<const ninfer::ops::SamplingConfig*>(device_configs.p),
            positions_tensor, ninfer::ops::kSamplePurposeDecode, workspace, device.stream);
        std::int32_t sampled_token = 0;
        // copy_to_host is a legacy-default-stream cudaMemcpy, which does not order against the
        // cudaStreamNonBlocking compute stream that ran the sampler: synchronize first or the
        // copy can observe the previous round's token.
        device.synchronize();
        device_out.copy_to_host(&sampled_token, sizeof(std::int32_t));

        std::vector<LaneCommitDecision> decision = {{.accept = true}};
        round.commit(decision);

        ++generated_tokens;

        const ninfer::TokenId tok_id = static_cast<ninfer::TokenId>(sampled_token);
        const std::uint32_t remaining_budget =
            opts.max_tokens >= (generated_tokens - 1) ? opts.max_tokens - (generated_tokens - 1) : 0;
        const auto dec = output_session.preview_model(
            std::span<const ninfer::TokenId>(&tok_id, 1),
            remaining_budget,
            ninfer::FinishReason::OutputLimit);

        auto published = output_session.commit_preview();
        for (const auto& delta : published) {
            if (!opts.json_output) {
                if (delta.channel == ninfer::OutputChannel::Reasoning) {
                    std::cout << delta.text;
                    std::cout.flush();
                } else if (delta.channel == ninfer::OutputChannel::Content) {
                    std::cout << delta.text;
                    std::cout.flush();
                }
            }
            if (delta.channel == ninfer::OutputChannel::Reasoning) {
                accumulated_reasoning += delta.text;
            } else if (delta.channel == ninfer::OutputChannel::Content) {
                accumulated_content += delta.text;
            }
        }

        if (dec.finished() || generated_tokens >= opts.max_tokens) { break; }

        ++current_pos;
        ++current_token_index;
        LaneStepRequest next_req{
            .handle          = lane,
            .token_id        = sampled_token,
            .token_index     = current_token_index,
            .mrope_positions = {current_pos, current_pos, current_pos},
        };
        round = executor.execute_round(std::span<const LaneStepRequest>(&next_req, 1));
    }

    executor.release_lane(lane);

    if (opts.json_output) {
        auto json_escape = [](std::string_view s) {
            std::string out;
            for (char c : s) {
                if (c == '"')
                    out += "\\\"";
                else if (c == '\\')
                    out += "\\\\";
                else if (c == '\n')
                    out += "\\n";
                else if (c == '\r')
                    out += "\\r";
                else if (c == '\t')
                    out += "\\t";
                else
                    out += c;
            }
            return out;
        };
        std::cout << "{\n"
                  << "  \"mode\": \"chat-diagnostic\",\n"
                  << "  \"prompt_tokens\": " << prompt_len << ",\n"
                  << "  \"generated_tokens\": " << generated_tokens << ",\n"
                  << "  \"reasoning_content\": \"" << json_escape(accumulated_reasoning) << "\",\n"
                  << "  \"content\": \"" << json_escape(accumulated_content) << "\",\n"
                  << "  \"status\": \"OK\"\n"
                  << "}\n";
    } else {
        std::cout << "\n\n--- Done (" << generated_tokens << " generated tokens) ---\n";
    }
    return 0;
}

std::vector<std::uint8_t> make_flash_next_test_bmp(int width, int height, int pattern) {
    const int row_padded = (width * 3 + 3) & (~3);
    const std::uint32_t image_size = static_cast<std::uint32_t>(row_padded * height);
    const std::uint32_t file_size = 54U + image_size;

    std::vector<std::uint8_t> bmp(file_size, 0);
    bmp[0] = 'B'; bmp[1] = 'M';
    *reinterpret_cast<std::uint32_t*>(&bmp[2]) = file_size;
    *reinterpret_cast<std::uint32_t*>(&bmp[10]) = 54U;
    *reinterpret_cast<std::uint32_t*>(&bmp[14]) = 40U;
    *reinterpret_cast<std::int32_t*>(&bmp[18]) = width;
    *reinterpret_cast<std::int32_t*>(&bmp[22]) = height;
    *reinterpret_cast<std::uint16_t*>(&bmp[26]) = 1;
    *reinterpret_cast<std::uint16_t*>(&bmp[28]) = 24;
    *reinterpret_cast<std::uint32_t*>(&bmp[34]) = image_size;

    std::uint8_t* pixels = &bmp[54];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::uint8_t r = 0, g = 0, b = 0;
            if (pattern == 0) {
                r = static_cast<std::uint8_t>((x * 255) / width);
                g = static_cast<std::uint8_t>((y * 255) / height);
                b = static_cast<std::uint8_t>(((x + y) * 128) / (width + height));
            } else {
                r = ((x / 16) % 2 == 0) ? 255 : 0;
                g = ((y / 16) % 2 == 0) ? 255 : 0;
                b = ((x / 16 + y / 16) % 2 == 0) ? 255 : 0;
            }
            const int idx = y * row_padded + x * 3;
            pixels[idx + 0] = b;
            pixels[idx + 1] = g;
            pixels[idx + 2] = r;
        }
    }
    return bmp;
}

int run_execute_vision(const ReferenceToolOptions& opts) {
    int device_count = 0;
    const auto err   = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        throw std::runtime_error("No CUDA device available for vision execution");
    }

    DeviceContext device(0);
    if (!opts.json_output) {
        std::cout << "Loading and materializing text + vision weights: " << opts.model_path << " ...\n";
    }

    auto model = LoadedModel::load_from_file(opts.model_path, device,
                                             LoadFeatures{.vision = true, .mtp = false});
    device.synchronize();

    if (!model.has_vision()) {
        throw std::logic_error("LoadedModel has_vision() is false");
    }

    // 1. Construct Frontend
    auto frontend = make_frontend(
        model, ninfer::targets::qwen3_6::FrontendOptions{.vision_enabled = true});

    // 2. Initialize Shared Vision Encoder
    const auto vision_weights = adapt_vision_weights(model.vision_view());
    ninfer::targets::qwen3_vision::Encoder vision_encoder(device, vision_weights);

    // 3. Encode Image 1 and Image 2
    struct ImageRunResult {
        std::uint64_t vision_checksum = 0;
        std::size_t patch_count       = 0;
        std::size_t merged_tokens     = 0;
        std::int32_t prompt_tokens    = 0;
        std::int32_t argmax_token     = 0;
        float argmax_logit            = 0.0f;
        double logits_sum             = 0.0;
        std::uint64_t logits_checksum = 0;
    };

    const auto process_image_prompt = [&](int pattern) -> ImageRunResult {
        ImageRunResult res{};
        std::vector<std::uint8_t> bmp_bytes = make_flash_next_test_bmp(256, 256, pattern);

        ninfer::PromptInput prompt_input;
        ninfer::ChatMessage msg;
        msg.role = ninfer::ChatRole::User;

        ninfer::MessagePart media_part;
        media_part.kind = ninfer::MessagePartKind::Media;
        media_part.media.kind = ninfer::MediaKind::Image;
        media_part.media.bytes = std::move(bmp_bytes);
        media_part.media.media_type = "image/bmp";
        media_part.media.source_name = "test_image.bmp";

        ninfer::MessagePart text_part;
        text_part.kind = ninfer::MessagePartKind::Text;
        text_part.text = "Describe this image in detail.";

        msg.parts.push_back(std::move(media_part));
        msg.parts.push_back(std::move(text_part));
        prompt_input.messages.push_back(std::move(msg));

        auto prepared_prompt = frontend.prepare(prompt_input);
        auto data = ninfer::targets::qwen3_6::PreparedPromptAccess::take(std::move(prepared_prompt));

        if (data.vision_items.empty() || data.media_payloads.empty()) {
            throw std::logic_error("Prepared prompt contains no vision items");
        }

        const auto control_plan = ninfer::targets::qwen3_6::plan_vision_control(data);
        const auto control      = ninfer::targets::qwen3_6::build_vision_control(data, control_plan, 0);

        if (control.items.empty()) {
            throw std::logic_error("Vision control plan produced 0 items");
        }

        const auto& item_ctrl = control.items[0];
        const auto& payload   = *data.media_payloads[0];

        res.patch_count   = item_ctrl.patch_count;
        res.merged_tokens = item_ctrl.merged_count;
        res.prompt_tokens = static_cast<std::int32_t>(data.token_ids.size());

        // Plan workspace and output binding
        const auto plan = ninfer::targets::qwen3_vision::Encoder::plan_workspace(
            static_cast<std::uint32_t>(res.merged_tokens), 64 * 1024 * 1024, 2560);
        DeviceBuffer workspace_buf(plan.capacity_bytes);
        DeviceSpan backing{workspace_buf.p, workspace_buf.bytes};
        Tensor vision_out = ninfer::targets::qwen3_vision::Encoder::bind_output(
            backing, plan, static_cast<std::uint32_t>(res.merged_tokens), 2560);

        ninfer::targets::qwen3_vision::EncodeItemView item_view{
            .patches                = payload.span(),
            .patch_count            = item_ctrl.patch_count,
            .merged_count           = item_ctrl.merged_count,
            .segment_length         = item_ctrl.segment_length,
            .position_ids           = item_ctrl.position_ids,
            .position_table_indices = item_ctrl.position_table_indices,
            .position_table_weights = item_ctrl.position_table_weights,
        };

        vision_encoder.encode(item_view, vision_out, backing, plan);
        device.synchronize();

        std::vector<std::uint16_t> host_vision_bf16(2560 * res.merged_tokens);
        CUDA_CHECK(cudaMemcpy(host_vision_bf16.data(), vision_out.data,
                              host_vision_bf16.size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost));

        res.vision_checksum = fnv1a_64(host_vision_bf16.data(),
                                       host_vision_bf16.size() * sizeof(std::uint16_t));

        // Now run sequential prefill rounds through FlashNextTextExecutor
        FlashNextRuntimeConfig config{
            .max_concurrency     = 1,
            .max_context         = ((static_cast<std::uint32_t>(res.prompt_tokens + 64) + 127U) / 128U) * 128U,
            .state_slot_capacity = 2,
            .prefill_chunk       = 128, // must be a multiple of 128 and <= max_context (5c-1)
        };
        const auto curve = flash_next_capacity_curve(config);
        const std::uint32_t resolved_groups =
            opts.page_groups == 0 ? curve.minimum_main_page_groups : opts.page_groups;
        auto runtime_plan = finalize_flash_next_runtime_plan(config, resolved_groups);
        FlashNextRuntimeAllocation alloc(runtime_plan);
        alloc.initialize(device.stream);
        FlashNextTextExecutor executor(model.text_view(), model.ple_metadata(), device, alloc);
        auto lane = executor.allocate_lane();

        std::vector<std::int32_t> vision_column_for_token(data.token_ids.size(), -1);
        for (std::size_t v = 0; v < item_ctrl.scatter_indices.size(); ++v) {
            const auto prompt_idx = item_ctrl.scatter_indices[v];
            if (prompt_idx >= 0 && prompt_idx < static_cast<std::int32_t>(data.token_ids.size())) {
                vision_column_for_token[prompt_idx] = static_cast<std::int32_t>(v);
            }
        }

        std::vector<std::uint16_t> last_logits_bf16(248'320);

        for (std::size_t t = 0; t < data.token_ids.size(); ++t) {
            const auto v_col = vision_column_for_token[t];
            Tensor custom_emb{};
            if (v_col >= 0) {
                custom_emb = Tensor(
                    static_cast<std::uint16_t*>(vision_out.data) + static_cast<std::size_t>(v_col) * 2560,
                    DType::BF16,
                    {2'560, 1});
            }

            LaneStepRequest step_req{
                .handle           = lane,
                .token_id         = data.token_ids[t],
                .token_index      = static_cast<std::int32_t>(t),
                .mrope_positions  = {data.positions[t * 3], data.positions[t * 3 + 1], data.positions[t * 3 + 2]},
                .custom_embedding = v_col >= 0 ? &custom_emb : nullptr,
            };

            std::array<LaneStepRequest, 1> reqs{step_req};
            auto round = executor.execute_round(reqs);
            device.synchronize();

            if (t + 1 == data.token_ids.size()) {
                CUDA_CHECK(cudaMemcpy(last_logits_bf16.data(), round.logits().data,
                                      248'320 * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
            }

            std::vector<LaneCommitDecision> decisions = {{.accept = true}};
            round.commit(decisions);
        }

        executor.release_lane(lane);

        std::int32_t best_token = 0;
        float best_logit        = -1e30f;
        double logit_sum        = 0.0;

        for (std::size_t i = 0; i < 248'320; ++i) {
            const float val = bf16_to_float(last_logits_bf16[i]);
            logit_sum += val;
            if (val > best_logit) {
                best_logit = val;
                best_token = static_cast<std::int32_t>(i);
            }
        }

        res.argmax_token     = best_token;
        res.argmax_logit     = best_logit;
        res.logits_sum       = logit_sum;
        res.logits_checksum  = fnv1a_64(last_logits_bf16.data(), last_logits_bf16.size() * sizeof(std::uint16_t));
        return res;
    };

    if (!opts.json_output) {
        std::cout << "Encoding and executing Image 1 (gradient pattern) ...\n";
    }
    const auto res1 = process_image_prompt(0);

    if (!opts.json_output) {
        std::cout << "Encoding and executing Image 2 (contrast pattern) ...\n";
    }
    const auto res2 = process_image_prompt(1);

    const bool vision_checksums_differ = res1.vision_checksum != res2.vision_checksum;
    const bool logits_checksums_differ = res1.logits_checksum != res2.logits_checksum;

    if (!vision_checksums_differ) {
        throw std::logic_error("Vision encoder produced identical output for two different images");
    }
    if (!logits_checksums_differ) {
        throw std::logic_error("Multimodal post-prefill logits did not alter between two different images");
    }

    if (opts.json_output) {
        std::cout << "{\n"
                  << "  \"mode\": \"execute-vision\",\n"
                  << "  \"image1\": {\n"
                  << "    \"patches\": " << res1.patch_count << ",\n"
                  << "    \"merged_tokens\": " << res1.merged_tokens << ",\n"
                  << "    \"prompt_tokens\": " << res1.prompt_tokens << ",\n"
                  << "    \"vision_checksum\": \"" << std::hex << res1.vision_checksum << std::dec << "\",\n"
                  << "    \"argmax_token\": " << res1.argmax_token << ",\n"
                  << "    \"argmax_logit\": " << res1.argmax_logit << ",\n"
                  << "    \"logits_checksum\": \"" << std::hex << res1.logits_checksum << std::dec << "\"\n"
                  << "  },\n"
                  << "  \"image2\": {\n"
                  << "    \"patches\": " << res2.patch_count << ",\n"
                  << "    \"merged_tokens\": " << res2.merged_tokens << ",\n"
                  << "    \"prompt_tokens\": " << res2.prompt_tokens << ",\n"
                  << "    \"vision_checksum\": \"" << std::hex << res2.vision_checksum << std::dec << "\",\n"
                  << "    \"argmax_token\": " << res2.argmax_token << ",\n"
                  << "    \"argmax_logit\": " << res2.argmax_logit << ",\n"
                  << "    \"logits_checksum\": \"" << std::hex << res2.logits_checksum << std::dec << "\"\n"
                  << "  },\n"
                  << "  \"vision_outputs_differ\": true,\n"
                  << "  \"post_prefill_logits_altered\": true,\n"
                  << "  \"status\": \"OK\"\n"
                  << "}\n";
    } else {
        std::cout << "\n=== Multimodal Vision Execution Report ===\n"
                  << "Image 1 (Gradient):\n"
                  << "  Raw Patches:                " << res1.patch_count << "\n"
                  << "  Merged Vision Tokens:       " << res1.merged_tokens << "\n"
                  << "  Prompt Sequence Length:     " << res1.prompt_tokens << "\n"
                  << "  Vision Output FNV-1a:       0x" << std::hex << res1.vision_checksum << std::dec << "\n"
                  << "  Post-Prefill Argmax Token:  " << res1.argmax_token << "\n"
                  << "  Post-Prefill Argmax Logit:  " << std::fixed << std::setprecision(4) << res1.argmax_logit << "\n"
                  << "  Post-Prefill Logits FNV-1a: 0x" << std::hex << res1.logits_checksum << std::dec << "\n\n"
                  << "Image 2 (Contrast):\n"
                  << "  Raw Patches:                " << res2.patch_count << "\n"
                  << "  Merged Vision Tokens:       " << res2.merged_tokens << "\n"
                  << "  Prompt Sequence Length:     " << res2.prompt_tokens << "\n"
                  << "  Vision Output FNV-1a:       0x" << std::hex << res2.vision_checksum << std::dec << "\n"
                  << "  Post-Prefill Argmax Token:  " << res2.argmax_token << "\n"
                  << "  Post-Prefill Argmax Logit:  " << std::fixed << std::setprecision(4) << res2.argmax_logit << "\n"
                  << "  Post-Prefill Logits FNV-1a: 0x" << std::hex << res2.logits_checksum << std::dec << "\n\n"
                  << "Vision Outputs Differ:        YES\n"
                  << "Post-Prefill Logits Altered:  YES\n"
                  << "Status:                       OK\n";
    }

    return 0;
}

int run_continuation_check(const ReferenceToolOptions& opts) {
    ninfer::DeviceContext device(0);

    // 1. Load model with Text features
    auto loaded = LoadedModel::load_from_file(
        opts.model_path, device, LoadFeatures{.vision = false, .mtp = false});

    // 2. Build Frontend
    auto frontend = make_frontend(
        loaded,
        ninfer::targets::qwen3_6::FrontendOptions{
            .vision_enabled = false,
            .max_context    = opts.max_context,
        });

    // 3. Setup Runtime Plan & Program
    FlashNextRuntimeConfig config{
        .max_concurrency       = 1,
        .max_context           = opts.max_context,
        // Lane slots plus catalog slots. 2*mc + capacity was not enough on the real artifact
        // (copy_state_slot: slot index exceeds state_slots) while 40 works: the Program's cache
        // slot numbering needs auditing (sequence 6f); keep a generous default meanwhile.
        .state_slot_capacity   = opts.state_slots > 0 ? opts.state_slots : 40u,
        .continuation_capacity = 16,
        .prefill_chunk         = opts.prefill_chunk > 0 ? opts.prefill_chunk : 1024,
        .use_cuda_graph        = opts.use_cuda_graph,
    };
    const auto curve = flash_next_capacity_curve(config);
    const std::uint32_t resolved_groups =
        opts.page_groups == 0 ? curve.maximum_main_page_groups : opts.page_groups;
    auto plan = finalize_flash_next_runtime_plan(config, resolved_groups);
    auto program_impl = std::make_unique<ProgramImpl>(
        nullptr, plan, device, loaded.text_view(), std::nullopt, loaded.ple_metadata());
    Program program(std::move(program_impl));

    std::atomic<bool> flag{false};
    ninfer::runtime::CancellationFlagView cancellation{&flag};
    std::array<ninfer::runtime::CommitDecision, 1> commit_dec = {{{.accepted_tokens = 1, .terminal = false}}};

    // 4. Prepare Turn 1 Prompt
    std::vector<ninfer::ChatMessage> messages_t1;
    if (!opts.system_prompt.empty()) {
        ninfer::ChatMessage sys_msg;
        sys_msg.role = ninfer::ChatRole::System;
        sys_msg.parts.push_back(ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = opts.system_prompt});
        messages_t1.push_back(std::move(sys_msg));
    }
    const std::string prompt_text_t1 = opts.prompt.empty()
                                        ? "Hello! Tell me who you are in one concise sentence."
                                        : opts.prompt;
    ninfer::ChatMessage user_msg_t1;
    user_msg_t1.role = ninfer::ChatRole::User;
    user_msg_t1.parts.push_back(ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = prompt_text_t1});
    messages_t1.push_back(std::move(user_msg_t1));

    ninfer::PromptInput input_t1{.messages = messages_t1};
    if (opts.reasoning_effort == "low") {
        input_t1.options.reasoning_effort = ninfer::ReasoningEffort::Low;
    } else if (opts.reasoning_effort == "high" || opts.reasoning_effort == "xhigh") {
        input_t1.options.reasoning_effort = ninfer::ReasoningEffort::XHigh;
    } else if (opts.reasoning_effort == "none") {
        input_t1.options.enable_thinking = false;
    } else {
        input_t1.options.reasoning_effort = ninfer::ReasoningEffort::Medium;
    }

    auto prepared_t1 = frontend.prepare(PromptInput(input_t1));
    const std::size_t t1_prompt_tokens = ninfer::targets::qwen3_6::PreparedPromptAccess::view(prepared_t1).token_ids.size();

    // 5. Execute Turn 1 through Program (capturing at F, finish)
    ninfer::runtime::ResolvedExecutionOptions exec_options{};
    exec_options.requested_output_tokens = 1;
    auto base_t1 = program.plan_request(prepared_t1, exec_options);
    ninfer::runtime::ContextMachineCostModel cost_model{};
    auto cand_t1 = program.inspect_admission(prepared_t1, base_t1, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
    if (!cand_t1) { throw std::runtime_error("Turn 1 admission failed"); }
    auto res_t1 = program.seal_identity(*cand_t1, prepared_t1);
    (void)program.start_resource_transaction(std::move(*res_t1), std::move(prepared_t1), cancellation);
    auto prog_t1 = program.progress_context_transaction(cancellation);
    auto* mat_t1 = std::get_if<MaterializationResult>(&prog_t1);
    if (!mat_t1 || !mat_t1->published) { throw std::runtime_error("Turn 1 materialization failed"); }
    SequenceHandle seq_t1 = mat_t1->published->sequence;
    program.finalize_context_transaction();

    std::uint32_t t1_capture_frontier = 0;
    while (true) {
        auto pref_t1 = program.advance_prefill(seq_t1);
        if (pref_t1.capture.has_value()) {
            t1_capture_frontier = pref_t1.processed_prompt_tokens;
            auto cap_res = program.reserve_active_capture(std::move(*pref_t1.capture), nullptr, nullptr, std::nullopt, cancellation);
            if (cap_res == ninfer::runtime::ContextTransactionReserveStatus::Reserved) {
                (void)program.progress_context_transaction(cancellation);
                program.finalize_context_transaction();
            }
        }
        if (pref_t1.pending.has_value()) {
            (void)program.commit(std::move(*pref_t1.pending), commit_dec);
        }
        if (pref_t1.complete) {
            break;
        }
    }
    FinishResult fin_t1 = program.finish(seq_t1);
    if (!fin_t1.continuation.has_value()) {
        throw std::runtime_error("Turn 1 finish did not yield continuation handle");
    }
    ContinuationHandle cont_t1 = std::move(*fin_t1.continuation);

    // 6. Build Turn 2 Prompt
    const std::string turn2_text = opts.continuation_check.empty()
                                       ? "Tell me more details about that."
                                       : opts.continuation_check;
    std::vector<ninfer::ChatMessage> messages_t2;
    if (!opts.system_prompt.empty()) {
        messages_t2.push_back(ninfer::ChatMessage{
            .role = ninfer::ChatRole::System,
            .parts = {ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = opts.system_prompt}},
        });
    }
    messages_t2.push_back(ninfer::ChatMessage{
        .role = ninfer::ChatRole::User,
        .parts = {ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = prompt_text_t1}},
    });
    messages_t2.push_back(ninfer::ChatMessage{
        .role = ninfer::ChatRole::Assistant,
        .parts = {ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "I am Qwen, an AI assistant developed by Alibaba Cloud."}},
    });
    messages_t2.push_back(ninfer::ChatMessage{
        .role = ninfer::ChatRole::User,
        .parts = {ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = turn2_text}},
    });

    ninfer::PromptInput input_t2{.messages = messages_t2};
    if (opts.reasoning_effort == "low") {
        input_t2.options.reasoning_effort = ninfer::ReasoningEffort::Low;
    } else if (opts.reasoning_effort == "high" || opts.reasoning_effort == "xhigh") {
        input_t2.options.reasoning_effort = ninfer::ReasoningEffort::XHigh;
    } else if (opts.reasoning_effort == "none") {
        input_t2.options.enable_thinking = false;
    } else {
        input_t2.options.reasoning_effort = ninfer::ReasoningEffort::Medium;
    }

    auto prepared_t2_resumed = frontend.prepare(PromptInput(input_t2));
    const std::size_t t2_prompt_tokens = ninfer::targets::qwen3_6::PreparedPromptAccess::view(prepared_t2_resumed).token_ids.size();

    // 7. Execute Turn 2 Resumed
    auto base_t2_res = program.plan_request(prepared_t2_resumed, exec_options);
    auto cand_t2_res = program.inspect_admission(prepared_t2_resumed, base_t2_res, ninfer::runtime::LaneId(0), &cont_t1, nullptr, std::nullopt, false, cost_model);
    if (!cand_t2_res) { throw std::runtime_error("Turn 2 resumed admission failed"); }
    const std::uint32_t reused_tokens = cand_t2_res->summary().reusable_prompt_tokens;
    auto res_t2 = program.seal_identity(*cand_t2_res, prepared_t2_resumed);
    (void)program.start_resource_transaction(std::move(*res_t2), std::move(prepared_t2_resumed), cancellation);
    auto prog_t2 = program.progress_context_transaction(cancellation);
    SequenceHandle seq_t2 = std::get_if<MaterializationResult>(&prog_t2)->published->sequence;
    program.finalize_context_transaction();

    std::vector<float> logits_resumed;
    std::vector<std::uint16_t> logits_resumed_raw;
    while (true) {
        auto pref_t2 = program.advance_prefill(seq_t2);
        if (pref_t2.capture.has_value()) {
            program.skip_capture(std::move(*pref_t2.capture));
        }
        if (pref_t2.pending.has_value()) {
            const auto& l_tensor = program.impl_->allocation_.round_tensors().logits;
            logits_resumed_raw.resize(l_tensor.numel());
            CHECK_CUDA(cudaMemcpy(logits_resumed_raw.data(), l_tensor.data,
                                  logits_resumed_raw.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost));
            logits_resumed.resize(logits_resumed_raw.size());
            for (std::size_t i = 0; i < logits_resumed_raw.size(); ++i) {
                std::uint32_t u = static_cast<std::uint32_t>(logits_resumed_raw[i]) << 16;
                float f;
                std::memcpy(&f, &u, sizeof(float));
                logits_resumed[i] = f;
            }
            (void)program.commit(std::move(*pref_t2.pending), commit_dec);
        }
        if (pref_t2.complete) {
            break;
        }
    }
    (void)program.finish(seq_t2);

    // 8. Execute Turn 2 From Scratch
    auto prepared_t2_scratch = frontend.prepare(std::move(input_t2));
    auto base_t2_scr = program.plan_request(prepared_t2_scratch, exec_options);
    auto cand_t2_scr = program.inspect_admission(prepared_t2_scratch, base_t2_scr, ninfer::runtime::LaneId(0), nullptr, nullptr, std::nullopt, false, cost_model);
    if (!cand_t2_scr) { throw std::runtime_error("Turn 2 scratch admission failed"); }
    auto res_t2_scr = program.seal_identity(*cand_t2_scr, prepared_t2_scratch);
    (void)program.start_resource_transaction(std::move(*res_t2_scr), std::move(prepared_t2_scratch), cancellation);
    auto prog_t2_scr = program.progress_context_transaction(cancellation);
    SequenceHandle seq_t2_scr = std::get_if<MaterializationResult>(&prog_t2_scr)->published->sequence;
    program.finalize_context_transaction();

    std::vector<float> logits_scratch;
    std::vector<std::uint16_t> logits_scratch_raw;
    while (true) {
        auto pref_scr = program.advance_prefill(seq_t2_scr);
        if (pref_scr.capture.has_value()) {
            program.skip_capture(std::move(*pref_scr.capture));
        }
        if (pref_scr.pending.has_value()) {
            const auto& l_tensor = program.impl_->allocation_.round_tensors().logits;
            logits_scratch_raw.resize(l_tensor.numel());
            CHECK_CUDA(cudaMemcpy(logits_scratch_raw.data(), l_tensor.data,
                                  logits_scratch_raw.size() * sizeof(std::uint16_t),
                                  cudaMemcpyDeviceToHost));
            logits_scratch.resize(logits_scratch_raw.size());
            for (std::size_t i = 0; i < logits_scratch_raw.size(); ++i) {
                std::uint32_t u = static_cast<std::uint32_t>(logits_scratch_raw[i]) << 16;
                float f;
                std::memcpy(&f, &u, sizeof(float));
                logits_scratch[i] = f;
            }
            (void)program.commit(std::move(*pref_scr.pending), commit_dec);
        }
        if (pref_scr.complete) {
            break;
        }
    }
    (void)program.finish(seq_t2_scr);

    // 9. Compute Metrics & Output
    if (logits_resumed.empty() || logits_scratch.empty() || logits_resumed.size() != logits_scratch.size()) {
        throw std::runtime_error("Empty or mismatched logits size between resumed and scratch prefill");
    }

    double diff_norm_sq = 0.0;
    double ref_norm_sq = 0.0;
    for (std::size_t i = 0; i < logits_resumed.size(); ++i) {
        double d = static_cast<double>(logits_resumed[i]) - static_cast<double>(logits_scratch[i]);
        double r = static_cast<double>(logits_scratch[i]);
        diff_norm_sq += d * d;
        ref_norm_sq += r * r;
    }
    const double rel_l2 = (ref_norm_sq > 0.0) ? std::sqrt(diff_norm_sq / ref_norm_sq) : 0.0;

    const auto max_it_res = std::max_element(logits_resumed.begin(), logits_resumed.end());
    const std::int32_t argmax_res = static_cast<std::int32_t>(std::distance(logits_resumed.begin(), max_it_res));
    const float logit_res = *max_it_res;

    const auto max_it_scr = std::max_element(logits_scratch.begin(), logits_scratch.end());
    const std::int32_t argmax_scr = static_cast<std::int32_t>(std::distance(logits_scratch.begin(), max_it_scr));
    const float logit_scr = *max_it_scr;

    const bool argmax_match = (argmax_res == argmax_scr);

    // Dump raw logits if requested
    if (!opts.dump_gen_logits.empty()) {
        std::filesystem::create_directories(opts.dump_gen_logits);
        const std::string res_path = opts.dump_gen_logits + "/gen_000_resumed.bin";
        const std::string scr_path = opts.dump_gen_logits + "/gen_000_scratch.bin";
        std::ofstream f_res(res_path, std::ios::binary);
        if (f_res) { f_res.write(reinterpret_cast<const char*>(logits_resumed_raw.data()), logits_resumed_raw.size() * sizeof(std::uint16_t)); }
        std::ofstream f_scr(scr_path, std::ios::binary);
        if (f_scr) { f_scr.write(reinterpret_cast<const char*>(logits_scratch_raw.data()), logits_scratch_raw.size() * sizeof(std::uint16_t)); }
    }

    if (opts.json_output) {
        std::cout << "{\n"
                  << "  \"mode\": \"continuation-check\",\n"
                  << "  \"turn1_prompt_tokens\": " << t1_prompt_tokens << ",\n"
                  << "  \"turn1_capture_frontier\": " << t1_capture_frontier << ",\n"
                  << "  \"turn2_prompt_tokens\": " << t2_prompt_tokens << ",\n"
                  << "  \"turn2_reused_tokens\": " << reused_tokens << ",\n"
                  << "  \"argmax_resumed\": " << argmax_res << ",\n"
                  << "  \"logit_resumed\": " << logit_res << ",\n"
                  << "  \"argmax_scratch\": " << argmax_scr << ",\n"
                  << "  \"logit_scratch\": " << logit_scr << ",\n"
                  << "  \"argmax_match\": " << (argmax_match ? "true" : "false") << ",\n"
                  << "  \"rel_l2\": " << rel_l2 << ",\n"
                  << "  \"status\": \"" << (argmax_match && rel_l2 <= 0.1 ? "OK" : "MISMATCH") << "\"\n"
                  << "}\n";
    } else {
        std::cout << "\n=== Continuation Check Report ===\n"
                  << "Turn 1 Prompt Tokens:        " << t1_prompt_tokens << "\n"
                  << "Turn 1 Capture Frontier (F): " << t1_capture_frontier << "\n"
                  << "Turn 2 Prompt Tokens:        " << t2_prompt_tokens << "\n"
                  << "Turn 2 Reused Prompt Tokens: " << reused_tokens << "\n"
                  << "Resumed First-Gen Argmax:    " << argmax_res << " (logit: " << std::fixed << std::setprecision(4) << logit_res << ")\n"
                  << "Scratch First-Gen Argmax:    " << argmax_scr << " (logit: " << std::fixed << std::setprecision(4) << logit_scr << ")\n"
                  << "Argmax Match:                " << (argmax_match ? "YES" : "NO") << "\n"
                  << "Relative L2 Error:           " << std::scientific << std::setprecision(6) << rel_l2 << "\n"
                  << "Status:                      " << (argmax_match && rel_l2 <= 0.1 ? "OK" : "DIVERGENCE_MISMATCH") << "\n";
    }

    return (argmax_match && rel_l2 <= 0.1) ? 0 : 1;
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
        } else if (opts.mode == "chat-diagnostic") {
            return run_chat_diagnostic(opts);
        } else if (opts.mode == "execute-vision") {
            return run_execute_vision(opts);
        } else if (opts.mode == "continuation-check") {
            return run_continuation_check(opts);
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
