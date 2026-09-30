#include "artifact/fixture.h"
#include "artifact/reader.h"
#include "core/device_memory.h"
#include "models/qwen3_5/frontend/digest.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen4_exp/load.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace flash = ninfer::models::qwen4_exp;
using ninfer::test::artifact_fixture::require;

struct Sample {
    std::uint64_t offset;
    std::vector<std::byte> bytes;
};

// Byte samples at the start, middle, end and plane boundaries of one stored object.
std::vector<Sample> read_samples(const artifact::Reader& reader, artifact::ObjectHandle handle) {
    const auto& object   = reader.directory().tensor(handle);
    const auto& geometry = reader.geometry(handle);
    const auto size      = std::min<std::uint64_t>(64, object.bytes);
    std::set<std::uint64_t> offsets{0, (object.bytes - size) / 2, object.bytes - size};
    if (geometry.scale_bytes) { offsets.insert(geometry.scale_offset); }
    if (geometry.divisor_offset) { offsets.insert(geometry.divisor_offset); }
    std::vector<Sample> out;
    for (const auto offset : offsets) {
        const auto count = std::min<std::uint64_t>(size, object.bytes - offset);
        out.push_back({offset, reader.read_range(object.offset + offset, count)});
    }
    return out;
}

std::string hex_digest(std::span<const std::byte> bytes) {
    return qwen3_5::frontend::sha256_hex(qwen3_5::frontend::sha256(
        {reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()}));
}

// SHA256 of every Device parent (read back) and every mapped parent, keyed by object id.
void write_digests(const std::filesystem::path& path,
                   const std::vector<std::pair<std::string, const WeightParent*>>& device,
                   const std::vector<std::pair<std::string, const WeightParent*>>& mapped) {
    std::map<std::string, std::string> device_digests;
    std::map<std::string, std::string> mapped_digests;
    std::mutex lock;
    std::atomic<std::size_t> next{0};
    std::exception_ptr failure;
    const auto total   = device.size() + mapped.size();
    const auto workers = std::max(1U, std::min(32U, std::thread::hardware_concurrency()));
    std::vector<std::thread> threads;
    for (unsigned worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&] {
            try {
                CUDA_CHECK(cudaSetDevice(0));
                std::vector<std::byte> host;
                for (auto i = next++; i < total; i = next++) {
                    const bool on_device       = i < device.size();
                    const auto& [name, parent] = on_device ? device[i] : mapped[i - device.size()];
                    const auto bytes           = static_cast<std::size_t>(parent->geometry.bytes);
                    std::string digest;
                    if (on_device) {
                        host.resize(bytes);
                        CUDA_CHECK(
                            cudaMemcpy(host.data(), parent->data, bytes, cudaMemcpyDeviceToHost));
                        digest = hex_digest(host);
                    } else {
                        digest = hex_digest({parent->data, bytes});
                    }
                    const std::lock_guard guard(lock);
                    (on_device ? device_digests : mapped_digests)[name] = digest;
                }
            } catch (...) {
                const std::lock_guard guard(lock);
                failure = std::current_exception();
            }
        });
    }
    for (auto& thread : threads) { thread.join(); }
    if (failure) { std::rethrow_exception(failure); }
    std::ofstream out(path);
    out << "{\n  \"device\": {";
    const char* separator = "\n";
    for (const auto& [name, digest] : device_digests) {
        out << separator << "    \"" << name << "\": \"" << digest << '"';
        separator = ",\n";
    }
    out << "\n  },\n  \"mapped\": {";
    separator = "\n";
    for (const auto& [name, digest] : mapped_digests) {
        out << separator << "    \"" << name << "\": \"" << digest << '"';
        separator = ",\n";
    }
    out << "\n  }\n}\n";
    require(out.good(), "cannot write the digest file");
}

// Exercise native preparation of every mathematical Use after Reader destruction.
void check_native_inputs(const flash::Model& model) {
    const auto linear = [&](flash::WeightId id) {
        (void)ops::prepare_linear_weight(model.input(id));
    };
    const auto direct = [&](flash::WeightId id) {
        const auto& view = model.weight(id).view;
        (void)weight_tensor(view, {static_cast<std::int32_t>(weight_element_count(view.shape))});
    };
    const auto hyper = [&](const flash::HyperConnectionWeights& weights) {
        direct(weights.norm);
        linear(weights.input_mix_down);
        linear(weights.input_mix_up);
        linear(weights.block_inject);
    };
    const auto mixer = [&](const flash::HyperMixerWeights& weights) {
        direct(weights.norm);
        linear(weights.input_mix_down);
        linear(weights.input_mix_up);
    };
    std::size_t banks      = 0;
    const auto expert_bank = [&](flash::WeightId id) {
        const auto bank = ops::prepare_nvfp4_expert_bank_weight(model.input(id));
        require(bank.policy == ops::LinearPolicy::AllowA4,
                "expert bank lost its dynamic A4 activation policy");
        std::vector<float> divisors(static_cast<std::size_t>(bank.experts));
        CUDA_CHECK(cudaMemcpy(divisors.data(), bank.weight_scale_divisors,
                              divisors.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (const float divisor : divisors) {
            require(std::isfinite(divisor) && divisor > 0,
                    "expert bank holds a non-positive per-expert weight divisor");
        }
        ++banks;
    };
    const auto block = [&](const flash::BlockWeights& weights) {
        hyper(weights.attention_hyper);
        hyper(weights.mlp_hyper);
        if (const auto* attention = std::get_if<flash::AttentionWeights>(&weights.mixer)) {
            linear(attention->query_gate_key_value);
            linear(attention->output);
            linear(attention->indexer_query_key);
            direct(attention->query_norm);
            direct(attention->key_norm);
            direct(attention->indexer_query_norm);
            direct(attention->indexer_key_norm);
        } else {
            const auto& gdn = std::get<flash::GdnWeights>(weights.mixer);
            linear(gdn.query_key_value_z);
            linear(gdn.a_b_projection);
            linear(gdn.output);
            direct(gdn.a_log);
            direct(gdn.dt_bias);
            direct(gdn.norm);
        }
        const auto& moe = weights.moe;
        linear(moe.router);
        linear(moe.shared_expert_gate);
        linear(moe.shared_gate);
        linear(moe.shared_up);
        linear(moe.shared_down);
        expert_bank(moe.experts_gate_up);
        expert_bank(moe.experts_down);
    };
    const auto& weights = model.weights();
    (void)native_weight(model.weight(weights.text.token_embedding).view);
    (void)ops::prepare_linear_weight(model.input(weights.text.output_head_use));
    for (const auto& layer : weights.text.layers) { block(layer); }
    linear(weights.text.ple.key_projection);
    linear(weights.text.ple.value_projection);
    mixer(weights.text.final_mixer);
    if (weights.mtp) {
        const auto& mtp = *weights.mtp;
        linear(mtp.embedding_projection);
        linear(mtp.hidden_projection);
        direct(mtp.embedding_norm);
        direct(mtp.hidden_norm);
        block(mtp.layer);
        mixer(mtp.final_mixer);
        (void)ops::prepare_linear_weight(model.input(mtp.output_head_use));
    }
    if (weights.vision) {
        const auto& vision = *weights.vision;
        linear(vision.patch_embedding);
        for (const auto& layer : vision.layers) {
            const std::array qkv{model.input(layer.query), model.input(layer.key),
                                 model.input(layer.value)};
            (void)ops::prepare_linear_weight(qkv);
            linear(layer.output);
            linear(layer.fc1);
            linear(layer.fc2);
        }
        linear(vision.merger_fc1);
        linear(vision.merger_fc2);
    }
    const std::size_t expected_banks =
        2 * (weights.text.layers.size() + (weights.mtp.has_value() ? 1 : 0));
    require(banks == expected_banks, "not every expert bank was admitted");
    if (weights.mtp) {
        ninfer::test::artifact_fixture::rejects<std::invalid_argument>(
            [&] { (void)model.input(weights.text.output_head); },
            "ambiguous shared output-head Use was silently selected");
    }
}

void check_ple(const flash::Model& model) {
    const auto& table = model.ple_table();
    const auto& ple   = model.config().text.ple;
    require(table.shards.size() == ple.shards && table.layer_multipliers.size() == ple.ngram_size &&
                table.ngram_head_offsets.size() == ple.heads() &&
                table.ngram_head_vocab_sizes.size() == ple.heads(),
            "PLE table lost shards or index tables");
    for (const auto& shard : table.shards) {
        require(shard.parts.size() == 1 && shard.shape[0] == table.rows_per_shard() &&
                    shard.shape[1] == ple.head_width() &&
                    shard.parts[0].parent->geometry.format == QType::U4Z8_G16_FP16,
                "PLE shard view differs from its table geometry");
        cudaPointerAttributes attributes{};
        CUDA_CHECK(cudaPointerGetAttributes(&attributes, shard.parts[0].parent->data));
        require(attributes.type == cudaMemoryTypeUnregistered,
                "PLE shard is not a plain Host mapping");
    }
    std::cout << "PLE: layer=" << ple.layer << " shards=" << table.shards.size()
              << " rows_per_shard=" << table.rows_per_shard() << " multipliers=";
    for (const auto value : table.layer_multipliers) { std::cout << value << ' '; }
    std::cout << "head_rows="
              << table.ngram_head_offsets.back() + table.ngram_head_vocab_sizes.back() << '\n';
}

void check_frontend(const flash::Model& model) {
    const auto frontend = qwen3_5::make_frontend(
        model.resources(), {.architecture   = Architecture::Qwen4Exp,
                            .vision_enabled = model.config().vision.has_value(),
                            .max_context    = 8192});
    const auto ids = frontend.tokenize_text("Hello, Flash-Next");
    require(!ids.empty(), "Frontend did not tokenize text");
    std::string decoded;
    for (const auto id : ids) { decoded += frontend.token_bytes(id); }
    require(decoded == "Hello, Flash-Next", "tokenizer did not round-trip text");
    PromptInput input;
    ChatMessage message;
    message.parts.push_back(
        MessagePart{.kind = MessagePartKind::Text, .text = "Name three primes.", .media = {}});
    input.messages.push_back(std::move(message));
    const auto prompt = frontend.prepare(std::move(input));
    require(prompt.summary().prompt_tokens > ids.size(),
            "the carried chat template did not render a prompt");
    std::cout << "Frontend: V=" << model.resources().public_token_count
              << " chat prompt tokens=" << prompt.summary().prompt_tokens << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path path;
        std::filesystem::path digests;
        LoadOptions options{.vision = true, .speculative = SpeculativeBackend::Mtp};
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const auto value = [&]() -> std::string {
                if (++i >= argc) { throw std::invalid_argument("missing value for " + arg); }
                return argv[i];
            };
            if (arg == "--artifact") {
                path = value();
            } else if (arg == "--digests") {
                digests = value();
            } else if (arg == "--text-only") {
                options = {};
            } else {
                throw std::invalid_argument(
                    "expected --artifact PATH [--digests OUT] [--text-only]");
            }
        }
        if (path.empty()) {
            std::cout << "SKIP: supply an explicit --artifact\n";
            return 77;
        }
        std::unique_ptr<DeviceContext> device;
        std::unique_ptr<flash::Model> model;
        std::map<std::string, std::vector<Sample>> samples;
        std::uint64_t expected_h2d    = 0;
        std::uint64_t expected_mapped = 0;
        std::size_t bindings          = 0;
        const auto before             = query_device_memory(0);
        {
            artifact::Reader reader(path);
            auto plan = flash::plan_load(reader, options);
            bindings  = plan.binding_count();
            if (options.vision && options.mtp()) {
                require(bindings == reader.directory().bindings.size(),
                        "a full load did not bind every artifact Binding");
            }
            for (const auto& item : plan.materialization().device_objects) {
                expected_h2d += reader.directory().tensor(item.object).bytes;
                samples[reader.directory().tensor(item.object).id] =
                    read_samples(reader, item.object);
            }
            for (const auto handle : plan.materialization().mapped_objects) {
                expected_mapped += reader.directory().tensor(handle).bytes;
                samples[reader.directory().tensor(handle).id] = read_samples(reader, handle);
            }
            device = std::make_unique<DeviceContext>();
            model  = flash::materialize_model(std::move(plan), *device);
        }
        const auto& stats = model->storage_stats();
        require(stats.h2d_bytes == expected_h2d, "H2D did not cover all selected parent bytes");
        require(stats.mapped_bytes == expected_mapped &&
                    stats.mapped_object_count == model->config().text.ple.shards,
                "mapped residency did not cover the PLE table");
        std::vector<std::pair<std::string, const WeightParent*>> device_parents;
        std::vector<std::pair<std::string, const WeightParent*>> mapped_parents;
        std::set<const WeightParent*> visited;
        const auto check_parent = [&](const std::string& object, const WeightParent* parent,
                                      bool on_device) {
            if (!visited.insert(parent).second) { return; }
            (on_device ? device_parents : mapped_parents).emplace_back(object, parent);
            for (const auto& sample : samples.at(object)) {
                std::vector<std::byte> received(sample.bytes.size());
                if (on_device) {
                    CUDA_CHECK(cudaMemcpy(received.data(), parent->data + sample.offset,
                                          received.size(), cudaMemcpyDeviceToHost));
                } else {
                    std::memcpy(received.data(), parent->data + sample.offset, received.size());
                }
                require(received == sample.bytes, "resident parent differs from its object bytes");
            }
        };
        for (const auto& weight : model->weight_data()) {
            require(weight.view.parts.size() == weight.source_objects.size(),
                    "diagnostic parent association lost");
            for (std::size_t i = 0; i < weight.view.parts.size(); ++i) {
                check_parent(weight.source_objects[i], weight.view.parts[i].parent, true);
            }
        }
        const auto& ple = model->config().text.ple;
        for (std::size_t i = 0; i < model->ple_table().shards.size(); ++i) {
            check_parent("text/layers/" + std::to_string(ple.layer) + "/ple/embedding/shards/" +
                             std::to_string(i),
                         model->ple_table().shards[i].parts[0].parent, false);
        }
        require(visited.size() == samples.size(),
                "some resident parents are not reachable from the model");
        check_native_inputs(*model);
        check_ple(*model);
        check_frontend(*model);
        device->synchronize();
        const auto after = query_device_memory(0);
        std::cout << path.filename().string() << ": qwen4_exp loading passed, bindings=" << bindings
                  << " device_parents=" << device_parents.size()
                  << " device_bytes=" << stats.device_capacity_bytes << " ("
                  << format_device_memory_bytes(stats.device_capacity_bytes)
                  << ") mapped_parents=" << mapped_parents.size()
                  << " mapped_bytes=" << stats.mapped_bytes
                  << " mapped_copies=" << stats.mapped_copy_count
                  << " mapped_warm_s=" << stats.mapped_warm_seconds
                  << " upload_s=" << stats.upload_seconds << " device_used_delta="
                  << format_device_memory_bytes(before.free_bytes > after.free_bytes
                                                    ? before.free_bytes - after.free_bytes
                                                    : 0)
                  << '\n';
        if (!digests.empty()) {
            write_digests(digests, device_parents, mapped_parents);
            std::cout << "digests: " << digests.string() << '\n';
        }
        model.reset();
        device->synchronize();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
