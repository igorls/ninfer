#include "models/qwen4_exp/program/program_impl.h"
#include "models/qwen4_exp/program/context_work.h"
#include "models/qwen4_exp/program/context.h"
#include "models/qwen4_exp/program/planning/graph_profiles.h"
#include "core/nvtx.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen4_exp::detail {

namespace {

void validate_graph_profiles(const std::vector<GraphExecutionProfile>& profiles,
                             std::uint32_t max_frontier, const char* label);

template <class Prepare>
void instantiate_graph_family(DecodeGraphFamily& family, const char* label, DeviceContext& device,
                              Prepare&& prepare);

void validate_graph_profiles(const std::vector<GraphExecutionProfile>& profiles,
                             std::uint32_t max_frontier, const char* label) {
    if (profiles.empty() || profiles.front().min != 0 || profiles.back().max != max_frontier) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage has invalid endpoints");
    }
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        if (profiles[i].min > profiles[i].max ||
            (i != 0 && profiles[i].min != profiles[i - 1].max + 1)) {
            throw std::logic_error(std::string(label) + " CUDA Graph coverage has a gap");
        }
    }
}

template <class Prepare>
void instantiate_graph_family(DecodeGraphFamily& family, const char* label, DeviceContext& device,
                              Prepare&& prepare) {
    if (family.profiles.empty()) {
        throw std::logic_error(std::string(label) + " CUDA Graph family has no profiles");
    }

    for (std::size_t i = 0; i < family.profiles.size(); ++i) {
        DecodeGraphProfile& profile = family.profiles[i];
        if (!profile.definition.ready()) {
            throw std::logic_error(std::string(label) + " CUDA Graph definition is empty");
        }
        const auto existing =
            std::find_if(family.topologies.begin(), family.topologies.end(),
                         [&](const DecodeGraphTopology& topology) {
                             return topology.topology_class == profile.topology_class;
                         });
        if (existing != family.topologies.end()) { continue; }

        family.topologies.emplace_back();
        DecodeGraphTopology& topology = family.topologies.back();
        topology.topology_class       = profile.topology_class;
        topology.executable.instantiate(profile.definition);
        topology.installed_profile = i;
    }

    const auto install_and_upload = [&](DecodeGraphTopology& topology, std::size_t profile_index) {
        DecodeGraphProfile& profile = family.profiles[profile_index];
        if (topology.installed_profile != profile_index) {
            topology.executable.update(profile.definition);
            topology.installed_profile = profile_index;
        }
        topology.executable.upload(device.stream);
        device.synchronize();
    };

    for (DecodeGraphTopology& topology : family.topologies) {
        std::optional<std::size_t> first_profile;
        for (std::size_t i = 0; i < family.profiles.size(); ++i) {
            if (family.profiles[i].topology_class == topology.topology_class) {
                if (!first_profile) {
                    first_profile = i;
                    install_and_upload(topology, i);

                    DecodeGraphProfile& profile = family.profiles[i];
                    prepare(profile.min_execution_frontier, profile.batch_size);
                    device.synchronize();
                    topology.executable.launch(device.stream);
                    device.synchronize();
                    continue;
                }
                install_and_upload(topology, i);
            }
        }
        if (!first_profile) {
            throw std::logic_error(std::string(label) + " CUDA Graph topology has no definitions");
        }
        if (topology.installed_profile != *first_profile) {
            install_and_upload(topology, *first_profile);
        }
    }
}

} // namespace

void ProgramImpl::prepare_graphs() {
    if (!use_cuda_graph) { return; }
    nvtx::ScopedRange prepare_range(nvtx::Name::CudaGraphPrepare, nvtx::Category::Graph);

    std::array<StateImageHandle, kMaximumConcurrency> capture_states{};
    for (std::uint32_t row = 0; row < max_concurrency; ++row) {
        std::optional<StateImageHandle> state = state_store->reserve_reset(device.stream);
        if (!state) { throw std::bad_alloc(); }
        capture_states[row] = *state;
    }
    const auto capture_state_slot = [&](std::uint32_t row) {
        return state_store->physical_slot(capture_states.at(row));
    };

    std::vector<KVAddressSpaceHandle> text_capture_allocations;
    {
        DeviceKVPagePool& pool       = decoder->text_kv.page_pool();
        KVExecutionTablePool& tables = decoder->text_kv.execution_tables();
        if (pool.capacity_pages() < max_concurrency) {
            throw std::invalid_argument(
                "target KV cache cannot provide one Paged KV page per concurrent request");
        }
        text_capture_allocations.reserve(max_concurrency);
        for (std::uint32_t row = 0; row < max_concurrency; ++row) {
            std::optional<KVAddressSpaceHandle> allocation =
                text_kv_addresses->create_active(1, static_cast<std::int32_t>(row));
            if (!allocation) { throw std::bad_alloc(); }
            text_capture_allocations.push_back(*allocation);
            text_kv_addresses->ensure_mapped_to_tokens(*allocation, 1, device.stream);
            // Capture profiles exercise arbitrary context envelopes. Repeating each row's private
            // page across its temporary table keeps every dummy read/write address valid without
            // reserving C full contexts solely for graph construction.
            tables.publish_repeated(text_kv_addresses->execution_row(*allocation).handle(),
                                    text_kv_addresses->physical_page(*allocation, 0),
                                    tables.logical_page_capacity(), device.stream);
        }
    }
    device.synchronize();

    const auto prepare_representative = [&](std::uint32_t frontier, std::uint32_t batch_size) {
        if (batch_size == 0 || batch_size > max_concurrency) {
            throw std::logic_error("CUDA Graph representative batch is invalid");
        }
        work.reset();
        std::vector<DeviceKVPageHandle> pages;
        pages.reserve(batch_size);
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            pages.push_back(text_kv_addresses->physical_page(text_capture_allocations[row], 0));
            state_images->zero_slot(capture_state_slot(row), device.stream);
        }
        decoder->text_kv.page_pool().zero_pages(pages, device.stream);
        *ordinary_host_ingress = {};
        *ordinary_host_egress  = {};
        const std::int32_t position = checked_i32(frontier, "graph representative position");
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            ordinary_host_ingress->tokens[row]          = 0;
            ordinary_host_ingress->cache_positions[row] = position;
            for (std::uint32_t axis = 0; axis < 3; ++axis) {
                ordinary_host_ingress->rope_positions[axis * batch_size + row] = position;
            }
            ordinary_host_ingress->text_kv_table_rows[row]      = static_cast<std::int32_t>(row);
            ordinary_host_ingress->state_source_slots[row]      = capture_state_slot(row);
            ordinary_host_ingress->state_destination_slots[row] = capture_state_slot(row);
            ordinary_host_ingress->sampling[row]                = {};
        }
    };

    const auto& config        = parameters.model.config().text;
    const auto indexer_block  = config.indexer.compress_ratio;
    const auto selected       = config.indexer.budget / config.indexer.compress_ratio;
    const auto profiles       = ordinary_graph_profiles(capacity, indexer_block, selected);
    validate_graph_profiles(profiles, capacity - 1, "ordinary");
    execution::OrdinaryBatchContext ordinary_state{
        {device, parameters, work, *state_images, *ple_gather, io, prefill_hidden, prefill_chunk},
        decoder->text_kv,
        *io.ordinary,
        *ordinary_host_ingress,
        *ordinary_host_egress,
        state_images->continuation_hidden_store()};
    // One eager traversal loads every module before the first capture.
    const GraphExecutionProfile code_warm = profiles.front();
    prepare_representative(code_warm.min, 1);
    device.synchronize();
    execution::ordinary_decode_batch(
        ordinary_state, 1, {indexer_envelope_blocks(code_warm.max, indexer_block)}, nullptr);
    device.synchronize();

    ordinary_graphs.profiles.reserve(profiles.size() * max_concurrency);
    for (std::uint32_t batch_size = 1; batch_size <= max_concurrency; ++batch_size) {
        for (const GraphExecutionProfile planned : profiles) {
            ordinary_graphs.profiles.emplace_back();
            DecodeGraphProfile& profile    = ordinary_graphs.profiles.back();
            profile.batch_size             = batch_size;
            profile.min_execution_frontier = planned.min;
            profile.max_execution_frontier = planned.max;
            profile.topology_class = planned.topology_class * max_concurrency + (batch_size - 1U);
            execution::capture_ordinary_decode_batch(
                ordinary_state, static_cast<std::int32_t>(batch_size),
                {indexer_envelope_blocks(planned.max, indexer_block)}, profile.definition);
        }
    }
    instantiate_graph_family(ordinary_graphs, "ordinary", device, prepare_representative);

    state_images->zero_all(device.stream);
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    device.synchronize();
    for (std::uint32_t row = 0; row < max_concurrency; ++row) {
        if (!state_store->release(capture_states[row])) {
            throw std::logic_error("CUDA Graph capture StateImage could not be released");
        }
    }
    for (const KVAddressSpaceHandle allocation : text_capture_allocations) {
        text_kv_addresses->deactivate(allocation);
        if (!text_kv_addresses->release(allocation)) {
            throw std::logic_error("CUDA Graph capture KV address space could not be released");
        }
    }
}

} // namespace ninfer::models::qwen4_exp::detail
