#pragma once

#include "artifact/reader.h"
#include "core/device.h"
#include "models/qwen4_exp/definition.h"
#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/execution/profile.h"
#include "models/qwen4_exp/program/planning.h"

#include <algorithm>

namespace ninfer::test::qwen4_exp {

inline runtime::ContextCapacityPlan context_capacity_plan(const EngineOptions& options) {
    artifact::Reader reader(options.artifact_path);
    const auto definition = models::qwen4_exp::read_definition(reader, models::load_options(options));
    const auto profile = models::qwen4_exp::execution::describe_model(*definition);
    DeviceContext device(options.device);
    const auto load = models::qwen4_exp::resolve_load(reader, definition, device.device());
    return models::qwen4_exp::describe_execution_capacity(
        profile, load.ple_table_owner()->geometry(), options, device);
}

// Retention competes with complete public execution working sets in the same physical pool.
// Account for record packing, including each stored history, rather than convert tokens to bytes.
inline std::size_t context_capacity_bytes(const EngineOptions& options,
                                          std::uint32_t retained_states = 0,
                                          std::uint32_t full_contexts = 0) {
    auto plan = context_capacity_plan(options);
    const auto geometry = describe_context_pool(plan.records);
    auto bytes = plan.minimum_capacity().device_bytes;
    if (full_contexts) {
        std::vector<RecordRequest> records{{geometry.spec_classes.at(0), options.max_concurrency}};
        for (std::size_t index = 1; index < geometry.spec_classes.size(); ++index) {
            const auto tokens = options.max_context - (index == 2 ? 1 : 0);
            records.push_back({geometry.spec_classes[index], full_contexts * ((tokens + 63) / 64)});
        }
        plan.working_sets.push_back({"fixture full contexts", std::move(records)});
    }
    for (const auto& set : plan.working_sets) {
        auto records = set.records;
        records.push_back({geometry.spec_classes.at(0), retained_states});
        bytes = std::max(bytes, geometry.required_bytes(records));
    }
    return bytes;
}

inline void set_context_capacity(EngineOptions& options, std::uint32_t retained_states = 0,
                                  std::uint32_t full_contexts = 0) {
    options.context_capacity = ContextCapacityPolicy::explicit_capacity(
        context_capacity_bytes(options, retained_states, full_contexts));
}

} // namespace ninfer::test::qwen4_exp
