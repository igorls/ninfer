#pragma once

#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/program/program.h"

#include <cstdint>

namespace ninfer::models::qwen4_exp {

inline constexpr std::uint32_t kPrefillChunkAlignment = 128;

} // namespace ninfer::models::qwen4_exp

namespace ninfer::models::qwen4_exp::detail {
using ContractAccess = RuntimeContractAccess;

// Frontier of the speculative backend's KV beside a Main KV frontier. The Program supports
// ordinary decoding only until the Qwen4Exp MTP backend lands; its backend KV store stays empty.
[[nodiscard]] inline std::uint32_t backend_frontier_at(SpeculativeBackend backend,
                                                       std::uint32_t main_frontier) noexcept {
    if (backend == SpeculativeBackend::Mtp) { return main_frontier == 0 ? 0U : main_frontier - 1U; }
    return 0U;
}

} // namespace ninfer::models::qwen4_exp::detail
