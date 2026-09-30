// Validation, workspace sizing and route selection of the NVFP4-bank SparseMoe overload.

#include "ninfer/ops/sparse_moe.h"

#include "core/layout.h"
#include "core/nvtx.h"
#include "ops/sparse_moe/nvfp4_bank/nvfp4_bank_moe.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::ops {
namespace detail {

Nvfp4MoeRoute resolve_nvfp4_moe_route(std::int32_t tokens, bool allow_a4) {
    if (tokens <= 0) { throw std::invalid_argument("sparse_moe: T must be positive"); }
    if (tokens <= kNvfp4MoeDecodeMaxTokens) { return Nvfp4MoeRoute::Decode; }
    return allow_a4 && tokens >= kNvfp4MoeA4MinTokens ? Nvfp4MoeRoute::GroupedA4
                                                      : Nvfp4MoeRoute::GroupedA16;
}

std::size_t nvfp4_moe_workspace_bytes(std::int32_t tokens, Nvfp4MoeRoute route) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_nvfp4_moe_workspace(layout, tokens, route);
    return layout.peak_bytes(1);
}

} // namespace detail

namespace {

using detail::kNvfp4MoeExperts;
using detail::kNvfp4MoeHidden;
using detail::kNvfp4MoeIntermediate;
using detail::Nvfp4MoeRoute;

struct AddressRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end   = 0;
    std::string name;
};

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

AddressRange address_range(const void* pointer, std::size_t bytes, std::string name) {
    if (pointer == nullptr || bytes == 0) {
        throw std::invalid_argument("sparse_moe: " + name + " storage must be non-empty");
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::overflow_error("sparse_moe: " + name + " address range overflows");
    }
    return {begin, begin + bytes, std::move(name)};
}

void require_disjoint(const std::vector<AddressRange>& ranges) {
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            if (ranges[i].begin < ranges[j].end && ranges[j].begin < ranges[i].end) {
                throw std::invalid_argument("sparse_moe: " + ranges[i].name + " overlaps " +
                                            ranges[j].name);
            }
        }
    }
}

std::int32_t require_activation(const Tensor& tensor, const char* name) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != kNvfp4MoeHidden || tensor.ne[1] < 1 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("sparse_moe: invalid ") + name);
    }
    return tensor.ne[1];
}

void require_bf16(const Weight& weight, std::int32_t n, std::int32_t k, const char* name,
                  std::vector<AddressRange>& ranges) {
    const std::size_t bytes = static_cast<std::size_t>(n) * k * 2;
    if (weight.qtype != QType::BF16 || weight.layout != QuantLayout::Contiguous ||
        weight.ndim != 2 || weight.n != n || weight.k != k || weight.shape[0] != n ||
        weight.shape[1] != k || weight.shape[2] != 1 || weight.shape[3] != 1 ||
        weight.padded_shape[0] != n || weight.padded_shape[1] != k || weight.qhigh != nullptr ||
        weight.payload_bytes < bytes || !aligned_to(weight.qdata, 16)) {
        throw std::invalid_argument(std::string("sparse_moe: ") + name +
                                    " must be aligned contiguous BF16 of its registered shape");
    }
    ranges.push_back(address_range(weight.qdata, bytes, name));
}

void require_bank(const Nvfp4ExpertBankWeight& bank, std::int32_t n, std::int32_t k,
                  const char* name, std::vector<AddressRange>& ranges) {
    const auto elements = static_cast<std::uint64_t>(n) * k;
    if (bank.experts != kNvfp4MoeExperts || bank.n != n || bank.k != k ||
        bank.code_bytes_per_expert != elements / 2 ||
        bank.scale_bytes_per_expert != elements / 16 || !aligned_to(bank.codes, 16) ||
        !aligned_to(bank.scales, 16) || !aligned_to(bank.weight_scale_divisors, 16) ||
        !valid_linear_policy(bank.policy)) {
        throw std::invalid_argument(std::string("sparse_moe: ") + name +
                                    " must be a complete aligned NVFP4 [512," + std::to_string(n) +
                                    "," + std::to_string(k) + "] expert bank");
    }
    const std::string label(name);
    ranges.push_back(
        address_range(bank.codes, bank.code_bytes_per_expert * kNvfp4MoeExperts, label + " codes"));
    ranges.push_back(address_range(bank.scales, bank.scale_bytes_per_expert * kNvfp4MoeExperts,
                                   label + " scales"));
    ranges.push_back(address_range(bank.weight_scale_divisors, sizeof(float) * kNvfp4MoeExperts,
                                   label + " divisors"));
}

bool allows_bank_a4(LinearPolicy gate_up, LinearPolicy down) {
    if (!valid_linear_policy(gate_up) || !valid_linear_policy(down)) {
        throw std::invalid_argument("sparse_moe: invalid expert-bank policy");
    }
    return allows_a4(gate_up) && allows_a4(down);
}

} // namespace

std::size_t sparse_moe_workspace_capacity_bytes(LinearPolicy gate_up_policy,
                                                LinearPolicy down_policy, std::int32_t min_tokens,
                                                std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("sparse_moe workspace: invalid token interval");
    }
    const bool a4 = allows_bank_a4(gate_up_policy, down_policy);
    // Every route's requirement grows with T, so each route's largest admitted T bounds it.
    std::size_t required = 0;
    for (std::int32_t tokens :
         {std::min(max_tokens, detail::kNvfp4MoeDecodeMaxTokens),
          std::min(max_tokens, a4 ? detail::kNvfp4MoeA4MinTokens - 1 : max_tokens), max_tokens}) {
        if (tokens < min_tokens) { continue; }
        required = std::max(required, detail::nvfp4_moe_workspace_bytes(
                                          tokens, detail::resolve_nvfp4_moe_route(tokens, a4)));
    }
    return required;
}

void sparse_moe(const Tensor& x, const SparseMoeNvfp4BankWeights& weights,
                SparseMoeEpilogue epilogue, Tensor& destination, WorkspaceArena& workspace,
                cudaStream_t stream) {
    if (epilogue != SparseMoeEpilogue::Store) {
        throw std::invalid_argument("sparse_moe: the NVFP4-bank profile supports only Store");
    }
    const std::int32_t tokens = require_activation(x, "x");
    if (require_activation(destination, "destination") != tokens) {
        throw std::invalid_argument("sparse_moe: x and destination token counts must match");
    }

    std::vector<AddressRange> ranges;
    ranges.reserve(16);
    ranges.push_back(address_range(x.data, x.bytes(), "x"));
    ranges.push_back(address_range(destination.data, destination.bytes(), "destination"));
    require_bf16(weights.router, kNvfp4MoeExperts, kNvfp4MoeHidden, "router", ranges);
    require_bf16(weights.shared_expert_gate, 1, kNvfp4MoeHidden, "shared_expert_gate", ranges);
    require_bf16(weights.shared_gate, kNvfp4MoeIntermediate, kNvfp4MoeHidden, "shared_gate",
                 ranges);
    require_bf16(weights.shared_up, kNvfp4MoeIntermediate, kNvfp4MoeHidden, "shared_up", ranges);
    require_bf16(weights.shared_down, kNvfp4MoeHidden, kNvfp4MoeIntermediate, "shared_down",
                 ranges);
    require_bank(weights.gate_up, 2 * kNvfp4MoeIntermediate, kNvfp4MoeHidden, "gate_up", ranges);
    require_bank(weights.down, kNvfp4MoeHidden, kNvfp4MoeIntermediate, "down", ranges);

    const bool a4              = allows_bank_a4(weights.gate_up.policy, weights.down.policy);
    const Nvfp4MoeRoute route  = detail::resolve_nvfp4_moe_route(tokens, a4);
    const std::size_t required = detail::nvfp4_moe_workspace_bytes(tokens, route);
    if (workspace.base() == nullptr || workspace.capacity() < required ||
        workspace.used() > workspace.capacity() - required) {
        throw std::invalid_argument("sparse_moe: insufficient workspace capacity");
    }
    ranges.push_back(address_range(workspace.base(), workspace.capacity(), "workspace"));
    require_disjoint(ranges);

    nvtx::ScopedRange range(route == Nvfp4MoeRoute::Decode ? nvtx::Name::SparseMoeDecode
                                                           : nvtx::Name::SparseMoePrefill,
                            nvtx::Category::Moe, static_cast<std::uint64_t>(tokens));
    auto scope = workspace.scope();
    const detail::Nvfp4MoeWorkspace views =
        detail::allocate_nvfp4_moe_workspace(workspace, tokens, route);
    detail::nvfp4_moe_route(x, weights, views, stream);
    if (route == Nvfp4MoeRoute::Decode) {
        detail::nvfp4_moe_decode(x, weights, views, destination, stream);
    } else {
        detail::nvfp4_moe_grouped(x, weights, views, route, destination, stream);
    }
}

} // namespace ninfer::ops
