#pragma once

#include "core/weight_view.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/sparse_moe.h"

#include <optional>
#include <span>
#include <variant>

namespace ninfer::ops {

// One mathematical use of a logical matrix. The model owns the view and its backing.
struct WeightInput {
    const WeightView& weight;
    LinearPolicy policy = LinearPolicy::A16Only;
    std::optional<float> activation_input_divisor;
};

struct SingleProjectionWeight {
    Weight weight;
    LinearPolicy policy = LinearPolicy::A16Only;
};

struct PairedProjectionWeights {
    Weight first, second;
};

using ProjectionWeights = std::variant<SingleProjectionWeight, PairedProjectionWeights>;

// One complete NVFP4 expert bank [E,N,K] (expert_block_scale_k16_m128x4_v1): E code planes, then E
// swizzled K16 block-scale planes, then E FP32 weight divisors. A bank Use stores no activation
// divisor: its A4 route quantizes each activation row with a dynamic scale.
struct Nvfp4ExpertBankWeight {
    const std::byte* codes               = nullptr;
    const std::byte* scales              = nullptr;
    const float* weight_scale_divisors   = nullptr;
    std::int32_t experts                 = 0;
    std::int32_t n                       = 0;
    std::int32_t k                       = 0;
    std::uint64_t code_bytes_per_expert  = 0;
    std::uint64_t scale_bytes_per_expert = 0;
    LinearPolicy policy                  = LinearPolicy::A16Only;
};

// Prepare the existing native forms; no device allocation, upload, execution or graph rewrite.
// Runtime shape/phase choices and scratch remain with the actual calling Op.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(const WeightInput& input);
// Row order is supplied by the calling implementation, for example a Vision Q/K/V bank.
[[nodiscard]] SingleProjectionWeight prepare_linear_weight(std::span<const WeightInput> rows);
[[nodiscard]] SingleProjectionWeight prepare_attn_input_proj_weights(const WeightInput& query,
                                                                     const WeightInput& key,
                                                                     const WeightInput& value);
[[nodiscard]] ProjectionWeights prepare_attn_input_proj_weights(const WeightInput& query,
                                                                const WeightInput& key,
                                                                const WeightInput& gate,
                                                                const WeightInput& value);
[[nodiscard]] ProjectionWeights prepare_gdn_input_proj_weights(const WeightInput& query,
                                                               const WeightInput& key,
                                                               const WeightInput& value,
                                                               const WeightInput& z);
[[nodiscard]] ProjectionWeights prepare_gdn_gating_proj_weights(const WeightInput& a,
                                                                const WeightInput& b);
[[nodiscard]] SingleProjectionWeight prepare_linear_swiglu_weight(const WeightInput& gate,
                                                                  const WeightInput& up);
[[nodiscard]] Nvfp4ExpertBankWeight prepare_nvfp4_expert_bank_weight(const WeightInput& input);
[[nodiscard]] SparseMoeWeights
prepare_sparse_moe_weights(const WeightInput& router, const WeightInput& shared_score,
                           std::span<const WeightInput> expert_gate_up,
                           std::span<const WeightInput> expert_down, const WeightInput& shared_gate,
                           const WeightInput& shared_up, const WeightInput& shared_down);

} // namespace ninfer::ops
