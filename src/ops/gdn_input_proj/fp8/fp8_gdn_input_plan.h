#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

using Fp8GdnInputA16Launch = void (*)(const Tensor& x, const Weight& weight, Tensor& qkv,
                                      Tensor& z, cudaStream_t stream);
using Fp8GdnInputA8Launch  = void (*)(const Tensor& x, const Weight& weight, Tensor& qkv,
                                     Tensor& z, Fp8A8Workspace workspace, cudaStream_t stream);
using Fp8GdnSnapshotFusedLaunch = void (*)(const Tensor& x, const Weight& weight,
                                           const Tensor& conv_weight, Tensor& conv_states,
                                           const Tensor& valid_columns, const Tensor& initial_slot,
                                           const Tensor& snapshot_base_slot, Tensor& query,
                                           Tensor& key, Tensor& value, Tensor& z,
                                           cudaStream_t stream);
using Fp8GdnRecordFusedLaunch = void (*)(const Tensor& x, const Weight& weight,
                                         const Tensor& conv_weight, const Tensor& conv_states,
                                         const Tensor& valid_columns, const Tensor& initial_slot,
                                         Tensor& conv_record, Tensor& query, Tensor& key,
                                         Tensor& value, Tensor& z, cudaStream_t stream);

// One registered row-scaled FP8 [16384,K] q/k/value/z parent. K and the stored row-scale word fix
// its kernel instances, which one translation unit per profile owns. Route frontiers shared by
// every profile (A8 from T=17, fused convolution through W=3) belong to the plans.
struct Fp8GdnInputProfile {
    QType qtype;
    std::int32_t input_rows;
    Fp8GdnInputA16Launch a16; // every positive T
    Fp8GdnInputA8Launch a8;   // activation quantization + A8 contraction, T >= 17
    std::size_t (*a8_partial_bytes)(std::int32_t max_tokens);
    Fp8GdnSnapshotFusedLaunch snapshot_fused; // B = 1, W = 1..3
    Fp8GdnRecordFusedLaunch record_fused;     // B = 1, W = 2..3
};

// Qwen3.8-27B: FP8_E4M3FN_ROW_BF16 [16384,5120].
extern const Fp8GdnInputProfile kFp8GdnInputBf16K5120;
// Qwen3.8-Flash-Next: FP8_E4M3FN_ROW_FP32 [16384,2560].
extern const Fp8GdnInputProfile kFp8GdnInputFp32K2560;

// The profile registered for the exact parent, or nullptr.
[[nodiscard]] const Fp8GdnInputProfile* find_fp8_gdn_input_profile(QType qtype,
                                                                   std::int32_t parent_rows,
                                                                   std::int32_t input_rows) noexcept;

[[nodiscard]] std::size_t fp8_gdn_input_workspace_capacity_bytes(const Fp8GdnInputProfile& profile,
                                                                 LinearPolicy policy,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens);

// Exact contraction mechanisms shared by G1/G2/G3. Semantic Ops own their route frontier and
// call one of these launchers after resolving their complete-form plan.
void fp8_gdn_input_a8_dispatch(const Fp8GdnInputProfile& profile, const Tensor& x,
                               const Weight& weight, Tensor& qkv, Tensor& z,
                               WorkspaceArena& workspace, cudaStream_t stream);

void fp8_gdn_input_dispatch(const Fp8GdnInputProfile& profile, const Tensor& x,
                            const Weight& weight, Tensor& qkv, Tensor& z, LinearPolicy policy,
                            WorkspaceArena* workspace, cudaStream_t stream);

// K=5120 BF16-scale mechanisms of kFp8GdnInputBf16K5120.
[[nodiscard]] std::size_t fp8_gdn_input_partial_capacity_bytes(std::int32_t max_tokens);

void fp8_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream);

void fp8_gdn_input_matrix_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream);

void fp8_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream);

void fp8_gdn_snapshot_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                   Tensor& conv_states, const Tensor& valid_columns,
                                   const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                   Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                   cudaStream_t stream);

void fp8_gdn_record_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 const Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                                 Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream);

} // namespace ninfer::ops::detail
