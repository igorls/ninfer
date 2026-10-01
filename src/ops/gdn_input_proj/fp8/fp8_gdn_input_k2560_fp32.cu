// FP8_E4M3FN_ROW_FP32 [16384,2560] q/k/value/z parent (Qwen3.8-Flash-Next GDN input). Every
// kernel instance of this K and row-scale word lives in this one translation unit. The contraction
// schedules are those selected for linear [16384,2560]; the fused B=1 convolution routes use the
// K=5120 fused schedules, whose K tiles divide 2560.

#include "ops/gdn_input_proj/fp8/fp8_gdn_conv_fused.cuh"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/linear/fp8/fp8_instances.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

constexpr int kInputRows = 2560;
using Scale              = float;

using Gemv = Fp8A16GemvSchedule<4, 4, 16, 4, Fp8CodeCache::Default, 1, 1>;
using Sliced16 =
    Fp8A16SlicedKMmaSchedule<4, 16, 7, Cache::ca, Cache::cg, Fp8ActivationStage::PaddedZero, 1>;
using Tma64x128  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma192x128 = Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>;
using MidBulk = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>, 170, 4, 8>;
using Bulk    = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

Fp8GdnInputOutput gdn_output(Tensor& qkv, Tensor& z) {
    return {static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
}

template <class Schedule>
void a16_sliced(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                cudaStream_t stream) {
    launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, kInputRows>>(
        fp8_a16_operands<Scale>(x, weight), gdn_output(qkv, z), LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void a16_mma(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z, cudaStream_t stream) {
    static_assert(10240 % Schedule::kBlockRows == 0 && 6144 % Schedule::kBlockRows == 0);
    launch_fp8_a16_mma<Fp8ScheduleInstance<Schedule, kInputRows>>(
        fp8_a16_operands<Scale>(x, weight), gdn_output(qkv, z), LinearIdentityEpilogue{}, stream);
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) {
        launch_fp8_a16_gemv<Fp8ScheduleInstance<Gemv, kInputRows>>(
            fp8_a16_operands<Scale>(x, weight), gdn_output(qkv, z), LinearIdentityEpilogue{},
            stream);
        return;
    }
    if (tokens <= 16) return a16_sliced<Sliced16>(x, weight, qkv, z, stream);
    if (tokens <= 24) return a16_sliced<Fp8SlicedInstance<32, 8, 2>>(x, weight, qkv, z, stream);
    if (tokens <= 32) return a16_sliced<Fp8SlicedInstance<32, 4, 1>>(x, weight, qkv, z, stream);
    if (tokens <= 64)
        return a16_mma<Fp8A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>(x, weight, qkv, z, stream);
    if (tokens <= 96)
        return a16_mma<Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(x, weight, qkv, z, stream);
    a16_mma<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, qkv, z, stream);
}

void launch_a8(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
               Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const Fp8GdnInputOutput output = gdn_output(qkv, z);
    const auto operands            = fp8_a8_operands<Scale>(weight, workspace, x.ne[1]);
    const auto launch              = [&]<class Schedule>() {
        using S = Fp8ScheduleInstance<Schedule, kInputRows>;
        if constexpr (S::kTmaSwizzle)
            launch_fp8_a8_tma_mma<S>(operands, output, LinearIdentityEpilogue{}, stream,
                                     workspace.partials);
        else
            launch_fp8_a8_mma<S>(operands, output, LinearIdentityEpilogue{}, stream);
    };
    if (x.ne[1] <= 32) return launch.template operator()<Fp8A8T32R32K128>();
    if (x.ne[1] <= 64) return launch.template operator()<Fp8A8T64R128K256>();
    if (x.ne[1] <= 128) return launch.template operator()<Tma64x128>();
    if (x.ne[1] <= 192) return launch.template operator()<Tma192x128>();
    // Smaller output tiles leave only two full-K tiles to split near the 512-token anchor.
    if (x.ne[1] > 384 && x.ne[1] <= 512) return launch.template operator()<MidBulk>();
    launch.template operator()<Bulk>();
}

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    return max_tokens > 256 ? Bulk::kPartialBytes : 0;
}

void snapshot_fused(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
                    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
                    Tensor& z, cudaStream_t stream) {
    fp8_gdn_snapshot_fused<kInputRows, Scale>(x, weight, conv_weight, conv_states, valid_columns,
                                              initial_slot, snapshot_base_slot, query, key, value,
                                              z, stream);
}

void record_fused(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                  const Tensor& conv_states, const Tensor& valid_columns,
                  const Tensor& initial_slot, Tensor& conv_record, Tensor& query, Tensor& key,
                  Tensor& value, Tensor& z, cudaStream_t stream) {
    fp8_gdn_record_fused<kInputRows, Scale>(x, weight, conv_weight, conv_states, valid_columns,
                                            initial_slot, conv_record, query, key, value, z,
                                            stream);
}

} // namespace

const Fp8GdnInputProfile kFp8GdnInputFp32K2560{
    QType::FP8_E4M3FN_ROW_FP32, kInputRows,     launch_a16,  launch_a8,
    partial_capacity_bytes,     snapshot_fused, record_fused};

} // namespace ninfer::ops::detail
