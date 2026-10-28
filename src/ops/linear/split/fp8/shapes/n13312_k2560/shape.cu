#include "ops/linear/split/dispatch.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/kernels/fp8/fp8_instances.cuh"
#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail::linear_split {
namespace {

constexpr int kInputRows = 2560;
using Scale = float;
using Fp8SplitOutput = LinearBf16SegmentedOutput<6144,6144,512,512>;

using Gemv = Fp8A16GemvSchedule<4, 4, 16, 4, Fp8CodeCache::Default, 1, 1>;
using Decode = Fp8A16SimtSchedule<4, 4, 16, 4, 4,
    Fp8SimtActivationAccess::SharedPhase, Fp8CodeCache::Default, 1,
    Fp8SimtBlockOrder::RowsContiguous, 1>;
using Tma64x128  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma192x128 = Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>;
// No tail split-K: which tiles form the final wave depends on the token count, so splitting it
// would make a token's K reduction order depend on how the prompt is chunked for prefill.
using MidBulk = Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>;
using Bulk    = Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>;

Fp8SplitOutput qsa_output(std::span<const Tensor> out) {
    return {{static_cast<__nv_bfloat16*>(out[0].data),static_cast<__nv_bfloat16*>(out[1].data),static_cast<__nv_bfloat16*>(out[2].data),static_cast<__nv_bfloat16*>(out[3].data)}};
}

template <class Schedule>
void a16_mma(const Tensor& x, const Weight& weight, std::span<const Tensor> out, cudaStream_t stream) {
    static_assert(6144 % Schedule::kBlockRows == 0 && 512 % Schedule::kBlockRows == 0);
    launch_fp8_a16_mma<Fp8ScheduleInstance<Schedule, kInputRows>>(
        fp8_a16_operands<Scale>(x, weight), qsa_output(out), LinearIdentityEpilogue{}, stream);
}

void launch_a16(const Tensor& x, const Weight& weight, std::span<const Tensor> out,
                cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) {
        launch_fp8_a16_gemv<Fp8ScheduleInstance<Gemv, kInputRows>>(
            fp8_a16_operands<Scale>(x, weight), qsa_output(out), LinearIdentityEpilogue{},
            stream);
        return;
    }
    if (tokens <= 48) {
        // Keep GEMV's K partition and accumulator chains across target windows.
        launch_fp8_a16_simt<Fp8ScheduleInstance<Decode, kInputRows>>(
            fp8_a16_operands<Scale>(x, weight), qsa_output(out), LinearIdentityEpilogue{}, stream);
        return;
    }
    if (tokens <= 64)
        return a16_mma<Fp8A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>>(x, weight, out, stream);
    if (tokens <= 96)
        return a16_mma<Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(x, weight, out, stream);
    a16_mma<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out, stream);
}

void launch_a8(const Tensor& x, const Weight& weight, std::span<const Tensor> out,
               Fp8A8Workspace workspace, cudaStream_t stream) {
    // The narrow K/V heads fail the public-input A8 numerical criterion at real
    // prefill widths. Preserve BF16 activations for those heads. Q/G retain A8;
    // both contractions read their original stored FP8 coefficients directly.
    auto kv = fp8_a16_operands<Scale>(x, weight);
    kv.rows = 1024;
    kv.codes += std::size_t(12288) * kInputRows;
    kv.scales += 12288;
    const LinearBf16SegmentedOutput<512, 512> kv_output{
        {static_cast<__nv_bfloat16*>(out[2].data), static_cast<__nv_bfloat16*>(out[3].data)}};
    using KvSchedule = Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>;
    launch_fp8_a16_mma<Fp8ScheduleInstance<KvSchedule, kInputRows>>(
        kv, kv_output, LinearIdentityEpilogue{}, stream);
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    const Fp8SplitOutput output = qsa_output(out);
    auto operands = fp8_a8_operands<Scale>(weight, workspace, x.ne[1]);
    operands.rows = 12288;
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
    // The narrower row tile stays at the 512-token anchor, where it fills more of the GPU.
    if (x.ne[1] > 384 && x.ne[1] <= 512) return launch.template operator()<MidBulk>();
    launch.template operator()<Bulk>();
}

std::size_t capacity(LinearPolicy policy, std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first || !valid_linear_policy(policy))
        throw std::invalid_argument("linear_split: invalid FP32-scale capacity query");
    return allows_a8(policy) && last > 48 ? fp8_a8_workspace_capacity_bytes(last, 2560, 0) : 0;
}
void launch(const Tensor& x, Weights weights, std::span<const Tensor> outputs,
            MemoryArena* workspace, cudaStream_t stream) {
    if (!allows_a8(weights.policy) || x.ne[1] <= 48)
        return launch_a16(x, weights.first, outputs, stream);
    if (!workspace) throw std::logic_error("linear_split: missing FP32-scale A8 scratch");
    auto scope = workspace->scope();
    auto scratch = allocate_fp8_a8_workspace(*workspace, x.ne[1], 2560, 0);
    launch_a8(x, weights.first, outputs, scratch, stream);
}
} // namespace
const SplitShape kFp8F32SplitN13312K2560{13312, 2560, kSplitRows<6144,6144,512,512>, launch, capacity};
} // namespace ninfer::ops::detail::linear_split
