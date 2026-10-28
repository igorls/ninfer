#include "ops/gated_residual/mix/launch.h"
#include "ops/common/host/launch.h"
#include "ops/gated_residual/mix/kernel/output.cuh"
#include "ops/gated_residual/mix/kernel/up_mma_mix.cuh"
#include "ops/gated_residual/mix/kernel/up_tma_mix.cuh"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_split_k_launch.cuh"

#include <type_traits>

namespace ninfer::ops::detail::hc::mix {
namespace {
template <Kind Form, class Schedule>
using DownRows = std::conditional_t<Form == Kind::Block, Bf16RowTailSchedule<Schedule>, Schedule>;

template <Kind Form, class Schedule>
using DownSchedule = DownRows<Form, Bf16ScheduleInstance<Schedule, kResidualWidth>>;

template <int Rows, int Tokens, int K, int MinBlocks>
using SplitSchedule =
    Bf16A16MmaSchedule<Rows, Tokens, K, 16, 8, 3, MinBlocks, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;

using DownSimt = Bf16ScheduleInstance<
    Bf16A16SimtSchedule<8, 2, 1, 8, 4, 2, Bf16SimtActivationAccess::DirectStream,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>,
    kResidualWidth>;

template <Kind Form>
void down(Operands o, Down schedule, Scratch scratch, cudaStream_t stream) {
    const Bf16A16Operands p{static_cast<const __nv_bfloat16*>(o.normalized),
                            static_cast<const __nv_bfloat16*>(o.down), down_rows(Form),
                            kResidualWidth, o.columns};
    const DownOutput<Form> output{static_cast<__nv_bfloat16*>(scratch.activation), o.injection};
    const LinearIdentityEpilogue identity;
    const Bf16SplitKWorkspace partials{static_cast<float*>(scratch.partials.data),
                                       scratch.partials.bytes};
    switch (schedule) {
    case Down::Simt:
        launch_bf16_a16_simt<DownSimt>(p, output, identity, stream);
        break;
    case Down::Split8x32:
        launch_bf16_a16_split_k_mma<DownSchedule<Form, SplitSchedule<32, 32, 64, 1>>, 8>(
            p, output, identity, partials, stream);
        break;
    case Down::TmaSplit2:
        launch_bf16_a16_tma_split_k_mma<
            DownSchedule<Form, Bf16A16TmaMmaSchedule<32, 64, 64, 16, 8, 4, 1>>, 2>(
            p, output, identity, partials, stream);
        break;
    case Down::TmaSplit4:
        launch_bf16_a16_tma_split_k_mma<
            DownSchedule<Form, Bf16A16TmaMmaSchedule<32, 64, 64, 16, 8, 4, 1>>, 4>(
            p, output, identity, partials, stream);
        break;
    case Down::Tma96Split4:
        launch_bf16_a16_tma_split_k_mma<
            DownSchedule<Form, Bf16A16TmaMmaSchedule<32, 96, 64, 16, 16, 3, 1>>, 4>(
            p, output, identity, partials, stream);
        break;
    case Down::Tma96Split8:
        launch_bf16_a16_tma_split_k_mma<
            DownSchedule<Form, Bf16A16TmaMmaSchedule<32, 96, 64, 16, 16, 3, 1>>, 8>(
            p, output, identity, partials, stream);
        break;
    }
}

template <int Features, int Tokens>
void folded_up(Operands o, Scratch scratch, cudaStream_t stream) {
    using S               = FoldedUpSchedule<Features, Tokens>;
    constexpr auto kernel = up_mma_mix_kernel<Features, Tokens>;
    const int bytes       = common::prepare_shared<S::kSharedBytes, kernel, true>();
    common::for_each_token_slice(o.columns, Tokens, [&](int offset, int count) {
        const unsigned blocks = (kWidth / Features) * common::div_up(count, Tokens);
        kernel<<<blocks, S::kThreads, bytes, stream>>>(
            o, static_cast<const __nv_bfloat16*>(scratch.activation), offset);
        CUDA_CHECK(cudaGetLastError());
    });
}

template <int Features>
void tma_folded_up(Operands o, Scratch scratch, cudaStream_t stream) {
    using S = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<kStreams * Features, 64, 64, 32, 16, 2, 1>,
                                   kRank>;
    const Bf16A16Operands p{static_cast<const __nv_bfloat16*>(scratch.activation),
                            static_cast<const __nv_bfloat16*>(o.up), kResidualWidth, kRank,
                            o.columns};
    launch_bf16_a16_tma_mma<S, 1, LinearBf16Output, TmaMixEpilogue, StripedTmaWeightAccess>(
        p, {static_cast<__nv_bfloat16*>(o.output), kWidth},
        {static_cast<const __nv_bfloat16*>(o.normalized)}, stream);
}

__global__ void injection_kernel(Operands o) {
    const auto token = std::int64_t(blockIdx.x / kStreams);
    const int stream = blockIdx.x % kStreams, lane = threadIdx.x & 31;
    const auto* x = static_cast<const __nv_bfloat16*>(o.normalized) + token * kResidualWidth;
    const auto* w = static_cast<const __nv_bfloat16*>(o.separate_injection) + stream * kResidualWidth;
    float sum = 0.F;
    for (int k = threadIdx.x; k < kResidualWidth; k += 256)
        sum = fmaf(__bfloat162float(x[k]), __bfloat162float(w[k]), sum);
    for (int delta = 16; delta; delta >>= 1) sum += __shfl_down_sync(0xffffffffU, sum, delta);
    __shared__ float partial[8];
    if (lane == 0) partial[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.F;
#pragma unroll
        for (int warp = 0; warp < 8; ++warp) total += partial[warp];
        o.injection[token * kStreams + stream] = 2.F * common::sigmoid(total * 0.25F);
    }
}

} // namespace

void launch(Kind kind, Operands o, Plan p, Scratch scratch, CudaExecutionView execution) {
    if (kind == Kind::Block)
        down<Kind::Block>(o, p.down, scratch, execution.stream());
    else
        down<Kind::Final>(o, p.down, scratch, execution.stream());
    if (o.separate_injection) {
        injection_kernel<<<o.columns * kStreams, 256, 0, execution.stream()>>>(o);
        CUDA_CHECK(cudaGetLastError());
    }
    switch (p.up) {
    case Up::Folded16x16:
        folded_up<16, 16>(o, scratch, execution.stream());
        break;
    case Up::Folded32x32:
        folded_up<32, 32>(o, scratch, execution.stream());
        break;
    case Up::TmaFolded16x64:
        tma_folded_up<16>(o, scratch, execution.stream());
        break;
    case Up::TmaFolded32x64:
        tma_folded_up<32>(o, scratch, execution.stream());
        break;
    }
}
} // namespace ninfer::ops::detail::hc::mix
