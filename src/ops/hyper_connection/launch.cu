#include "ops/hyper_connection/launch.h"

#include "core/device.h"
#include "ops/hyper_connection/kernel.cuh"
#include "ops/linear/bf16/bf16_split_k_launch.cuh"
#include "ops/linear/bf16/bf16_template_launch.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Wide-T down projection [320,10240]: a 32x32 tile gives only 10 row tiles, so split-K over
// four K quarters fills the GPU at prefill widths. The FP32 partials are reduced in fixed split
// order and the reduction applies silu(x/4) before the BF16 store.
using DownSchedule = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<32, 32, 256, 16, 16, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    kHyperConcat>;
// Wide-T up projection [10240,320].
using UpSchedule = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<64, 64, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    kHyperLowRank>;

struct HyperLowRankEpilogue {
    __device__ __forceinline__ float apply(int, int, float value) const {
        return silu(value * 0.25F);
    }
};

const __nv_bfloat16* bf16(const void* pointer) {
    return static_cast<const __nv_bfloat16*>(pointer);
}

__nv_bfloat16* bf16(void* pointer) { return static_cast<__nv_bfloat16*>(pointer); }

void launch_decode(const void* hidden, const void* norm, const void* down, const void* up,
                   const void* inject, void* block_input, float* injection,
                   const HyperScratch& scratch, std::int32_t tokens, cudaStream_t stream) {
    hyper_group_norm_decode_kernel<<<dim3(kHyperStreams, tokens), kHyperNormThreads, 0, stream>>>(
        bf16(hidden), bf16(norm), bf16(scratch.normalized));
    CUDA_CHECK(cudaGetLastError());
    const int rows = inject != nullptr ? kHyperLowRank + kHyperStreams : kHyperLowRank;
    hyper_low_rank_decode_kernel<<<dim3(rows, tokens), 256, 0, stream>>>(
        bf16(scratch.normalized), bf16(down), bf16(inject), bf16(scratch.low_rank), injection);
    CUDA_CHECK(cudaGetLastError());
    const auto mix = [&]<int Tokens>() {
        hyper_mix_up_decode_kernel<Tokens><<<kHyperHidden, 128, 0, stream>>>(
            bf16(scratch.normalized), bf16(scratch.low_rank), bf16(up), bf16(block_input));
    };
    switch (tokens) {
    case 1: mix.template operator()<1>(); break;
    case 2: mix.template operator()<2>(); break;
    case 3: mix.template operator()<3>(); break;
    case 4: mix.template operator()<4>(); break;
    case 5: mix.template operator()<5>(); break;
    case 6: mix.template operator()<6>(); break;
    case 7: mix.template operator()<7>(); break;
    case 8: mix.template operator()<8>(); break;
    default: throw std::logic_error("hyper_connection: decode chain expects 1..8 tokens");
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_wide(const void* hidden, const void* norm, const void* down, const void* up,
                 const void* inject, void* block_input, float* injection,
                 const HyperScratch& scratch, std::int32_t tokens, cudaStream_t stream) {
    hyper_group_norm_wide_kernel<<<tokens, 256, 0, stream>>>(bf16(hidden), bf16(norm),
                                                              bf16(scratch.normalized));
    CUDA_CHECK(cudaGetLastError());
    launch_bf16_a16_split_k_mma<DownSchedule, kHyperDownSplits>(
        Bf16A16Operands{bf16(scratch.normalized), bf16(down), kHyperLowRank, kHyperConcat, tokens},
        LinearBf16Output{bf16(scratch.low_rank), kHyperLowRank}, HyperLowRankEpilogue{},
        Bf16SplitKWorkspace{scratch.partials, scratch.partial_bytes}, stream);
    if (inject != nullptr) {
        hyper_injection_wide_kernel<<<tokens, 128, 0, stream>>>(bf16(scratch.normalized),
                                                                 bf16(inject), injection);
        CUDA_CHECK(cudaGetLastError());
    }
    launch_bf16_a16_mma<UpSchedule>(
        Bf16A16Operands{bf16(scratch.low_rank), bf16(up), kHyperConcat, kHyperLowRank, tokens},
        LinearBf16Output{bf16(scratch.up), kHyperConcat}, LinearIdentityEpilogue{}, stream);
    hyper_mix_reduce_wide_kernel<<<tokens, 256, 0, stream>>>(
        bf16(scratch.up), bf16(scratch.normalized), bf16(block_input));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void hyper_mix_launch(const void* hidden, const void* norm, const void* down, const void* up,
                      const void* inject, void* block_input, float* injection,
                      const HyperScratch& scratch, std::int32_t tokens, cudaStream_t stream) {
    if (tokens <= kHyperDecodeTokens) {
        launch_decode(hidden, norm, down, up, inject, block_input, injection, scratch, tokens,
                      stream);
    } else {
        launch_wide(hidden, norm, down, up, inject, block_input, injection, scratch, tokens,
                    stream);
    }
}

void hyper_inject_launch(const void* block_output, const float* injection, void* hidden,
                         std::int32_t tokens, cudaStream_t stream) {
    hyper_inject_kernel<<<static_cast<unsigned>(tokens) * kHyperInjectBlocksPerToken, 256, 0,
                          stream>>>(
        bf16(block_output), injection, bf16(hidden));
    CUDA_CHECK(cudaGetLastError());
}

void hyper_expand_launch(const void* x, void* hidden, std::int32_t tokens, cudaStream_t stream) {
    const std::int64_t chunks = static_cast<std::int64_t>(tokens) * kHyperHiddenChunks;
    const auto blocks =
        static_cast<unsigned>((chunks + kHyperExpandThreads - 1) / kHyperExpandThreads);
    hyper_expand_kernel<<<blocks, kHyperExpandThreads, 0, stream>>>(
        static_cast<const ulonglong2*>(x), static_cast<ulonglong2*>(hidden), chunks);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
