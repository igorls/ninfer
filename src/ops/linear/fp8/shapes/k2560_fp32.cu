#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

// FP8_E4M3FN_ROW_FP32 K=2560 problems: [13312,2560] (Flash-Next gated attention input) and
// [16384,2560] (GDN [q,k,v,z] input). Kernel instances depend on K, the token tile and the scale
// word, not on N, so both problems live in one translation unit to register each instance once.
namespace ninfer::ops::detail {

namespace n13312 {
using Geometry  = Fp8Geometry<13312, 2560>;
using Scale     = float;
using Gemv      = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using Tma64x128 = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma64x256 = Fp8A8TmaMmaSchedule<64, 256, 128, 2, 4, 2, 1>;
using Tma96x256 = Fp8A8TmaMmaSchedule<96, 256, 128, 3, 4, 2, 1>;
// No tail split-K: which tiles form the final wave depends on the token count, so splitting it
// would make a token's K reduction order depend on how the prompt is chunked for prefill.
using Bulk      = Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv, Scale>(x, weight, out, stream);
    if (tokens <= 48) {
        using Tile = Fp8A16SimtSchedule<4, 2, 8, 8, 4, Fp8SimtActivationAccess::SharedPhase,
            Fp8CodeCache::Default, 2, Fp8SimtBlockOrder::RowsContiguous, 2>;
        return fp8_linear_a16_simt<Geometry, 0, Tile, false, Scale>(x, weight, out, stream);
    }
    if (tokens <= 64)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>, Scale>(
            x, weight, out, stream);
    if (tokens <= 128)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>, Scale>(
            x, weight, out, stream);
    fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, Scale>(
        x, weight, out, stream);
}

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    if (x.ne[1] <= 32)
        return launch_fp8_a8<Geometry, Fp8A8T32R32K128, Scale>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 96)
        return launch_fp8_a8<Geometry, Fp8A8T32R128K128, Scale>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 128)
        return launch_fp8_a8_tma<Geometry, Tma64x128, Scale>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 192)
        return launch_fp8_a8_tma<Geometry, Tma64x256, Scale>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 288)
        return launch_fp8_a8_tma<Geometry, Tma96x256, Scale>(x, weight, out, scratch, stream);
    launch_fp8_a8_tma<Geometry, Bulk, Scale>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t) { return 0; }
} // namespace



namespace n16384 {
using Geometry = Fp8Geometry<16384, 2560>;
using Scale    = float;
using Gemv     = Fp8A16GemvSchedule<4, 4, 16, 4, Fp8CodeCache::Default, 1, 1>;
// Seven resident CTAs keep the 1,024 row tiles in one wave without register spills.
using Sliced16 =
    Fp8A16SlicedKMmaSchedule<4, 16, 7, Cache::ca, Cache::cg, Fp8ActivationStage::PaddedZero, 1>;
using Tma64x128  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma192x128 = Fp8A8TmaMmaSchedule<192, 128, 128, 3, 4, 2, 1>;
// No tail split-K: which tiles form the final wave depends on the token count, so splitting it
// would make a token's K reduction order depend on how the prompt is chunked for prefill.
using MidBulk = Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>;
using Bulk    = Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv, Scale>(x, weight, out, stream);
    if (tokens <= 16) return fp8_linear_a16_sliced_k<Geometry, Sliced16, Scale>(x, weight, out, stream);
    if (tokens <= 24)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 8, 2>, Scale>(x, weight, out,
                                                                              stream);
    if (tokens <= 32)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 1>, Scale>(x, weight, out,
                                                                              stream);
    if (tokens <= 64)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<32, 64, 128, 32, 16, 2, 2>, Scale>(
            x, weight, out, stream);
    if (tokens <= 96)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>, Scale>(
            x, weight, out, stream);
    fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, Scale>(x, weight, out,
                                                                               stream);
}

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    if (x.ne[1] <= 32)
        return launch_fp8_a8<Geometry, Fp8A8T32R32K128, Scale>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 64)
        return launch_fp8_a8<Geometry, Fp8A8T64R128K256, Scale>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 128)
        return launch_fp8_a8_tma<Geometry, Tma64x128, Scale>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 192)
        return launch_fp8_a8_tma<Geometry, Tma192x128, Scale>(x, weight, out, scratch, stream);
    // The narrower row tile stays at the 512-token anchor, where it fills more of the GPU.
    if (x.ne[1] > 384 && x.ne[1] <= 512)
        return launch_fp8_a8_tma<Geometry, MidBulk, Scale>(x, weight, out, scratch, stream);
    launch_fp8_a8_tma<Geometry, Bulk, Scale>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t) { return 0; }
} // namespace



const Fp8LinearShape kFp8F32N13312K2560{
    13312, 2560, n13312::launch_a16, n13312::launch_a8, n13312::uses_a8,
    n13312::partial_capacity_bytes, QType::FP8_E4M3FN_ROW_FP32};
const Fp8LinearShape kFp8F32N16384K2560{
    16384, 2560, n16384::launch_a16, n16384::launch_a8, n16384::uses_a8,
    n16384::partial_capacity_bytes, QType::FP8_E4M3FN_ROW_FP32};
} // namespace ninfer::ops::detail
