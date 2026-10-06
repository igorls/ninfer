#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

// FP8_E4M3FN_ROW_FP32 [2560,6144]: the Flash-Next attention and GDN output projections.
namespace ninfer::ops::detail {
namespace {
using Geometry  = Fp8Geometry<2560, 6144>;
using Scale     = float;
using Gemv      = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using Tma64x128 = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 3, 1>;
using MidBulk   = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 3, 1>, 170, 4, 8>;
using Bulk      = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv, Scale>(x, weight, out, stream);
    if (tokens <= 48) {
        using Tile = Fp8A16SimtSchedule<4, 2, 8, 8, 4, Fp8SimtActivationAccess::SharedPhase,
            Fp8CodeCache::Default, 2, Fp8SimtBlockOrder::RowsContiguous, 2>;
        return fp8_linear_a16_simt<Geometry, 0, Tile, false, Scale>(x, weight, out, stream);
    }
    if (tokens <= 64)
        return fp8_linear_a16_sliced_k<Geometry, Fp8SlicedInstance<32, 4, 1>, Scale>(x, weight, out,
                                                                              stream);
    if (tokens <= 128)
        return fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 64, 128, 32, 16, 2, 2>, Scale>(
            x, weight, out, stream);
    fp8_linear_a16_mma<Geometry, Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, Scale>(x, weight, out,
                                                                               stream);
}

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    // Measured on the G4 for the FP32-scale problem: 32x32 tiles through T=32, 64x64 through 128.
    if (x.ne[1] <= 32)
        return launch_fp8_a8<Geometry, Fp8A8T32R32K128, Scale>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 64)
        return launch_fp8_a8<Geometry, Fp8A8T64R64K128, Scale>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 128)
        return launch_fp8_a8<Geometry, Fp8A8T64R64K128, Scale>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 192)
        return launch_fp8_a8_tma<Geometry, Tma64x128, Scale>(x, weight, out, scratch, stream);
    // The narrower row tile fills the GPU before the large-tile path reaches a full wave.
    if (x.ne[1] <= 768)
        return launch_fp8_a8_tma<Geometry, MidBulk, Scale>(x, weight, out, scratch, stream);
    launch_fp8_a8_tma<Geometry, Bulk, Scale>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 17; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    if (max_tokens > 768) return Bulk::kPartialBytes;
    return max_tokens > 192 ? MidBulk::kPartialBytes : 0;
}

} // namespace

const Fp8LinearShape kFp8F32N2560K6144{2560,    6144,    launch_a16,           launch_a8,
                                       uses_a8, partial_capacity_bytes, QType::FP8_E4M3FN_ROW_FP32};
} // namespace ninfer::ops::detail
