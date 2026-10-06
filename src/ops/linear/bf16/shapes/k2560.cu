#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

// Every BF16 K=2560 problem. Kernel instances depend on K and the token tile, not on N, so all
// routes live in this one translation unit: a second TU instantiating the same schedule would
// register a duplicate device kernel. Wide rows ([10240,2560], [13312,2560] and the [248320,2560]
// vocabulary head) share one route table; the narrow [2560,2560] and [640,2560] problems use
// eight-row GEMV tiles that keep 80-320 CTAs resident at T=1.
namespace ninfer::ops::detail {
namespace {
using Gemv = Bf16A16GemvSchedule<4, 1, 8, 8, 4, Bf16ActivationAccess::Direct,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>;
// PLE and the target readout retain the single-column reduction when a speculative window
// changes the physical column count. Reuse weights across four independent columns,
// preserving GEMV's phase order and four accumulator chains.
using DecodeBatch = Bf16A16SimtSchedule<4, 1, 8, 8, 4, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>;
} // namespace

Bf16Launch select_bf16_n10240_k2560(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Bf16ScheduleInstance<Gemv, 2560>>;
    if (tokens <= 48) return launch_bf16_simt<Bf16ScheduleInstance<DecodeBatch, 2560>>;
    if (tokens <= 64) return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K64S3, 2560>>;
    if (tokens <= 96) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR64T32K64S3, 2560>>;
    if (tokens <= 128)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T128K64S2, 2560>>;
    if (tokens <= 192)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K64S3, 2560>>;
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T128K64S2, 2560>>;
}

Bf16Launch select_bf16_n13312_k2560(std::int32_t tokens) {
    return select_bf16_n10240_k2560(tokens);
}

Bf16Launch select_bf16_n248320_k2560(std::int32_t tokens) {
    return select_bf16_n10240_k2560(tokens);
}
namespace {
using NarrowGemv = Bf16A16GemvSchedule<8, 2, 2, 8, 4, Bf16ActivationAccess::Direct,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 1>;
using NarrowC2 = Bf16A16SimtSchedule<4, 1, 4, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;
using NarrowC4 = Bf16A16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;
using NarrowDecodeBatch = Bf16A16SimtSchedule<4, 2, 2, 8, 4, 4,
                                 Bf16SimtActivationAccess::WarpPacked, Bf16WeightCache::Default,
                                 Bf16PhaseOrder::RowSwizzled, 1, 1, 2, 1>;
} // namespace

Bf16Launch select_bf16_n2560_k2560(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Bf16ScheduleInstance<NarrowGemv, 2560>>;
    if (tokens <= 48) return launch_bf16_simt<Bf16ScheduleInstance<NarrowDecodeBatch, 2560>>;
    if (tokens <= 64) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, 2560>>;
    if (tokens <= 128)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K64S3, 2560>>;
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T128K64S2, 2560>>;
}

Bf16Launch select_bf16_n640_k2560(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Bf16ScheduleInstance<NarrowGemv, 2560>>;
    if (tokens <= 2) return launch_bf16_simt<Bf16ScheduleInstance<NarrowC2, 2560, 2>>;
    if (tokens <= 4) return launch_bf16_simt<Bf16ScheduleInstance<NarrowC4, 2560, 4>>;
    if (tokens <= 64)
        return launch_bf16_sliced_k_mma<
            Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 8>, 2560>>;
    // Measured on the G4 (RTX PRO 6000): 32x32 MMA tiles win through the 640-token chunk, 64x64
    // TMA tiles through 2048 and the 128-token TMA tile at the 4096/8192 prefill anchors.
    if (tokens <= 640) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, 2560>>;
    if (tokens <= 2048)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K64S3, 2560>>;
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T128K64S2, 2560>>;
}
} // namespace ninfer::ops::detail
