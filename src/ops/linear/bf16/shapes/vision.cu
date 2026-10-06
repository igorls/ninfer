#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Narrow = Bf16A16MmaSchedule<16, 32, 64, 16, 8, 3, 2, Cache::cg, Cache::cg,
                                  Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using Wide   = Bf16A16MmaSchedule<16, 64, 64, 16, 16, 3, 2, Cache::cg, Cache::cg,
                                  Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;

template <int K>
Bf16Launch vision(std::int32_t tokens) {
    if (tokens <= 32) return launch_bf16_mma<Bf16ScheduleInstance<Narrow, K>>;
    return launch_bf16_mma<Bf16ScheduleInstance<Wide, K>>;
}
} // namespace

Bf16Launch select_bf16_n1152_k1536(std::int32_t t) { return vision<1536>(t); }

Bf16Launch select_bf16_n3456_k1152(std::int32_t t) { return vision<1152>(t); }

Bf16Launch select_bf16_n1152_k1152(std::int32_t t) { return vision<1152>(t); }

Bf16Launch select_bf16_n4304_k1152(std::int32_t t) { return vision<1152>(t); }

Bf16Launch select_bf16_n1152_k4304(std::int32_t t) { return vision<4304>(t); }

Bf16Launch select_bf16_n4608_k4608(std::int32_t t) { return vision<4608>(t); }

Bf16Launch select_bf16_n2560_k4608(std::int32_t t) { return vision<4608>(t); }
} // namespace ninfer::ops::detail
