#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using S4 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 32, 128, 32, 16, 2, 1>, 2560>;
using S5 = Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 48, 128, 32, 16, 2, 1>, 2560>;
using S6 = Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>;
} // namespace

Bf16Launch select_bf16_n248320_k2560(std::int32_t tokens) {
    // Preserve the target head's single-column reduction through every native MTP width.
    if (tokens <= 48) return select_bf16_n10240_k2560(tokens);
    if (tokens <= 64) return launch_bf16_tma_mma<S4>;
    if (tokens <= 96) return launch_bf16_tma_mma<S5>;
    return launch_bf16_tma_mma<S6>;
}
} // namespace ninfer::ops::detail
