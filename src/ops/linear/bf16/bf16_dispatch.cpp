#include "ops/linear/bf16/bf16_dispatch.h"

#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_launch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

Bf16Launch select_bf16_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    const bool qsa_indexer       = n == 640 && k == 2560;
    const bool ple_key           = n == 10240 && k == 2560;
    const bool ple_value         = n == 2560 && k == 2560;
    const bool output_head       = n == 248320 && k == 2560;
    const bool flash_next_text   = qsa_indexer || ple_key || ple_value || output_head;
    const bool vision_patch      = n == 1152 && k == 1536;
    const bool vision_qkv        = n == 3456 && k == 1152;
    const bool vision_proj       = n == 1152 && k == 1152;
    const bool vision_fc1        = n == 4304 && k == 1152;
    const bool vision_fc2        = n == 1152 && k == 4304;
    const bool vision_merger_fc1 = n == 4608 && k == 4608;
    const bool vision_merger_fc2 = n == 2560 && k == 4608;
    const bool vision_raw = vision_patch || vision_qkv || vision_proj || vision_fc1 || vision_fc2;
    const bool vision_merged     = vision_merger_fc1 || vision_merger_fc2;
    const bool flash_next_vision = vision_raw || vision_merged;
    const bool supported_problem = (n == 14336 && k == 5120) || (n == 5120 && k == 6144) ||
                                   flash_next_text || flash_next_vision;
    if (!supported_problem || t <= 0) {
        throw std::invalid_argument("bf16 linear: unsupported shape or T");
    }
    if (flash_next_text && t > 8) {
        throw std::invalid_argument("bf16 linear: Flash-Next target requires T in [1,8]");
    }
    if (vision_raw && (t > 131072 || (t % 4) != 0)) {
        throw std::invalid_argument(
            "bf16 linear: Vision raw-patch problems require T in {4,8,...,131072}");
    }
    if (vision_merged && t > 32768) {
        throw std::invalid_argument(
            "bf16 linear: Vision merged-token problems require T in [1,32768]");
    }
    if (flash_next_vision) { return launch_bf16_mma; }
    if (t == 1) { return launch_bf16_decode; }
    const std::int32_t small_t_end =
        n == 5120 ? kBf16SmallTMaxTokens : kBf16LinearSmallTDispatchEnd;
    if (t <= small_t_end) { return launch_bf16_small_t; }
    return launch_bf16_mma;
}

Bf16Launch select_bf16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
        return select_bf16_a16_launch(n, k, t);
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("bf16 linear: unsupported policy");
}

void bf16_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   cudaStream_t stream) {
    const Bf16Launch launch = select_bf16_launch(weight.n, weight.k, x.ne[1], policy);
    launch(x, weight, out, stream);
}

} // namespace ninfer::ops::detail
