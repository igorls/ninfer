#include "ops/linear/bf16/bf16_shapes.h"

// The BF16 [2560,6144] projection (Flash-Next MTP attention output). K=6144 schedules are
// instantiated once, for [5120,6144]; every tile there divides 2560 rows.
namespace ninfer::ops::detail {

Bf16Launch select_bf16_n2560_k6144(std::int32_t tokens) {
    return select_bf16_n5120_k6144(tokens);
}

} // namespace ninfer::ops::detail
