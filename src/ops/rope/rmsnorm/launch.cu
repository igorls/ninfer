#include "ops/rope/rmsnorm/launch.h"

#include "core/device.h"
#include "ops/rope/rmsnorm/gated_d256.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
void rmsnorm_rope_gated_d256_launch(const Tensor& projected, const Tensor& positions,
                                    const Tensor& q_norm_weight, const Tensor& k_norm_weight,
                                    Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                    std::int32_t tokens, cudaStream_t stream) {
    // One CTA per token: its 26 warps own the 24 query heads (with their gate copies) and the two
    // key heads (with their value copies); the token's 32 rotations are shared in shared memory.
    rmsnorm_rope_gated_d256_kernel<<<tokens, gated_d256::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const __nv_bfloat16*>(q_norm_weight.data),
        static_cast<const __nv_bfloat16*>(k_norm_weight.data), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(v.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
