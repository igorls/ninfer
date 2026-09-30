#include "ops/rmsnorm_rope/launch.h"

#include "core/device.h"
#include "ops/rmsnorm_rope/gated_d256.cuh"
#include "ops/rmsnorm_rope/kernel.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <bool Pair>
void launch_fixed(const Tensor& positions, const Tensor* q_norm_weight, const Tensor& k_norm_weight,
                  Tensor* q, Tensor& k, std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(tokens, Pair ? 5 : 1);
    rmsnorm_rope_d128_kernel<Pair><<<grid, 256, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data),
        q_norm_weight == nullptr ? nullptr : static_cast<const __nv_bfloat16*>(q_norm_weight->data),
        static_cast<const __nv_bfloat16*>(k_norm_weight.data),
        q == nullptr ? nullptr : static_cast<__nv_bfloat16*>(q->data),
        static_cast<__nv_bfloat16*>(k.data));
}

} // namespace

void rmsnorm_rope_pair_launch(const Tensor& positions, const Tensor& q_norm_weight,
                              const Tensor& k_norm_weight, Tensor& q, Tensor& k,
                              std::int32_t tokens, cudaStream_t stream) {
    // Five independent head groups per token expose parallelism even at short block widths.
    launch_fixed<true>(positions, &q_norm_weight, k_norm_weight, &q, k, tokens, stream);
    CUDA_CHECK(cudaGetLastError());
}

void rmsnorm_rope_single_launch(const Tensor& positions, const Tensor& norm_weight, Tensor& x,
                                std::int32_t tokens, cudaStream_t stream) {
    // One warp owns each K head while the CTA shares one coefficient table.
    launch_fixed<false>(positions, nullptr, norm_weight, nullptr, x, tokens, stream);
    CUDA_CHECK(cudaGetLastError());
}

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
