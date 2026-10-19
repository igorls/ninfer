// Implements: include/ninfer/ops/candidate_logprobs.h
// Match: wrapper-validated contiguous tensors and valid vocabulary rows.
// Algorithm assumptions: one independent CTA per column; no global workspace.
#include "ops/reduction/candidate_logprobs_launch.h"

#include "core/device.h"
#include "ops/reduction/candidate_logprobs_kernel.cuh"

namespace ninfer::ops::detail {

void candidate_logprobs_launch(const Tensor& logits, std::int32_t valid_rows,
                               const Tensor& sampled_ids, const Tensor* candidate_ids,
                               const Tensor* allowed, Tensor& sampled_out, Tensor* candidates_out,
                               Tensor* top_ids_out, Tensor* top_out, cudaStream_t stream) {
    const std::int32_t columns    = logits.ne[1];
    const std::int32_t candidates = candidate_ids != nullptr ? candidate_ids->ne[0] : 0;
    const std::int32_t top_k      = top_ids_out != nullptr ? top_ids_out->ne[1] : 0;
    const auto launch             = [&]<bool HasTopK>() {
        candidate_logprobs_kernel<kCandidateLogprobsBlock, HasTopK>
            <<<static_cast<unsigned int>(columns), kCandidateLogprobsBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(logits.data), valid_rows, logits.ne[0],
                static_cast<const std::int32_t*>(sampled_ids.data),
                candidate_ids != nullptr ? static_cast<const std::int32_t*>(candidate_ids->data)
                                         : nullptr,
                candidates,
                allowed != nullptr ? static_cast<const std::int32_t*>(allowed->data) : nullptr,
                allowed != nullptr && allowed->ne[1] > 1 ? allowed->ne[0] : 0,
                static_cast<float*>(sampled_out.data),
                candidates_out != nullptr ? static_cast<float*>(candidates_out->data) : nullptr,
                top_ids_out != nullptr ? static_cast<std::int32_t*>(top_ids_out->data) : nullptr,
                top_out != nullptr ? static_cast<float*>(top_out->data) : nullptr, top_k, columns);
    };
    if (top_k > 0) {
        launch.template operator()<true>();
    } else {
        launch.template operator()<false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
