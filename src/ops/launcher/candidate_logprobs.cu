// Implements: include/ninfer/ops/candidate_logprobs.h
// Match: wrapper-validated contiguous tensors and valid vocabulary rows.
// Algorithm assumptions: one independent CTA per column; no global workspace.
#include "ops/launcher/candidate_logprobs.h"

#include "core/device.h"
#include "ops/kernel/candidate_logprobs.cuh"

namespace ninfer::ops::detail {

void candidate_logprobs_launch(const Tensor& logits, std::int32_t valid_rows,
                               const Tensor& sampled_ids, const Tensor* candidate_ids,
                               const Tensor* allowed, Tensor& sampled_out, Tensor* candidates_out,
                               cudaStream_t stream) {
    const std::int32_t columns    = logits.ne[1];
    const std::int32_t candidates = candidate_ids != nullptr ? candidate_ids->ne[0] : 0;
    candidate_logprobs_kernel<kCandidateLogprobsBlock>
        <<<static_cast<unsigned int>(columns), kCandidateLogprobsBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(logits.data), valid_rows, logits.ne[0],
            static_cast<const std::int32_t*>(sampled_ids.data),
            candidate_ids != nullptr ? static_cast<const std::int32_t*>(candidate_ids->data)
                                     : nullptr,
            candidates, allowed != nullptr ? static_cast<const std::int32_t*>(allowed->data) : nullptr,
            static_cast<float*>(sampled_out.data),
            candidates_out != nullptr ? static_cast<float*>(candidates_out->data) : nullptr,
            columns);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
