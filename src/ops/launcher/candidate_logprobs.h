#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Wrapper-validated tensors only. candidate_ids/candidates_out and allowed may be null.
void candidate_logprobs_launch(const Tensor& logits, std::int32_t valid_rows,
                               const Tensor& sampled_ids, const Tensor* candidate_ids,
                               const Tensor* allowed, Tensor& sampled_out, Tensor* candidates_out,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail
