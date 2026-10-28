#pragma once

#include "ops/moe/kernels/cuda/block_scaled/launch.h"

namespace ninfer::ops::detail::moe_cuda::block_scaled::nvfp4 {

void launch_input(const Tensor& x, const CudaSwiGLUExperts& experts, const Storage& storage,
                  CudaExecutionView execution, bool clear_counts);
void launch_experts(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                    const Plan& plan, const Storage& storage, CudaExecutionView execution);
// Returns true when destination includes every requested term.
bool launch_vector(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, const ScaledSwiGLUTerm* shared, MoeEpilogue mode,
                   Tensor& destination, const Storage& storage, CudaExecutionView execution,
                   bool shared_prepared);
} // namespace ninfer::ops::detail::moe_cuda::block_scaled::nvfp4
