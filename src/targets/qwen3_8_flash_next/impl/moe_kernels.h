#pragma once

#include "targets/qwen3_8_flash_next/impl/model_view.h"
#include "targets/qwen3_8_flash_next/impl/moe_workspace.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::targets::qwen3_8_flash_next::detail {

void flash_next_moe_kernels_launch(const Tensor& input, const MoeWeights& weights,
                                   const FlashNextMoeWorkspace& workspace, Tensor& output,
                                   cudaStream_t stream);

// Decode-arm (tokens <= 8) down-projection kernels. Both produce bitwise-identical BF16 output;
// they differ only in how the eleven per-row partials are spread over warps and CTAs.
//   Legacy   - (320, T) CTAs x 8 warps, one warp per row walking the paths serially.
//   PathWarp - (2560, T) CTAs x 11 warps, one warp per path, thread 0 combines.
enum class FlashNextMoeDownKernel : int { Legacy = 0, PathWarp = 1 };

// The kernel flash_next_moe_kernels_launch uses on the decode arm for `tokens` (1..8): PathWarp
// at T=1, where Legacy's 320 CTAs are 0.28 waves; Legacy from T=2, where its T x 320 CTAs fill
// the machine and PathWarp's 2,560 x T 11-warp CTAs cost 8-20% more from T=5 up.
FlashNextMoeDownKernel flash_next_moe_down_kernel_for(int tokens);

// Runs the decode-arm down projection with an explicit kernel choice on an already routed
// workspace (ids, alpha, shared_scale, activations[640, 11, T]) into output [2560, T]. This is
// the hook the bitwise gate in test_moe.cpp uses to run both kernels on identical inputs; the
// production launcher calls it with flash_next_moe_down_kernel_for(tokens). Throws
// std::invalid_argument unless 1 <= tokens <= 8.
void flash_next_moe_down_launch(FlashNextMoeDownKernel kernel, const MoeWeights& weights,
                                const FlashNextMoeWorkspace& workspace, int tokens,
                                Tensor& output, cudaStream_t stream);

// Compiled/launch attributes of a decode-arm down kernel, for the occupancy report in the test
// (PathWarp is designed for 4 resident CTAs per SM at 352 threads x 40 registers, no spills).
struct FlashNextMoeDownKernelAttributes {
    int threads_per_block         = 0;
    int registers_per_thread      = 0;
    std::size_t local_bytes       = 0; // per thread; nonzero means ptxas spilled
    std::size_t static_smem_bytes = 0;
    int max_blocks_per_sm         = 0; // cudaOccupancyMaxActiveBlocksPerMultiprocessor
};
FlashNextMoeDownKernelAttributes
flash_next_moe_down_kernel_attributes(FlashNextMoeDownKernel kernel);

void flash_next_moe_bf16_kernels_launch(const Tensor& input, const MoeBf16Weights& weights,
                                        const FlashNextMoeWorkspace& workspace, Tensor& output,
                                        cudaStream_t stream);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
