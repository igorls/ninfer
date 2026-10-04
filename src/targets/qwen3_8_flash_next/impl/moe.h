#pragma once

#include "core/arena.h"
#include "targets/qwen3_8_flash_next/impl/model_view.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {

[[nodiscard]] std::size_t flash_next_moe_workspace_capacity_bytes(std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

// Exact Qwen4-exp 512-expert/top-10 MoE leaf with top-10 renormalized probabilities
// (norm_topk_prob=true per transformers Qwen4ExpTextTopKRouter); the independent shared expert is sigmoid-gated.
// A cached layer gathers the routed experts into the cache's launch bank before the
// contiguous-bank kernels run. Full residency leaves that path unused.
void flash_next_moe(const Tensor& input, const MoeWeights& weights, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream);

// Copies `device_ids` (length `id_count`) to the host, admits those experts, and
// gathers them through the layer's pointer table into the launch bank. The copy
// synchronizes `stream`, so this must not run while that stream is capturing.
void flash_next_publish_cached_layer(const ExpertLayerCache& layer, const std::int32_t* device_ids,
                                     std::int32_t id_count, cudaStream_t stream);

void flash_next_moe_bf16(const Tensor& input, const MoeBf16Weights& weights, Tensor& output,
                         WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::targets::qwen3_8_flash_next::detail
