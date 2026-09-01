#include "targets/qwen3_8_flash_next/impl/mtp_forward_kernels.h"
#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

__global__ void mtp_stem_combine_and_repeat_kernel(
    const __nv_bfloat16* __restrict__ emb_proj,
    const __nv_bfloat16* __restrict__ hid_proj,
    __nv_bfloat16* __restrict__ trunk_sum,
    __nv_bfloat16* __restrict__ mtp_hyper_hidden,
    int batch) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = 2560 * batch;
    if (idx >= total) { return; }
    const int token = idx / 2560;
    const int col = idx % 2560;

    const float e = __bfloat162float(emb_proj[idx]);
    const float h = __bfloat162float(hid_proj[idx]);
    const float sum = e + h;
    const __nv_bfloat16 sum_bf16 = __float2bfloat16_rn(sum);

    if (trunk_sum != nullptr) {
        trunk_sum[idx] = sum_bf16;
    }

    #pragma unroll
    for (int stream_idx = 0; stream_idx < 4; ++stream_idx) {
        mtp_hyper_hidden[static_cast<std::int64_t>(token) * 10240 + stream_idx * 2560 + col] = sum_bf16;
    }
}

} // namespace

void flash_next_mtp_stem_combine_and_repeat_launch(const Tensor& emb_proj, const Tensor& hid_proj,
                                                  Tensor* trunk_sum, Tensor& mtp_hyper_hidden,
                                                  cudaStream_t stream) {
    const int batch = static_cast<int>(emb_proj.ne[1]);
    const int total = 2560 * batch;
    const int block = 256;
    const int grid = (total + block - 1) / block;

    mtp_stem_combine_and_repeat_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(emb_proj.data),
        static_cast<const __nv_bfloat16*>(hid_proj.data),
        trunk_sum != nullptr ? static_cast<__nv_bfloat16*>(trunk_sum->data) : nullptr,
        static_cast<__nv_bfloat16*>(mtp_hyper_hidden.data),
        batch);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
