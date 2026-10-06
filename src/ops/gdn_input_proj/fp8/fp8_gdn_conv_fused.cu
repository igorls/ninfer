#include "ops/gdn_input_proj/fp8/fp8_gdn_conv_fused.cuh"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

void fp8_gdn_snapshot_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                   Tensor& conv_states, const Tensor& valid_columns,
                                   const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                   Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                   cudaStream_t stream) {
    fp8_gdn_snapshot_fused<5120, __nv_bfloat16>(x, weight, conv_weight, conv_states, valid_columns,
                                                initial_slot, snapshot_base_slot, query, key,
                                                value, z, stream);
}

void fp8_gdn_record_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 const Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                                 Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    fp8_gdn_record_fused<5120, __nv_bfloat16>(x, weight, conv_weight, conv_states, valid_columns,
                                              initial_slot, conv_record, query, key, value, z,
                                              stream);
}

} // namespace ninfer::ops::detail
