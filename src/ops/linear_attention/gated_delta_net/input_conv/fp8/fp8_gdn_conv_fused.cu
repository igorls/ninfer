#include "ops/linear_attention/gated_delta_net/input_conv/fp8/fp8_gdn_conv_fused.cuh"
#include "ops/linear_attention/gated_delta_net/input_conv/fp8/fp8_gdn_input_plan.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

void fp8_gdn_snapshot_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                   const ConstStateFieldView& conv_states, const Tensor& valid_columns,
                                   const gdn_input::HistoryDestination& destination,
                                   Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                   cudaStream_t stream) {
    fp8_gdn_snapshot_fused<5120, __nv_bfloat16>(x, weight, conv_weight, conv_states, valid_columns,
                                                destination, query, key,
                                                value, z, stream);
}

void fp8_gdn_record_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 const ConstStateFieldView& conv_states, const Tensor& valid_columns,
                                 Tensor& conv_record, Tensor& query,
                                 Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    fp8_gdn_record_fused<5120, __nv_bfloat16>(x, weight, conv_weight, conv_states, valid_columns,
                                              conv_record, query, key, value, z,
                                              stream);
}

} // namespace ninfer::ops::detail
