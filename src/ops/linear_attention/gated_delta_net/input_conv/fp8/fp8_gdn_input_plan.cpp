#include "core/weight.h"
#include "ops/linear_attention/gated_delta_net/input_conv/fp8/fp8_gdn_input_plan.h"

#include <cstdint>
#include <initializer_list>

namespace ninfer::ops::detail {
namespace {

void a16_k5120(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
               cudaStream_t stream) {
    if (x.ne[1] == 1) {
        fp8_gdn_input_decode_launch(x, weight, qkv, z, stream);
    } else {
        fp8_gdn_input_matrix_launch(x, weight, qkv, z, stream);
    }
}

} // namespace

const Fp8GdnInputProfile kFp8GdnInputBf16K5120{
    QType::FP8_E4M3FN_ROW_BF16,           5120,
    a16_k5120,                            fp8_gdn_input_a8_launch,
    fp8_gdn_input_partial_capacity_bytes, fp8_gdn_snapshot_fused_launch,
    fp8_gdn_record_fused_launch};

const Fp8GdnInputProfile* find_fp8_gdn_input_profile(QType qtype, std::int32_t parent_rows,
                                                     std::int32_t input_rows) noexcept {
    if (parent_rows != 16384) { return nullptr; }
    for (const Fp8GdnInputProfile* profile : {&kFp8GdnInputBf16K5120, &kFp8GdnInputFp32K2560}) {
        if (profile->qtype == qtype && profile->input_rows == input_rows) { return profile; }
    }
    return nullptr;
}

} // namespace ninfer::ops::detail
