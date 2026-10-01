#pragma once

// Fused B=1 GDN convolution routes over one row-scaled FP8 [16384,K] q/k/value/z parent: the
// contraction's complete token row feeds the width-four convolution, the snapshot or record
// publication and the z store. Each profile's translation unit instantiates these once for its
// K and row-scale word.

#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"

#include "core/device.h"
#include "ops/gdn_input_proj/gdn_conv_output.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/fp8/fp8_a16_gemv.cuh"
#include "ops/linear/fp8/fp8_a16_simt.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {

template <int Tokens, class Publish>
struct Fp8GdnConvEpilogue {
    [[maybe_unused]] static constexpr int kRowTokens = Tokens;

    template <class Output>
    __device__ __forceinline__ void apply_row(const Output& output, int row, int,
                                              const float (&values)[Tokens], int) const {
        output.store_row(row, values);
    }
};

template <int InputRows, class Scale, int ActiveTokens, class Publish>
void fp8_gdn_conv_small_t(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                          const Tensor& conv_states, const Tensor& valid_columns,
                          const Tensor& initial_slot, Tensor& query, Tensor& key, Tensor& value,
                          Tensor& z, Publish publish, cudaStream_t stream) {
    using Schedule =
        Fp8A16SimtSchedule<8, 2, 16, ActiveTokens, 1, Fp8SimtActivationAccess::SharedPhase,
                           Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
    static_assert(Schedule::kBlockTokens == ActiveTokens);
    launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, InputRows, ActiveTokens, true>>(
        fp8_a16_operands<Scale>(x, weight),
        make_gdn_conv_output<ActiveTokens>(conv_weight, conv_states, valid_columns, initial_slot,
                                           query, key, value, z, publish),
        Fp8GdnConvEpilogue<ActiveTokens, Publish>{}, stream);
}

// Snapshot form, B = 1, W = 1..3.
template <int InputRows, class Scale>
void fp8_gdn_snapshot_fused(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                            Tensor& conv_states, const Tensor& valid_columns,
                            const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                            Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                            cudaStream_t stream) {
    if (x.ne[2] != 1 || x.ne[1] <= 0 || x.ne[1] > 3) {
        throw std::invalid_argument("fp8 GDN snapshot fused: unsupported B/W");
    }
    const SnapshotHistoryPublish publish{static_cast<__nv_bfloat16*>(conv_states.data),
                                         static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                         kGdnChannels};
    if (x.ne[1] == 1) {
        using Schedule = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
        launch_fp8_a16_gemv<Fp8ScheduleInstance<Schedule, InputRows>>(
            fp8_a16_operands<Scale>(x, weight),
            make_gdn_conv_output<1>(conv_weight, conv_states, valid_columns, initial_slot, query,
                                    key, value, z, publish),
            Fp8GdnConvEpilogue<1, SnapshotHistoryPublish>{}, stream);
        return;
    }
    if (x.ne[1] == 2) {
        fp8_gdn_conv_small_t<InputRows, Scale, 2>(x, weight, conv_weight, conv_states,
                                                  valid_columns, initial_slot, query, key, value,
                                                  z, publish, stream);
        return;
    }
    fp8_gdn_conv_small_t<InputRows, Scale, 3>(x, weight, conv_weight, conv_states, valid_columns,
                                              initial_slot, query, key, value, z, publish, stream);
}

// Record form, B = 1, W = 2..3.
template <int InputRows, class Scale>
void fp8_gdn_record_fused(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                          const Tensor& conv_states, const Tensor& valid_columns,
                          const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                          Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    if (x.ne[2] != 1 || x.ne[1] < 2 || x.ne[1] > 3) {
        throw std::invalid_argument("fp8 GDN record fused: unsupported B/W");
    }
    auto* record = static_cast<__nv_bfloat16*>(conv_record.data);
    if (x.ne[1] == 2) {
        fp8_gdn_conv_small_t<InputRows, Scale, 2>(x, weight, conv_weight, conv_states,
                                                  valid_columns, initial_slot, query, key, value,
                                                  z, RecordColumnPublish{record, kGdnChannels, 2},
                                                  stream);
        return;
    }
    fp8_gdn_conv_small_t<InputRows, Scale, 3>(x, weight, conv_weight, conv_states, valid_columns,
                                              initial_slot, query, key, value, z,
                                              RecordColumnPublish{record, kGdnChannels, 3},
                                              stream);
}

} // namespace ninfer::ops::detail
