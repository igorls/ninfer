#pragma once

// Fused GDN convolution routes over one row-scaled FP8 [16384,K] q/k/value/z parent: the
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

// Independent W=1 batch rows share the projection tile while preserving its FP32 result
// through convolution. Match the B=1 lane partition and four accumulation chains; changing
// that reduction can amplify into a different real-model trajectory. Only final outputs and
// persistent history are rounded to BF16.
struct Fp8GdnBatchSnapshotOutput : GdnConvOutput<1, SnapshotHistoryPublish> {
    __device__ __forceinline__ void store(int parent_row, int batch, float projected) const {
        auto row           = static_cast<const GdnConvOutput<1, SnapshotHistoryPublish>&>(*this);
        row.conv.batch_row = batch;
        row.z += static_cast<std::int64_t>(batch) * kGdnZRows;
        row.store(parent_row, 0, projected);
    }
};

template <int InputRows, class Scale, int BatchTile>
void fp8_gdn_snapshot_batch_one(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                Tensor& conv_states, const Tensor& valid_columns,
                                const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                cudaStream_t stream) {
    using Schedule =
        Fp8A16SimtSchedule<4, 2, 8, BatchTile, 4, Fp8SimtActivationAccess::SharedPhase,
                           Fp8CodeCache::Default, 2, Fp8SimtBlockOrder::RowsContiguous, 2>;
    const SnapshotHistoryPublish publish{static_cast<__nv_bfloat16*>(conv_states.data),
                                         static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                         kGdnChannels};
    Fp8GdnBatchSnapshotOutput output{make_gdn_conv_output<1>(
        conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z, publish)};
    Tensor flat(x.data, DType::BF16, {InputRows, x.ne[2]});
    launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, InputRows>>(
        fp8_a16_operands<Scale>(flat, weight), output, LinearIdentityEpilogue{}, stream);
}

// A CTA owns one sequence's full speculative window. Independent sequences reuse the same
// reduction order as W=1; convolution never crosses from one batch row to the next.
template <int Width, class Publish>
struct Fp8GdnWindowOutput : GdnConvOutput<Width, Publish, true> {
    __device__ __forceinline__ void store_window(int parent_row, int token_begin,
                                                 const float (&projected)[Width]) const {
        auto row = static_cast<const GdnConvOutput<Width, Publish, true>&>(*this);
        row.conv.batch_row = token_begin / Width;
        row.z += static_cast<std::int64_t>(token_begin) * kGdnZRows;
        row.store_row(parent_row, projected);
    }
};

template <int Width>
struct Fp8GdnWindowEpilogue {
    static constexpr int kRowTokens = Width;

    template <class Output>
    __device__ __forceinline__ void apply_row(const Output& output, int row, int token_begin,
                                               const float (&values)[Width], int) const {
        output.store_window(row, token_begin, values);
    }
};

template <int InputRows, class Scale, int Width, class Publish>
void fp8_gdn_conv_window(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                         const Tensor& conv_states, const Tensor& valid_columns,
                         const Tensor& initial_slot, Tensor& query, Tensor& key, Tensor& value,
                         Tensor& z, Publish publish, cudaStream_t stream) {
    using Schedule =
        Fp8A16SimtSchedule<4, 2, 8, Width, 4, Fp8SimtActivationAccess::SharedPhase,
                           Fp8CodeCache::Default, 2, Fp8SimtBlockOrder::RowsContiguous, 2>;
    Fp8GdnWindowOutput<Width, Publish> output{make_gdn_conv_output<Width, true>(
        conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z, publish)};
    Tensor flat(x.data, DType::BF16, {InputRows, x.ne[1] * x.ne[2]});
    launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, InputRows>>(
        fp8_a16_operands<Scale>(flat, weight), output, Fp8GdnWindowEpilogue<Width>{}, stream);
}

template <int InputRows, class Scale, class Publish>
void fp8_gdn_conv_window_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                  const Tensor& conv_states, const Tensor& valid_columns,
                                  const Tensor& initial_slot, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, Publish publish, cudaStream_t stream) {
    const auto launch = [&]<int Width>() {
        fp8_gdn_conv_window<InputRows, Scale, Width>(x, weight, conv_weight, conv_states,
            valid_columns, initial_slot, query, key, value, z, publish, stream);
    };
    switch (x.ne[1]) {
    case 2: launch.template operator()<2>(); return;
    case 3: launch.template operator()<3>(); return;
    case 4: launch.template operator()<4>(); return;
    case 5: launch.template operator()<5>(); return;
    case 6: launch.template operator()<6>(); return;
    default: throw std::invalid_argument("fp8 GDN fused window: unsupported width");
    }
}

// Snapshot form: the FP32-scale 2560 profile also fuses B=1..8, W=1..6.
template <int InputRows, class Scale>
void fp8_gdn_snapshot_fused(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                            Tensor& conv_states, const Tensor& valid_columns,
                            const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                            Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                            cudaStream_t stream) {
    if constexpr (InputRows == 2560) {
        if (x.ne[1] >= 2 && x.ne[1] <= 6 && x.ne[2] >= 1 && x.ne[2] <= 8) {
            fp8_gdn_conv_window_dispatch<InputRows, Scale>(
                x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                query, key, value, z,
                SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                    static_cast<const std::int32_t*>(snapshot_base_slot.data), kGdnChannels}, stream);
            return;
        }
        if (x.ne[1] == 1 && x.ne[2] > 1 && x.ne[2] <= 8) {
            const auto launch = [&]<int BatchTile>() {
                fp8_gdn_snapshot_batch_one<InputRows, Scale, BatchTile>(
                    x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                    snapshot_base_slot, query, key, value, z, stream);
            };
            if (x.ne[2] <= 2) {
                launch.template operator()<2>();
            } else if (x.ne[2] <= 4) {
                launch.template operator()<4>();
            } else {
                launch.template operator()<8>();
            }
            return;
        }
    }
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
                                                  valid_columns, initial_slot, query, key, value, z,
                                                  publish, stream);
        return;
    }
    fp8_gdn_conv_small_t<InputRows, Scale, 3>(x, weight, conv_weight, conv_states, valid_columns,
                                              initial_slot, query, key, value, z, publish, stream);
}

// Record form uses the same projection and convolution arithmetic as snapshot form.
template <int InputRows, class Scale>
void fp8_gdn_record_fused(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                          const Tensor& conv_states, const Tensor& valid_columns,
                          const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                          Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    if constexpr (InputRows == 2560) {
        if (x.ne[1] >= 2 && x.ne[1] <= 6 && x.ne[2] >= 1 && x.ne[2] <= 8) {
            fp8_gdn_conv_window_dispatch<InputRows, Scale>(
                x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                query, key, value, z,
                RecordColumnPublish{static_cast<__nv_bfloat16*>(conv_record.data),
                    kGdnChannels, static_cast<std::int32_t>(x.ne[1])}, stream);
            return;
        }
    }
    if (x.ne[2] != 1 || x.ne[1] < 2 || x.ne[1] > 3) {
        throw std::invalid_argument("fp8 GDN record fused: unsupported B/W");
    }
    auto* record = static_cast<__nv_bfloat16*>(conv_record.data);
    if (x.ne[1] == 2) {
        fp8_gdn_conv_small_t<InputRows, Scale, 2>(
            x, weight, conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
            RecordColumnPublish{record, kGdnChannels, 2}, stream);
        return;
    }
    fp8_gdn_conv_small_t<InputRows, Scale, 3>(x, weight, conv_weight, conv_states, valid_columns,
                                              initial_slot, query, key, value, z,
                                              RecordColumnPublish{record, kGdnChannels, 3}, stream);
}

} // namespace ninfer::ops::detail
