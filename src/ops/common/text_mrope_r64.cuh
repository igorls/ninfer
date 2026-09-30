#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Interleaved three-axis Text MRoPE over the first 64 dimensions of a head, theta 1e7. Rotary pair
// i in [0,32) couples dimensions i and i+32 and uses position axis i%3 with angle
// position[i%3] * theta^(-2i/64). Positions are planar I32 [tokens,3] (axis-major), matching the
// Text MRoPE convention of ops::rope.
//
// The phase is formed and reduced modulo 2*pi in FP64 so that positions up to the model's 262144
// context keep the rotation accurate to FP32 rounding; only the reduced angle enters FP32 sincos.
inline constexpr int kTextMropeR64Pairs = 32;

// theta^(-2i/64) for theta = 1e7, correctly rounded to binary64. Internal linkage: one copy per
// translation unit under relocatable device code.
static __constant__ double kTextMropeR64Frequency[kTextMropeR64Pairs] = {
    1,
    0.60429639023813286,
    0.36517412725483772,
    0.22067340690845899,
    0.1333521432163324,
    0.080584218776148187,
    0.048696752516586311,
    0.029427271762092817,
    0.017782794100389229,
    0.010746078283213174,
    0.006493816315762113,
    0.0039241897584845363,
    0.0023713737056616554,
    0.0014330125702369627,
    0.00086596432336006539,
    0.00052329911468149473,
    0.00031622776601683794,
    0.00019109529749704405,
    0.00011547819846894582,
    6.9783058485986635e-05,
    4.2169650342858222e-05,
    2.5482967479793464e-05,
    1.5399265260594919e-05,
    9.3057204092969904e-06,
    5.6234132519034912e-06,
    3.3982083289425593e-06,
    2.0535250264571461e-06,
    1.2409377607517195e-06,
    7.4989420933245585e-07,
    4.5315836376008179e-07,
    2.7384196342643614e-07,
    1.6548170999431814e-07};

__device__ __forceinline__ float2 text_mrope_r64_sincos(int pair, std::int32_t position) {
    constexpr double kTwoPi    = 6.283185307179586476925286766559;
    constexpr double kInvTwoPi = 0.15915494309189533576888376337251;
    const double phase         = static_cast<double>(position) * kTextMropeR64Frequency[pair];
    const double reduced       = fma(-kTwoPi, rint(phase * kInvTwoPi), phase);
    float sine                 = 0.0F;
    float cosine               = 0.0F;
    sincosf(static_cast<float>(reduced), &sine, &cosine);
    return make_float2(sine, cosine);
}

__device__ __forceinline__ std::int32_t text_mrope_axis_position(const std::int32_t* positions,
                                                                 std::int64_t tokens,
                                                                 std::int64_t token, int pair) {
    return positions[static_cast<std::int64_t>(pair % 3) * tokens + token];
}

} // namespace ninfer::ops
