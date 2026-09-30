#pragma once

#include <cstdint>

namespace ninfer::ops::detail {

// Registered paged-KV profiles consumed by selected-block attention.
enum class SelectedKvProfile : int {
    Bf16KFp16V = 0, ///< KvCacheStorage::BFloat16: BF16 K, FP16 V.
    Fp8Row256  = 1, ///< KvCacheStorage::Fp8E4M3Row256: Hadamard FP8 K, FP8 V, FP16 row scales.
};

inline constexpr int kSelectedHeadDim      = 256;
inline constexpr int kSelectedQueryHeads   = 24;
inline constexpr int kSelectedKvHeads      = 2;
inline constexpr int kSelectedHeadsPerKv   = 12;
inline constexpr int kSelectedPageTokens   = 64;
inline constexpr int kSelectedMaxBlocks    = 512;
inline constexpr float kSelectedScale      = 0.0625F; // 1/sqrt(256)
// Batched route: a column's visible set is split into at most this many partitions; each
// partition row holds 256 numerators, its maximum and its denominator (128-byte rows).
inline constexpr int kSelectedMaxPartitions = 32;
inline constexpr int kSelectedPartialStride = 288;

// Raw planes of one paged layer. Scale planes are null for the BF16/FP16 profile.
struct SelectedKvPlanes {
    const void* k               = nullptr;
    const void* v               = nullptr;
    const std::uint16_t* k_scale = nullptr; ///< FP16 bits
    const std::uint16_t* v_scale = nullptr; ///< FP16 bits
    const std::int32_t* tables  = nullptr; ///< [logical_pages, rows]
    std::int32_t logical_pages  = 0;
};

} // namespace ninfer::ops::detail
