#include "targets/qwen3_8_flash_next/impl/ple_table.h"

#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

constexpr std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

float fp16_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
    std::uint32_t exponent   = (bits >> 10U) & 0x1FU;
    std::uint32_t mantissa   = bits & 0x03FFU;
    std::uint32_t fp32       = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            fp32 = sign;
        } else {
            std::int32_t unbiased = -14;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                --unbiased;
            }
            mantissa &= 0x03FFU;
            fp32 = sign | (static_cast<std::uint32_t>(unbiased + 127) << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 0x1FU) {
        fp32 = sign | 0x7F80'0000U | (mantissa << 13U);
    } else {
        fp32 = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    return std::bit_cast<float>(fp32);
}

} // namespace

PleRowAddress locate_ple_row(std::uint64_t global_row) {
    if (global_row >= kPleTableRows) { throw std::out_of_range("PLE row is outside the table"); }
    return {
        .shard = static_cast<std::uint32_t>(global_row / kPleRowsPerShard),
        .row   = static_cast<std::uint32_t>(global_row % kPleRowsPerShard),
    };
}

PleShardView make_ple_shard_view(std::span<const std::byte> encoded, std::uint64_t rows,
                                 std::uint64_t width) {
    if (rows == 0 || width == 0 || width % 16 != 0 ||
        rows > std::numeric_limits<std::uint64_t>::max() / width) {
        throw std::invalid_argument("invalid PLE shard geometry");
    }
    const std::uint64_t elements     = rows * width;
    const std::uint64_t code_bytes   = elements / 2;
    const std::uint64_t scale_offset = align_up(code_bytes, 256);
    const std::uint64_t scale_bytes  = rows * (width / 16) * 2;
    if (scale_offset > encoded.size() || scale_bytes > encoded.size() - scale_offset ||
        scale_offset + scale_bytes != encoded.size()) {
        throw std::invalid_argument("PLE shard payload does not match packed-u4-g16-v1");
    }
    return {
        .codes          = encoded.first(static_cast<std::size_t>(code_bytes)),
        .scales         = encoded.subspan(static_cast<std::size_t>(scale_offset),
                                          static_cast<std::size_t>(scale_bytes)),
        .rows           = rows,
        .width          = width,
        .groups_per_row = width / 16,
    };
}

void dequantize_ple_row(const PleShardView& shard, std::uint64_t row, std::span<float> output) {
    if (row >= shard.rows) { throw std::out_of_range("PLE row is outside the shard"); }
    if (output.size() != shard.width) {
        throw std::invalid_argument("PLE row output has the wrong width");
    }
    const std::uint64_t code_row_offset  = row * shard.width / 2;
    const std::uint64_t scale_row_offset = row * shard.groups_per_row * 2;
    for (std::uint64_t column = 0; column < shard.width; ++column) {
        const std::uint8_t packed =
            std::to_integer<std::uint8_t>(shard.codes[code_row_offset + column / 2]);
        const std::uint8_t code =
            column % 2 == 0 ? packed & 0x0FU : static_cast<std::uint8_t>(packed >> 4U);
        std::uint16_t scale_bits         = 0;
        const std::uint64_t scale_offset = scale_row_offset + (column / 16) * 2;
        std::memcpy(&scale_bits, shard.scales.data() + scale_offset, sizeof(scale_bits));
        output[column] = (static_cast<float>(code) - 8.0F) * fp16_to_float(scale_bits);
    }
}

void gather_ple_rows(const PleTableView& table, std::span<const std::int64_t, 16> global_rows,
                     std::span<float> output) {
    constexpr std::size_t kOutputWidth = 16 * kPleRowWidth;
    if (output.size() != kOutputWidth) {
        throw std::invalid_argument("PLE gather output must contain 2560 values");
    }
    for (std::size_t head = 0; head < global_rows.size(); ++head) {
        if (global_rows[head] < 0) { throw std::out_of_range("PLE row must be non-negative"); }
        const PleRowAddress address = locate_ple_row(static_cast<std::uint64_t>(global_rows[head]));
        dequantize_ple_row(table.shards[address.shard], address.row,
                           output.subspan(head * kPleRowWidth, kPleRowWidth));
    }
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
