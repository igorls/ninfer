#pragma once

#include "models/qwen4_exp/ple/parameters.h"
#include "artifact/backing.h"
#include "core/nf4.h"
#include "core/u4z8.h"

namespace ninfer::models::qwen4_exp::ple::disk {

inline constexpr std::size_t kNf4RowBytes = kRowElements / 2 + kRowElements / 16;
inline constexpr std::size_t kRowBytes = kRowElements / 2 + 2 * kRowElements / 16;

struct RowSource {
    artifact::FileObjectView object;
    std::uint64_t offset; // Encoded row's byte offset within object.
    const Nf4Bf16Lut* lut;
    QType format = QType::NF4_G16_E4M3FN;
    std::uint64_t scale_offset = 0;
};

// Immutable direct-I/O backing and derived NF4 LUTs. Parameters were loaded from this Reader;
// construction reads framing, not table rows. Returned views survive owner moves and borrow
// the owner through their last I/O/CPU consumer.
class Table {
public:
    Table(const artifact::Reader&, LookupParameters);
    ~Table();
    Table(Table&&) noexcept;
    Table& operator=(Table&&) noexcept;
    Table(const Table&)            = delete;
    Table& operator=(const Table&) = delete;

    [[nodiscard]] RowSource row(std::uint32_t global_row) const;
    [[nodiscard]] std::size_t row_segments(const RowSource&, std::span<artifact::ReadSegment>) const;
    [[nodiscard]] const LookupParameters& parameters() const noexcept;
    [[nodiscard]] std::span<const io::FileView> files() const noexcept;
    [[nodiscard]] std::size_t max_row_segments() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen4_exp::ple::disk
