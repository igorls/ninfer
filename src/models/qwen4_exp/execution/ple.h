#pragma once

#include "core/arena.h"
#include "core/host_worker_pool.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/model.h"
#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp::execution {

// Host side of the per-layer n-gram embedding: n-gram row selection and the gather of the
// selected u4z8_g16 rows from the mapped table shards.
//
// Position i of a token ledger selects, for head h of n-gram order n = 2 + h / heads_per_ngram,
//   hash_n = XOR_{k<n} wrap64(token[i-k] * multiplier[k])
//   row_h  = head_offset[h] + floor_mod(int64(hash_n), head_vocab[h])
// where token[i-k] for k >= 1 comes from the segment history: it starts as the boundary token
// and committing the boundary token resets it, so a history token at or behind a boundary is the
// boundary token itself.
class PleRowSelector {
public:
    PleRowSelector(const PleTable& table, const PleConfig& config);

    [[nodiscard]] std::uint32_t heads() const noexcept { return heads_; }

    // Writes heads() global rows for ledger position `position` into `rows`.
    void select(std::span<const TokenId> ledger, std::uint32_t position,
                std::span<std::int64_t> rows) const;

private:
    std::vector<std::int64_t> multipliers_;
    std::vector<std::int64_t> head_offsets_;
    std::vector<std::int64_t> head_vocab_;
    std::uint32_t heads_per_ngram_ = 0;
    std::uint32_t heads_           = 0;
    std::uint32_t orders_          = 0;
    TokenId boundary_              = 0;
};

// Compressed row staging of one call: codes U8 [code_row_bytes, rows] and FP16 scales
// [groups, rows] in column-per-row order, the operand form of ops::ple_ngram_decode.
struct PleRowStaging {
    std::byte* codes  = nullptr;
    std::byte* scales = nullptr;
};

class PleGather {
public:
    // `staged_positions` sizes the pinned staging that gather() fills (one prefill chunk).
    PleGather(const PleTable& table, const PleConfig& config, std::uint32_t staged_positions,
              std::uint32_t worker_threads);
    ~PleGather();

    PleGather(const PleGather&)            = delete;
    PleGather& operator=(const PleGather&) = delete;

    [[nodiscard]] std::uint32_t heads() const noexcept { return selector_.heads(); }

    [[nodiscard]] std::size_t code_row_bytes() const noexcept { return code_row_bytes_; }

    [[nodiscard]] std::size_t scale_row_bytes() const noexcept { return scale_row_bytes_; }

    // Gathers the rows of ledger positions [begin, begin + count) into the pinned staging and
    // returns it; column (position - begin) * heads() + h holds head h. Small calls run on the
    // calling thread. The staging stays valid until the next gather; the caller orders any
    // asynchronous copy out of it before that.
    [[nodiscard]] PleRowStaging gather(std::span<const TokenId> ledger, std::uint32_t begin,
                                       std::uint32_t count);

    // One position's rows (decode rows are gathered one at a time into their ingress slots).
    void gather_position(std::span<const TokenId> ledger, std::uint32_t position,
                         PleRowStaging staging) const;

private:
    struct Shard {
        const std::byte* codes  = nullptr;
        const std::byte* scales = nullptr;
    };

    void copy_rows(std::span<const std::int64_t> rows, std::byte* codes,
                   std::byte* scales) const;

    PleRowSelector selector_;
    std::vector<Shard> shards_;
    std::uint64_t rows_per_shard_ = 0;
    std::size_t code_row_bytes_   = 0;
    std::size_t scale_row_bytes_  = 0;
    std::uint32_t staged_positions_ = 0;
    std::optional<PinnedHostBuffer> staging_;
    std::unique_ptr<HostWorkerPool> workers_;
};

} // namespace ninfer::models::qwen4_exp::execution
