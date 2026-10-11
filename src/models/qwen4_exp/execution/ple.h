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

    // Ledger positions before `position` that select() can read.
    [[nodiscard]] std::uint32_t history_tokens() const noexcept { return orders_; }

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

// Compressed row destination of one gather: codes U8 [code_row_bytes, rows] and FP16 scales
// [groups, rows] in column-per-row order, the operand form of ops::ple_ngram_decode. PLE layer
// k's rows begin layer_code_bytes / layer_scale_bytes after layer k - 1's.
struct PleRowStaging {
    std::byte* codes              = nullptr;
    std::byte* scales             = nullptr;
    std::size_t layer_code_bytes  = 0;
    std::size_t layer_scale_bytes = 0;
};

// Gathers the rows of every PLE layer. A transfer of T tokens is planar by layer: the heads()
// rows of token t for PLE layer k are rows (k * T + t) * heads() .. + heads(), so each layer's
// rows form one contiguous decode operand of T * heads() rows.
class PleGather {
public:
    // `staged_positions` sizes the pinned staging that gather() fills (one prefill chunk).
    PleGather(std::span<const PleTable> tables, const PleConfig& config,
              std::uint32_t staged_positions, std::uint32_t worker_threads);
    ~PleGather();

    PleGather(const PleGather&)            = delete;
    PleGather& operator=(const PleGather&) = delete;

    [[nodiscard]] std::uint32_t layers() const noexcept {
        return static_cast<std::uint32_t>(layers_.size());
    }

    [[nodiscard]] std::uint32_t heads() const noexcept { return heads_; }

    [[nodiscard]] std::uint32_t history_tokens() const noexcept { return history_tokens_; }

    [[nodiscard]] std::size_t code_row_bytes() const noexcept { return code_row_bytes_; }

    [[nodiscard]] std::size_t scale_row_bytes() const noexcept { return scale_row_bytes_; }

    // Destination of token `index` in a planar transfer of `tokens` tokens whose code and scale
    // planes begin at `codes` and `scales`.
    [[nodiscard]] PleRowStaging token_staging(std::byte* codes, std::byte* scales,
                                              std::uint32_t tokens, std::uint32_t index) const;

    // Gathers the rows of ledger positions [begin, begin + count) into the pinned staging, planar
    // by layer over `count` tokens, and returns it; the scale plane follows the code plane. Small
    // calls run on the calling thread. The staging stays valid until the next gather; the caller
    // orders any asynchronous copy out of it before that.
    [[nodiscard]] PleRowStaging gather(std::span<const TokenId> ledger, std::uint32_t begin,
                                       std::uint32_t count);

    // One position's rows of every layer (decode rows are gathered one at a time into their
    // ingress slots).
    void gather_position(std::span<const TokenId> ledger, std::uint32_t position,
                         PleRowStaging staging) const;

private:
    struct Shard {
        const std::byte* codes  = nullptr;
        const std::byte* scales = nullptr;
    };

    struct Layer {
        PleRowSelector selector;
        std::vector<Shard> shards;
        std::uint64_t rows_per_shard = 0;
    };

    void copy_rows(const Layer& layer, std::span<const std::int64_t> rows, std::byte* codes,
                   std::byte* scales) const;

    std::vector<Layer> layers_;
    std::uint32_t heads_            = 0;
    std::uint32_t history_tokens_   = 0;
    std::size_t code_row_bytes_     = 0;
    std::size_t scale_row_bytes_    = 0;
    std::uint32_t staged_positions_ = 0;
    std::optional<PinnedHostBuffer> staging_;
    std::unique_ptr<HostWorkerPool> workers_;
};

} // namespace ninfer::models::qwen4_exp::execution
