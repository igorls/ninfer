#include "models/qwen4_exp/execution/ple.h"

#include "core/weight_view.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <future>
#include <stdexcept>

namespace ninfer::models::qwen4_exp::execution {
namespace {

// Calls up to this many positions gather on the calling thread: a worker round-trip costs more
// than the copies (decode gathers one to eight positions per round).
constexpr std::uint32_t kInlinePositions = 8;
// Positions per worker task in a large gather.
constexpr std::uint32_t kPositionsPerTask = 256;

std::uint64_t wrapped_product(std::int64_t left, std::int64_t right) noexcept {
    return static_cast<std::uint64_t>(left) * static_cast<std::uint64_t>(right);
}

std::int64_t floor_mod(std::uint64_t bits, std::int64_t divisor) noexcept {
    const std::int64_t value  = std::bit_cast<std::int64_t>(bits);
    std::int64_t remainder    = value % divisor;
    if (remainder < 0) { remainder += divisor; }
    return remainder;
}

} // namespace

PleRowSelector::PleRowSelector(const PleTable& table, const PleConfig& config)
    : multipliers_(table.layer_multipliers), head_offsets_(table.ngram_head_offsets),
      head_vocab_(table.ngram_head_vocab_sizes), heads_per_ngram_(config.heads_per_ngram),
      heads_(config.heads()), orders_(config.ngram_size - 1U), boundary_(config.boundary_token) {
    if (heads_per_ngram_ == 0 || orders_ == 0 || multipliers_.size() != config.ngram_size ||
        head_offsets_.size() != heads_ || head_vocab_.size() != heads_) {
        throw std::invalid_argument("PLE row selector: index tables do not match the n-gram config");
    }
    for (std::uint32_t head = 0; head < heads_; ++head) {
        if (head_offsets_[head] < 0 || head_vocab_[head] <= 0) {
            throw std::invalid_argument("PLE row selector: invalid head range");
        }
    }
}

void PleRowSelector::select(std::span<const TokenId> ledger, std::uint32_t position,
                            std::span<std::int64_t> rows) const {
    if (position >= ledger.size() || rows.size() != heads_) {
        throw std::invalid_argument("PLE row selector: position or output is out of range");
    }
    // hashes[n - 2] is the order-n hash; history tokens walk back until a boundary.
    std::uint64_t hash = wrapped_product(ledger[position], multipliers_[0]);
    bool at_boundary   = false;
    std::uint32_t head = 0;
    for (std::uint32_t order = 1; order <= orders_; ++order) {
        TokenId history = boundary_;
        if (!at_boundary && position >= order) {
            history = ledger[position - order];
            if (history == boundary_) { at_boundary = true; }
        } else {
            at_boundary = true;
        }
        hash ^= wrapped_product(history, multipliers_[order]);
        for (std::uint32_t i = 0; i < heads_per_ngram_; ++i, ++head) {
            rows[head] = head_offsets_[head] + floor_mod(hash, head_vocab_[head]);
        }
    }
}

PleGather::PleGather(const PleTable& table, const PleConfig& config,
                     std::uint32_t staged_positions, std::uint32_t worker_threads)
    : selector_(table, config), staged_positions_(staged_positions) {
    if (table.shards.empty()) { throw std::invalid_argument("PLE gather: the table has no shards"); }
    rows_per_shard_ = table.rows_per_shard();
    shards_.reserve(table.shards.size());
    for (const WeightView& shard : table.shards) {
        if (shard.parts.size() != 1 || shard.shape.size() != 2 ||
            shard.shape[0] != rows_per_shard_) {
            throw std::invalid_argument("PLE gather: shards must be complete equal-height tables");
        }
        const WeightRowPlanes planes = weight_row_planes(shard.parts.front());
        if (planes.codes == nullptr || planes.scales == nullptr || planes.row_begin != 0 ||
            planes.row_count != rows_per_shard_) {
            throw std::invalid_argument("PLE gather: shard planes are not addressable");
        }
        if (code_row_bytes_ == 0) {
            code_row_bytes_  = planes.code_row_bytes;
            scale_row_bytes_ = planes.scale_row_bytes;
        } else if (code_row_bytes_ != planes.code_row_bytes ||
                   scale_row_bytes_ != planes.scale_row_bytes) {
            throw std::invalid_argument("PLE gather: shards disagree on the row geometry");
        }
        shards_.push_back({planes.codes, planes.scales});
    }
    if (staged_positions_ != 0) {
        staging_.emplace(static_cast<std::size_t>(staged_positions_) * heads() *
                         (code_row_bytes_ + scale_row_bytes_));
    }
    if (worker_threads != 0) { workers_ = std::make_unique<HostWorkerPool>(worker_threads, 4096); }
}

PleGather::~PleGather() = default;

void PleGather::copy_rows(std::span<const std::int64_t> rows, std::byte* codes,
                          std::byte* scales) const {
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto row   = static_cast<std::uint64_t>(rows[i]);
        const auto shard = row / rows_per_shard_;
        if (shard >= shards_.size()) {
            throw std::out_of_range("PLE gather: selected row is outside the table");
        }
        const std::uint64_t local = row % rows_per_shard_;
        std::memcpy(codes + i * code_row_bytes_, shards_[shard].codes + local * code_row_bytes_,
                    code_row_bytes_);
        std::memcpy(scales + i * scale_row_bytes_,
                    shards_[shard].scales + local * scale_row_bytes_, scale_row_bytes_);
    }
}

void PleGather::gather_position(std::span<const TokenId> ledger, std::uint32_t position,
                                PleRowStaging staging) const {
    std::int64_t rows[64];
    const std::uint32_t heads = selector_.heads();
    if (heads > std::size(rows)) { throw std::invalid_argument("PLE gather: too many heads"); }
    selector_.select(ledger, position, std::span<std::int64_t>(rows, heads));
    copy_rows(std::span<const std::int64_t>(rows, heads), staging.codes, staging.scales);
}

PleRowStaging PleGather::gather(std::span<const TokenId> ledger, std::uint32_t begin,
                               std::uint32_t count) {
    if (count == 0 || count > staged_positions_ ||
        static_cast<std::uint64_t>(begin) + count > ledger.size()) {
        throw std::invalid_argument("PLE gather: positions are outside the ledger or staging");
    }
    auto* base = static_cast<std::byte*>(staging_->data());
    const PleRowStaging staging{
        base, base + static_cast<std::size_t>(count) * heads() * code_row_bytes_};
    const std::uint32_t heads = selector_.heads();
    const auto run            = [&, heads](std::uint32_t first, std::uint32_t last) {
        for (std::uint32_t position = first; position < last; ++position) {
            const std::size_t column = static_cast<std::size_t>(position - begin) * heads;
            gather_position(ledger, position,
                            {staging.codes + column * code_row_bytes_,
                             staging.scales + column * scale_row_bytes_});
        }
    };
    if (count <= kInlinePositions || workers_ == nullptr) {
        run(begin, begin + count);
        return staging;
    }
    std::vector<std::future<void>> tasks;
    tasks.reserve((count + kPositionsPerTask - 1U) / kPositionsPerTask);
    for (std::uint32_t first = begin; first < begin + count; first += kPositionsPerTask) {
        const std::uint32_t last = std::min(begin + count, first + kPositionsPerTask);
        tasks.push_back(workers_->submit([run, first, last] { run(first, last); }));
    }
    std::exception_ptr failure;
    for (auto& task : tasks) {
        try {
            task.get();
        } catch (...) {
            if (!failure) { failure = std::current_exception(); }
        }
    }
    if (failure) { std::rethrow_exception(failure); }
    return staging;
}

} // namespace ninfer::models::qwen4_exp::execution
