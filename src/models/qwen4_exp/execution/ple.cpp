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

PleGather::PleGather(std::span<const PleTable> tables, const PleConfig& config,
                     std::uint32_t staged_positions, std::uint32_t worker_threads)
    : heads_(config.heads()), history_tokens_(config.ngram_size - 1U),
      code_row_bytes_(config.code_row_bytes()), scale_row_bytes_(config.scale_row_bytes()),
      staged_positions_(staged_positions) {
    if (tables.empty() || tables.size() != config.layers.size()) {
        throw std::invalid_argument("PLE gather: one table per configured PLE layer is required");
    }
    layers_.reserve(tables.size());
    for (const PleTable& table : tables) {
        if (table.shards.empty()) {
            throw std::invalid_argument("PLE gather: a table has no shards");
        }
        Layer& layer         = layers_.emplace_back(Layer{PleRowSelector(table, config), {}, 0});
        layer.rows_per_shard = table.rows_per_shard();
        layer.shards.reserve(table.shards.size());
        for (const WeightView& shard : table.shards) {
            if (shard.parts.size() != 1 || shard.shape.size() != 2 ||
                shard.shape[0] != layer.rows_per_shard) {
                throw std::invalid_argument(
                    "PLE gather: shards must be complete equal-height tables");
            }
            const WeightRowPlanes planes = weight_row_planes(shard.parts.front());
            if (planes.codes == nullptr || planes.scales == nullptr || planes.row_begin != 0 ||
                planes.row_count != layer.rows_per_shard) {
                throw std::invalid_argument("PLE gather: shard planes are not addressable");
            }
            // Transfers and the decode operand are sized from the configured head width.
            if (planes.code_row_bytes != code_row_bytes_ ||
                planes.scale_row_bytes != scale_row_bytes_) {
                throw std::invalid_argument(
                    "PLE gather: stored rows differ from the configured head width");
            }
            layer.shards.push_back({planes.codes, planes.scales});
        }
    }
    if (staged_positions_ != 0) {
        staging_.emplace(static_cast<std::size_t>(staged_positions_) * layers() * heads_ *
                         (code_row_bytes_ + scale_row_bytes_));
    }
    if (worker_threads != 0) { workers_ = std::make_unique<HostWorkerPool>(worker_threads, 4096); }
}

PleGather::~PleGather() = default;

void PleGather::copy_rows(const Layer& layer, std::span<const std::int64_t> rows, std::byte* codes,
                          std::byte* scales) const {
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto row   = static_cast<std::uint64_t>(rows[i]);
        const auto shard = row / layer.rows_per_shard;
        if (shard >= layer.shards.size()) {
            throw std::out_of_range("PLE gather: selected row is outside the table");
        }
        const std::uint64_t local = row % layer.rows_per_shard;
        std::memcpy(codes + i * code_row_bytes_,
                    layer.shards[shard].codes + local * code_row_bytes_, code_row_bytes_);
        std::memcpy(scales + i * scale_row_bytes_,
                    layer.shards[shard].scales + local * scale_row_bytes_, scale_row_bytes_);
    }
}

PleRowStaging PleGather::token_staging(std::byte* codes, std::byte* scales, std::uint32_t tokens,
                                       std::uint32_t index) const {
    if (codes == nullptr || scales == nullptr || index >= tokens) {
        throw std::invalid_argument("PLE gather: token destination is outside its transfer");
    }
    const std::size_t token_rows = heads_;
    return {codes + index * token_rows * code_row_bytes_,
            scales + index * token_rows * scale_row_bytes_, tokens * token_rows * code_row_bytes_,
            tokens * token_rows * scale_row_bytes_};
}

void PleGather::gather_position(std::span<const TokenId> ledger, std::uint32_t position,
                                PleRowStaging staging) const {
    std::int64_t rows[64];
    if (heads_ > std::size(rows)) { throw std::invalid_argument("PLE gather: too many heads"); }
    const std::span<std::int64_t> selected(rows, heads_);
    for (std::size_t k = 0; k < layers_.size(); ++k) {
        layers_[k].selector.select(ledger, position, selected);
        copy_rows(layers_[k], selected, staging.codes + k * staging.layer_code_bytes,
                  staging.scales + k * staging.layer_scale_bytes);
    }
}

PleRowStaging PleGather::gather(std::span<const TokenId> ledger, std::uint32_t begin,
                               std::uint32_t count) {
    if (count == 0 || count > staged_positions_ ||
        static_cast<std::uint64_t>(begin) + count > ledger.size()) {
        throw std::invalid_argument("PLE gather: positions are outside the ledger or staging");
    }
    auto* codes  = static_cast<std::byte*>(staging_->data());
    auto* scales = codes + static_cast<std::size_t>(count) * layers() * heads_ * code_row_bytes_;

    const auto run = [&](std::uint32_t first, std::uint32_t last) {
        for (std::uint32_t position = first; position < last; ++position) {
            gather_position(ledger, position,
                            token_staging(codes, scales, count, position - begin));
        }
    };
    const PleRowStaging staging = token_staging(codes, scales, count, 0);
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
