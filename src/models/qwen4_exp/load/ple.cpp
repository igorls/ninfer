#include "models/qwen4_exp/load/bindings.h"

#include <string>

namespace ninfer::models::qwen4_exp::loading {

PendingPleTable bind_ple_table(Bindings& b, const TextConfig& config) {
    const auto& ple = config.ple;
    const auto p    = "text/layers/" + std::to_string(ple.layer) + "/ple/embedding/";
    PendingPleTable out;
    out.layer_multipliers      = b.int64_values(p + "layer_multipliers", ple.ngram_size);
    out.ngram_head_offsets     = b.int64_values(p + "ngram_head_offsets", ple.heads());
    out.ngram_head_vocab_sizes = b.int64_values(p + "ngram_head_vocab_sizes", ple.heads());
    for (const auto multiplier : out.layer_multipliers) {
        if (multiplier <= 0) {
            throw artifact::ArtifactError("PLE n-gram hash multipliers must be positive");
        }
    }
    // Heads occupy consecutive row ranges of one global table.
    std::int64_t rows = 0;
    for (std::uint32_t head = 0; head < ple.heads(); ++head) {
        if (out.ngram_head_vocab_sizes[head] <= 0 || out.ngram_head_offsets[head] != rows) {
            throw artifact::ArtifactError("PLE head ranges must be positive and consecutive");
        }
        rows += out.ngram_head_vocab_sizes[head];
    }
    // Shard height is a storage fact: the table rounded up to split_ngram_parts equal shards.
    const auto& directory    = b.binder.reader().directory();
    std::uint64_t shard_rows = 0;
    for (std::uint32_t shard = 0; shard < ple.shards; ++shard) {
        const auto name  = p + "shards/" + std::to_string(shard);
        const auto found = directory.bindings.find(name);
        if (found == directory.bindings.end() || !found->second.whole_object) {
            throw artifact::ArtifactError(name + ": PLE shard must bind one whole object");
        }
        const auto& shape = directory.tensor(found->second.parts.at(0).object).shape;
        if (shape.size() != 2 || (shard && shape[0] != shard_rows)) {
            throw artifact::ArtifactError(name + ": PLE shards must be equal row blocks");
        }
        shard_rows = shape[0];
        out.shards.push_back(b.mapped(name, {shard_rows, ple.head_width()}, QType::U4Z8_G16_FP16));
    }
    if (artifact::checked_mul(shard_rows, ple.shards, "PLE rows") < std::uint64_t(rows)) {
        throw artifact::ArtifactError("PLE shards do not cover every n-gram head row");
    }
    return out;
}

} // namespace ninfer::models::qwen4_exp::loading
