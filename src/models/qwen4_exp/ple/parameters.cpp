#include "models/qwen4_exp/ple/parameters.h"

#include "artifact/reader.h"

#include <bit>
#include <cmath>
#include <limits>

namespace ninfer::models::qwen4_exp::ple {

LookupParameters load_lookup_parameters(const artifact::Reader& reader,
                                        const artifact::Binding& table,
                                        artifact::ObjectHandle controls,
                                        HashGeometry hash_geometry) {
    using namespace artifact;
    if (hash_geometry.ngram_size != 3 || hash_geometry.heads_per_ngram != kHeads / 2)
        throw ArtifactError(
            "PLE hash controls v1 require eight heads for each of 2-gram and 3-gram");
    if (table.parts.empty()) throw ArtifactError("PLE table has no shards");
    LookupParameters result{};
    result.shards.reserve(table.parts.size());
    for (const auto& part : table.parts) {
        const auto& geometry = reader.geometry(part.object);
        const bool nf4 = geometry.format == QType::NF4_G16_E4M3FN && geometry.layout == QuantLayout::RowPagedG16_4K;
        const bool u4 = geometry.format == QType::U4Z8_G16_FP16 && geometry.layout == QuantLayout::PackedU4G16;
        if ((!nf4 && !u4) || geometry.shape.size() != 2 ||
            geometry.shape[1] != kRowElements || part.begin || part.end != geometry.elements) {
            throw ArtifactError("PLE requires complete NF4 or U4Z8 shards with width 160");
        }
        result.rows = checked_add(result.rows, geometry.shape[0], "PLE rows");
        float divisor = 0.F;
        if (nf4) {
        std::array<std::byte, 4> word{};
        reader.read_into(checked_add(reader.directory().tensor(part.object).offset,
                                     geometry.divisor_offset, "PLE divisor offset"),
                         word);
        divisor = std::bit_cast<float>(read_u32_le(word.data()));
        if (!(divisor > 0) || !std::isfinite(divisor)) {
            throw ArtifactError("PLE requires a positive finite divisor");
        }
        }
        result.shards.push_back({part.object, geometry, result.rows, divisor});
    }
    if (result.rows > std::numeric_limits<std::uint32_t>::max() ||
        checked_mul(result.rows, kRowElements, "PLE elements") != table.elements) {
        throw ArtifactError("PLE table logical extent is inconsistent");
    }
    const auto& resource = reader.directory().object(controls);
    if (!std::holds_alternative<ResourceObject>(resource) || object_bytes(resource) != 280) {
        throw ArtifactError("invalid PLE hash controls resource");
    }
    const auto bytes = reader.read_object(controls);
    auto word        = [&](std::size_t i) {
        return std::bit_cast<std::int64_t>(read_u64_le(bytes.data() + 8 * i));
    };
    auto& hash = result.hash;
    for (std::size_t i = 0; i < 3; ++i) hash.multipliers[i] = word(i);
    for (std::size_t i = 0; i < kHeads; ++i) {
        hash.vocab_sizes[i] = word(3 + i);
        hash.offsets[i]     = word(19 + i);
        if (hash.vocab_sizes[i] <= 0 || hash.offsets[i] < 0 ||
            std::uint64_t(hash.offsets[i]) > result.rows ||
            std::uint64_t(hash.vocab_sizes[i]) > result.rows - hash.offsets[i]) {
            throw ArtifactError("PLE hash head exceeds table rows");
        }
    }
    return result;
}

} // namespace ninfer::models::qwen4_exp::ple
