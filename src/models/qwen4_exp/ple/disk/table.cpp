#include "models/qwen4_exp/ple/disk/table.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::ple::disk {

struct Table::Impl {
    struct Shard {
        artifact::FileObjectView object;
        Nf4Bf16Lut lut;
    };

    LookupParameters parameters;
    artifact::FileBacking backing;
    std::vector<Shard> shards;
    std::size_t max_segments = 0;

    Impl(const artifact::Reader& reader, LookupParameters params) : parameters(std::move(params)) {
        std::vector<artifact::ObjectHandle> objects;
        objects.reserve(parameters.shards.size());
        for (const auto& shard : parameters.shards) objects.push_back(shard.object);
        backing = artifact::prepare_file_backing(reader, objects);
        shards.reserve(objects.size());
        for (const auto& shard : parameters.shards) {
            const auto view = backing.object(shard.object);
            max_segments    = std::max(max_segments, view.segments.size() * (shard.geometry.format == QType::U4Z8_G16_FP16 ? 2 : 1));
            try {
                shards.push_back({view, shard.geometry.format == QType::NF4_G16_E4M3FN ? make_nf4_bf16_lut(shard.divisor) : Nf4Bf16Lut{}});
            } catch (const std::invalid_argument& e) {
                throw artifact::ArtifactError(reader.directory().tensor(shard.object).id + ": " +
                                              e.what());
            }
        }
    }
};

Table::Table(const artifact::Reader& reader, LookupParameters parameters)
    : impl_(std::make_unique<Impl>(reader, std::move(parameters))) {}

Table::~Table()                           = default;
Table::Table(Table&&) noexcept            = default;
Table& Table::operator=(Table&&) noexcept = default;

RowSource Table::row(std::uint32_t global_row) const {
    const auto& parameters = impl_->parameters;
    if (global_row >= parameters.rows) throw std::out_of_range("PLE global row exceeds table");
    const auto it = std::upper_bound(
        parameters.shards.begin(), parameters.shards.end(), global_row,
        [](std::uint32_t row, const TableShard& shard) { return row < shard.row_end; });
    const auto local   = global_row - (it->row_end - it->geometry.shape[0]);
    const auto& source = impl_->shards[it - parameters.shards.begin()];
    if (it->geometry.format == QType::NF4_G16_E4M3FN)
        return {source.object, weight_paged_row_offset(it->geometry, local), &source.lut};
    return {source.object, local * it->geometry.code_bytes_per_row, nullptr, QType::U4Z8_G16_FP16,
            it->geometry.scale_offset + local * it->geometry.scale_row_bytes};
}

std::size_t Table::row_segments(const RowSource& row, std::span<artifact::ReadSegment> output) const {
    if (row.format == QType::NF4_G16_E4M3FN)
        return artifact::map_file_range(row.object, row.offset, kNf4RowBytes, output);
    if (row.format != QType::U4Z8_G16_FP16) throw std::invalid_argument("unsupported PLE row representation");
    const auto codes = artifact::map_file_range(row.object, row.offset, kRowElements / 2, output);
    const auto scales = artifact::map_file_range(row.object, row.scale_offset, kRowElements / 8, output.subspan(codes));
    for (auto& part : output.subspan(codes, scales)) part.destination_offset += kRowElements / 2;
    return codes + scales;
}

const LookupParameters& Table::parameters() const noexcept { return impl_->parameters; }

std::span<const io::FileView> Table::files() const noexcept { return impl_->backing.files(); }

std::size_t Table::max_row_segments() const noexcept { return impl_->max_segments; }

} // namespace ninfer::models::qwen4_exp::ple::disk
