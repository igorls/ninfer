#include "serve/expert_route_recorder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <utility>

namespace ninfer::serve {
namespace {

constexpr char kMagic[8]              = {'N', 'X', 'T', 'R', 'A', 'C', 'E', '1'};
constexpr std::uint32_t kRecordMarker = 0x43525452U;
constexpr std::uint8_t kPrefillHistogramKind = 5;
constexpr const char* kKindNames[5]   = {"unknown", "prefill", "decode", "mtp_proposal",
                                         "mtp_verify"};

template <class T>
void put(std::vector<std::byte>& out, T value) {
    const auto* bytes = reinterpret_cast<const std::byte*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(T));
}

std::uint64_t unix_micros() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::system_clock::now().time_since_epoch())
                                          .count());
}

nlohmann::json heat_rows(const std::vector<double>& values, std::uint32_t slices,
                         std::uint32_t experts) {
    nlohmann::json rows = nlohmann::json::array();
    for (std::uint32_t slice = 0; slice < slices; ++slice) {
        nlohmann::json row = nlohmann::json::array();
        for (std::uint32_t expert = 0; expert < experts; ++expert) {
            row.push_back(std::llround(values[slice * experts + expert]));
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

nlohmann::json total_rows(const std::vector<std::uint64_t>& values, std::uint32_t slices,
                          std::uint32_t experts) {
    nlohmann::json rows = nlohmann::json::array();
    for (std::uint32_t slice = 0; slice < slices; ++slice) {
        rows.push_back(std::vector<std::uint64_t>(values.begin() + slice * experts,
                                                  values.begin() + (slice + 1) * experts));
    }
    return rows;
}

// Per slice, the fewest experts holding 80% of the decayed decode activations.
nlohmann::json coverage_80(const std::vector<double>& values, std::uint32_t slices,
                           std::uint32_t experts) {
    nlohmann::json out = nlohmann::json::array();
    std::vector<double> row(experts);
    for (std::uint32_t slice = 0; slice < slices; ++slice) {
        std::copy(values.begin() + slice * experts, values.begin() + (slice + 1) * experts,
                  row.begin());
        std::sort(row.begin(), row.end(), std::greater<>());
        double sum = 0.0;
        for (const double value : row) { sum += value; }
        if (sum <= 0.0) {
            out.push_back(nullptr);
            continue;
        }
        double running     = 0.0;
        std::uint32_t used = 0;
        while (used < experts && running < 0.8 * sum) { running += row[used++]; }
        out.push_back(used);
    }
    return out;
}

} // namespace

ExpertRouteRecorder::ExpertRouteRecorder(std::filesystem::path trace_directory,
                                         std::uint64_t max_file_bytes)
    : directory_(std::move(trace_directory)), max_file_bytes_(max_file_bytes),
      started_(std::chrono::steady_clock::now()), last_decay_(started_) {
    if (!directory_.empty()) {
        writer_ = std::thread([this] { write_loop(); });
    }
}

ExpertRouteRecorder::~ExpertRouteRecorder() {
    {
        std::lock_guard lock(queue_mutex_);
        stopping_ = true;
    }
    queue_ready_.notify_all();
    if (writer_.joinable()) { writer_.join(); }
    if (file_ != nullptr) { std::fclose(file_); }
}

void ExpertRouteRecorder::expert_route_started(const ExpertRouteGeometry& geometry) {
    if (geometry.text_layers == 0 || geometry.experts == 0 || geometry.top_k == 0) {
        throw std::invalid_argument("routed-expert geometry is empty");
    }
    std::lock_guard lock(mutex_);
    geometry_          = geometry;
    experts_per_layer_ = geometry.experts;
    slices_            = geometry.text_layers + (geometry.mtp_layer ? 1U : 0U);
    const auto entries = static_cast<std::size_t>(slices_) * geometry.experts;
    decode_.decayed.assign(entries, 0.0);
    decode_.total.assign(entries, 0);
    prefill_.decayed.assign(entries, 0.0);
    prefill_.total.assign(entries, 0);
}

void ExpertRouteRecorder::configure(const std::string& server_instance_id,
                                    const std::string& model_id, const std::string& artifact_path) {
    if (directory_.empty()) { return; }
    std::optional<ExpertRouteGeometry> geometry;
    {
        std::lock_guard lock(mutex_);
        geometry = geometry_;
    }
    if (!geometry) { throw std::logic_error("routed-expert trace has no geometry"); }
    std::filesystem::create_directories(directory_);
    const auto path = directory_ / ("expert-trace-" + server_instance_id + ".nxtr");
    std::FILE* file = std::fopen(path.string().c_str(), "wb");
    if (file == nullptr) {
        throw std::runtime_error("cannot create expert trace file " + path.string());
    }
    const nlohmann::json metadata = {
        {"format", "ninfer-expert-trace"},
        {"version", 1},
        {"server_instance_id", server_instance_id},
        {"model_id", model_id},
        {"artifact", std::filesystem::path(artifact_path).filename().string()},
        {"text_layers", geometry->text_layers},
        {"mtp_layer", geometry->mtp_layer ? nlohmann::json(*geometry->mtp_layer) : nullptr},
        {"experts", geometry->experts},
        {"top_k", geometry->top_k},
        {"created_unix_ms", unix_micros() / 1000},
    };
    const std::string text = metadata.dump();
    std::vector<std::byte> header;
    header.insert(header.end(), reinterpret_cast<const std::byte*>(kMagic),
                  reinterpret_cast<const std::byte*>(kMagic) + sizeof(kMagic));
    put<std::uint32_t>(header, static_cast<std::uint32_t>(text.size()));
    header.insert(header.end(), reinterpret_cast<const std::byte*>(text.data()),
                  reinterpret_cast<const std::byte*>(text.data()) + text.size());
    if (std::fwrite(header.data(), 1, header.size(), file) != header.size()) {
        std::fclose(file);
        throw std::runtime_error("cannot write expert trace header " + path.string());
    }
    std::lock_guard lock(queue_mutex_);
    file_path_ = path;
    file_      = file;
    written_bytes_ += header.size();
}

void ExpertRouteRecorder::decay(std::chrono::steady_clock::time_point now) {
    const double elapsed = std::chrono::duration<double>(now - last_decay_).count();
    if (elapsed < 1.0) { return; }
    const double factor = std::pow(0.5, elapsed / kHeatHalfLifeSeconds);
    for (double& value : decode_.decayed) { value *= factor; }
    for (double& value : prefill_.decayed) { value *= factor; }
    last_decay_ = now;
}

void ExpertRouteRecorder::expert_route(const ExpertRouteRecord& record) noexcept {
    try {
        const std::size_t columns   = record.columns.size();
        const std::size_t per_layer = record.top_k;
        if (columns == 0 || record.layer_count == 0 || per_layer == 0 ||
            record.experts.size() != columns * record.layer_count * per_layer) {
            return;
        }
        {
            std::lock_guard lock(mutex_);
            if (!geometry_) { return; }
            decay(std::chrono::steady_clock::now());
            const auto kind = static_cast<std::size_t>(record.kind);
            if (kind < 5) {
                ++records_[kind];
                columns_[kind] += columns;
            }
            Heat& heat = record.kind == ExpertRouteKind::Prefill ? prefill_ : decode_;
            const std::uint32_t experts = geometry_->experts;
            for (std::size_t column = 0; column < columns; ++column) {
                for (std::uint32_t layer = 0; layer < record.layer_count; ++layer) {
                    const std::uint32_t slice = record.first_layer + layer;
                    if (slice >= slices_) { continue; }
                    const std::uint16_t* ids =
                        record.experts.data() + (column * record.layer_count + layer) * per_layer;
                    for (std::size_t rank = 0; rank < per_layer; ++rank) {
                        if (ids[rank] >= experts) { continue; }
                        const std::size_t entry =
                            static_cast<std::size_t>(slice) * experts + ids[rank];
                        heat.decayed[entry] += 1.0;
                        ++heat.total[entry];
                    }
                }
            }
        }
        {
            std::lock_guard lock(queue_mutex_);
            if (file_ == nullptr) { return; }
        }
        std::vector<std::byte> bytes;
        // A prefill chunk whose ids outnumber one count per expert is written as a per-layer
        // histogram: its streaming cost and its prompt's experts need counts, not token order.
        const std::uint32_t experts = experts_per_layer_;
        const bool histogram =
            record.kind == ExpertRouteKind::Prefill && experts != 0 && columns * per_layer > experts;
        const std::size_t payload =
            histogram ? static_cast<std::size_t>(record.layer_count) * experts
                      : record.experts.size();
        bytes.reserve(24 + (histogram ? 1 : columns) * 16 + payload * 2);
        put<std::uint32_t>(bytes, kRecordMarker);
        put<std::uint8_t>(bytes, histogram ? kPrefillHistogramKind
                                           : static_cast<std::uint8_t>(record.kind));
        put<std::uint8_t>(bytes, 0);
        put<std::uint16_t>(bytes, record.layer_count);
        put<std::uint16_t>(bytes, record.first_layer);
        put<std::uint16_t>(bytes, 0);
        put<std::uint32_t>(bytes, histogram ? 1U : static_cast<std::uint32_t>(columns));
        put<std::uint64_t>(bytes, unix_micros());
        if (histogram) {
            // One summary column: the request, the chunk's first position and its token count.
            put<std::uint64_t>(bytes, record.columns.front().request_tag);
            put<std::uint32_t>(bytes, record.columns.front().position);
            put<std::uint32_t>(bytes, static_cast<std::uint32_t>(columns));
            std::vector<std::uint16_t> counts(payload, 0);
            for (std::size_t column = 0; column < columns; ++column) {
                for (std::uint32_t layer = 0; layer < record.layer_count; ++layer) {
                    const std::uint16_t* ids =
                        record.experts.data() + (column * record.layer_count + layer) * per_layer;
                    for (std::size_t rank = 0; rank < per_layer; ++rank) {
                        if (ids[rank] >= experts) { continue; }
                        std::uint16_t& count = counts[layer * experts + ids[rank]];
                        if (count != UINT16_MAX) { ++count; }
                    }
                }
            }
            const auto* data = reinterpret_cast<const std::byte*>(counts.data());
            bytes.insert(bytes.end(), data, data + counts.size() * sizeof(std::uint16_t));
        } else {
            for (const ExpertRouteColumn& column : record.columns) {
                put<std::uint64_t>(bytes, column.request_tag);
                put<std::uint32_t>(bytes, column.position);
                put<std::uint32_t>(bytes, 0);
            }
            const auto* ids = reinterpret_cast<const std::byte*>(record.experts.data());
            bytes.insert(bytes.end(), ids, ids + record.experts.size() * sizeof(std::uint16_t));
        }
        enqueue(std::move(bytes));
    } catch (...) {
        std::lock_guard lock(queue_mutex_);
        ++dropped_records_;
    }
}

void ExpertRouteRecorder::enqueue(std::vector<std::byte>&& bytes) {
    {
        std::lock_guard lock(queue_mutex_);
        if (queued_bytes_ + bytes.size() > kQueueLimitBytes ||
            written_bytes_ + queued_bytes_ + bytes.size() > max_file_bytes_) {
            ++dropped_records_;
            return;
        }
        queued_bytes_ += bytes.size();
        queue_.push_back(std::move(bytes));
    }
    queue_ready_.notify_one();
}

void ExpertRouteRecorder::write_loop() {
    std::unique_lock lock(queue_mutex_);
    for (;;) {
        queue_ready_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
        if (queue_.empty() && stopping_) { break; }
        std::deque<std::vector<std::byte>> batch;
        batch.swap(queue_);
        std::FILE* file = file_;
        lock.unlock();
        std::size_t written = 0;
        std::size_t drained = 0;
        for (const auto& bytes : batch) {
            drained += bytes.size();
            if (file != nullptr &&
                std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size()) {
                written += bytes.size();
            }
        }
        if (file != nullptr) { std::fflush(file); }
        lock.lock();
        queued_bytes_ -= drained;
        written_bytes_ += written;
    }
}

nlohmann::json ExpertRouteRecorder::snapshot(bool totals) const {
    nlohmann::json body = {{"object", "expert_utilization"}};
    {
        std::lock_guard lock(mutex_);
        if (!geometry_) {
            body["enabled"] = false;
            return body;
        }
        const std::uint32_t experts = geometry_->experts;
        // Decay is applied on the next record; scale the copy to now for an idle server.
        const double idle =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - last_decay_).count();
        const double factor = std::pow(0.5, idle / kHeatHalfLifeSeconds);
        std::vector<double> decode_heat(decode_.decayed);
        std::vector<double> prefill_heat(prefill_.decayed);
        for (double& value : decode_heat) { value *= factor; }
        for (double& value : prefill_heat) { value *= factor; }
        body["enabled"]  = true;
        body["geometry"] = {
            {"text_layers", geometry_->text_layers},
            {"mtp_layer", geometry_->mtp_layer ? nlohmann::json(*geometry_->mtp_layer) : nullptr},
            {"experts", experts},
            {"top_k", geometry_->top_k}};
        body["half_life_seconds"] = kHeatHalfLifeSeconds;
        body["uptime_seconds"] =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started_).count();
        nlohmann::json records = nlohmann::json::object();
        nlohmann::json columns = nlohmann::json::object();
        for (std::size_t kind = 1; kind < 5; ++kind) {
            records[kKindNames[kind]] = records_[kind];
            columns[kKindNames[kind]] = columns_[kind];
        }
        body["records"] = std::move(records);
        body["columns"] = std::move(columns);
        body["decode"]  = {{"heat", heat_rows(decode_heat, slices_, experts)},
                           {"experts_for_80_percent", coverage_80(decode_heat, slices_, experts)}};
        body["prefill"] = {{"heat", heat_rows(prefill_heat, slices_, experts)}};
        if (totals) {
            body["decode"]["total"]  = total_rows(decode_.total, slices_, experts);
            body["prefill"]["total"] = total_rows(prefill_.total, slices_, experts);
        }
    }
    std::lock_guard lock(queue_mutex_);
    if (directory_.empty()) {
        body["trace"] = nullptr;
    } else {
        body["trace"] = {{"file", file_path_.filename().string()},
                         {"bytes_written", written_bytes_},
                         {"bytes_queued", queued_bytes_},
                         {"max_bytes", max_file_bytes_},
                         {"records_dropped", dropped_records_}};
    }
    return body;
}

} // namespace ninfer::serve
