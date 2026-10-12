#pragma once

#include "ninfer/types.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace ninfer::serve {

// Engine routed-expert observer of ninfer-serve. It keeps live per-layer expert counts for
// GET /admin/experts and, with a trace directory, appends every record to
// expert-trace-<server instance>.nxtr (format: tools/bench/expert_trace/README.md). A prefill
// chunk with more ids than one count per expert is stored as a per-layer histogram.
//
// Records arrive on the Engine worker; the file is written by a background thread through a
// bounded queue. A full queue or a file at its byte limit drops records (counted) instead of
// delaying the Engine.
class ExpertRouteRecorder final : public ExpertRouteObserver {
public:
    static constexpr std::size_t kQueueLimitBytes = 64ULL * 1024 * 1024;
    static constexpr double kHeatHalfLifeSeconds  = 60.0;

    // An empty directory keeps live counts only; a trace file stops growing at max_file_bytes.
    explicit ExpertRouteRecorder(std::filesystem::path trace_directory,
                                 std::uint64_t max_file_bytes = UINT64_MAX);
    ~ExpertRouteRecorder() override;

    ExpertRouteRecorder(const ExpertRouteRecorder&)            = delete;
    ExpertRouteRecorder& operator=(const ExpertRouteRecorder&) = delete;

    // Names the trace file and writes its header; records before this call are counted live
    // but not written.
    void configure(const std::string& server_instance_id, const std::string& model_id,
                   const std::string& artifact_path);

    void expert_route_started(const ExpertRouteGeometry& geometry) override;
    void expert_route(const ExpertRouteRecord& record) noexcept override;

    // Live counts and trace status. `totals` adds the cumulative counts since start.
    [[nodiscard]] nlohmann::json snapshot(bool totals) const;

private:
    struct Heat {
        std::vector<double> decayed;      // [slice][expert]
        std::vector<std::uint64_t> total; // [slice][expert]
    };

    void decay(std::chrono::steady_clock::time_point now);
    void enqueue(std::vector<std::byte>&& bytes);
    void write_loop();

    const std::filesystem::path directory_;
    const std::uint64_t max_file_bytes_;
    mutable std::mutex mutex_;
    std::optional<ExpertRouteGeometry> geometry_;
    std::uint32_t slices_ = 0;
    std::atomic<std::uint32_t> experts_per_layer_{0};
    Heat decode_;
    Heat prefill_;
    std::uint64_t records_[5]{};
    std::uint64_t columns_[5]{};
    std::chrono::steady_clock::time_point started_;
    std::chrono::steady_clock::time_point last_decay_;

    mutable std::mutex queue_mutex_;
    std::condition_variable queue_ready_;
    std::deque<std::vector<std::byte>> queue_;
    std::size_t queued_bytes_      = 0;
    std::uint64_t dropped_records_ = 0;
    std::uint64_t written_bytes_   = 0;
    std::filesystem::path file_path_;
    std::FILE* file_ = nullptr;
    bool stopping_   = false;
    std::thread writer_;
};

} // namespace ninfer::serve
