#pragma once

#include "core/device.h"
#include "core/host_worker_pool.h"
#include "core/tensor.h"
#include "targets/qwen3_8_flash_next/impl/ple_table.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next::detail {

class PleGatherPipeline {
public:
    class Ticket {
    public:
        Ticket() noexcept = default;

    private:
        friend class PleGatherPipeline;
        Ticket(PleGatherPipeline* owner, std::size_t slot, std::uint64_t generation,
               std::size_t tokens) noexcept;

        PleGatherPipeline* owner_ = nullptr;
        std::size_t slot_         = 0;
        std::uint64_t generation_ = 0;
        std::size_t tokens_       = 0;
    };

    PleGatherPipeline(PleTableView table, DeviceContext& device, std::size_t max_tokens,
                      std::size_t slot_count = 2, std::uint32_t worker_threads = 2);
    ~PleGatherPipeline();

    PleGatherPipeline(const PleGatherPipeline&)            = delete;
    PleGatherPipeline& operator=(const PleGatherPipeline&) = delete;

    [[nodiscard]] Ticket prepare(std::span<const std::array<std::int64_t, 16>> global_rows);

    // Waits only for the host gather, enqueues pinned BF16 H2D on the transfer stream, and inserts
    // a dependency into the main stream. The main stream itself is never host-synchronized.
    void enqueue_copy(Ticket&& ticket, Tensor& output);

private:
    struct Slot;

    [[nodiscard]] std::size_t acquire_slot();

    PleTableView table_;
    DeviceContext& device_;
    std::size_t max_tokens_ = 0;
    std::vector<std::unique_ptr<Slot>> slots_;
    HostWorkerPool workers_;
    std::size_t next_slot_ = 0;
};

} // namespace ninfer::targets::qwen3_8_flash_next::detail
