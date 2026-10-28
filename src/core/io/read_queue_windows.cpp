#include "core/io/read_queue.h"
#include "core/io/windows_calls.h"

#include <array>
#include <deque>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::io {
const std::error_category& read_error_category() noexcept { return std::system_category(); }
namespace {
[[noreturn]] void fail(const char* operation, DWORD error = GetLastError()) {
    throw std::system_error(static_cast<int>(error), std::system_category(), operation);
}
}

struct DirectReadQueue::Impl {
    enum class State { Free, Prepared, Issued, Completed };
    struct Slot {
        OVERLAPPED operation{};
        DirectRead request{};
        State state = State::Free;
    };
    HANDLE port = nullptr;
    bool closed = false;
    bool signalled = false;
    std::uint32_t capacity = 0;
    std::size_t pending = 0;
    std::vector<FileView> files;
    std::vector<HANDLE> handles;
    std::vector<MemorySpan> buffers;
    std::vector<Slot> slots;
    std::deque<ReadCompletion> ready;

    ~Impl() {
        for (auto file : handles) CloseHandle(file);
        if (port) CloseHandle(port);
    }

    void completed(std::uint32_t index, DWORD bytes, DWORD error) {
        auto& slot = slots[index];
        if (slot.state != State::Issued) throw std::logic_error("unmatched direct read completion");
        slot.state = State::Completed;
        if (error == ERROR_HANDLE_EOF) error = ERROR_SUCCESS;
        ready.push_back({slot.request.tag, index, error ? 0 : bytes, static_cast<int>(error)});
    }

    // A wake packet has no OVERLAPPED pointer. Every actual request gets exactly one packet.
    std::optional<ReadWake> receive(DWORD timeout) {
        DWORD bytes = 0;
        ULONG_PTR key = 0;
        OVERLAPPED* operation = nullptr;
        const bool ok = windows::completion(port, &bytes, &key, &operation, timeout);
        const auto error = ok ? ERROR_SUCCESS : GetLastError();
        if (!operation) {
            if (ok) return ReadWake::Signalled;
            if (error == WAIT_TIMEOUT) return {};
            fail("wait for direct read completion", error);
        }
        const auto address = reinterpret_cast<std::uintptr_t>(operation);
        const auto base = reinterpret_cast<std::uintptr_t>(slots.data());
        if (address < base || (address - base) % sizeof(Slot) ||
            (address - base) / sizeof(Slot) >= slots.size())
            throw std::logic_error("foreign direct read completion");
        const auto index = static_cast<std::uint32_t>((address - base) / sizeof(Slot));
        if (key != std::size_t(slots[index].request.file_index) + 1)
            throw std::logic_error("direct read completion file differs");
        completed(index, bytes, error);
        return ReadWake::Completion;
    }
};

DirectReadQueue::DirectReadQueue(std::span<const FileView> files,
                                 std::span<const MemorySpan> buffers, std::uint32_t capacity)
    : impl_(std::make_unique<Impl>()) {
    if (!capacity || capacity > buffers.size() || files.empty() ||
        files.size() > std::numeric_limits<int>::max() || buffers.size() > 65535)
        throw std::invalid_argument("invalid direct read queue capacity");
    auto& p = *impl_;
    p.capacity = capacity;
    p.files.assign(files.begin(), files.end());
    p.buffers.assign(buffers.begin(), buffers.end());
    p.slots.resize(buffers.size());
    for (std::size_t i = 0; i < buffers.size(); ++i) {
        const auto buffer = buffers[i];
        validate_memory(buffer);
        if (buffer.device != Device::cpu() || !buffer.bytes ||
            buffer.bytes > std::numeric_limits<DWORD>::max())
            throw std::invalid_argument("read queue requires nonempty Host buffers");
        for (const auto file : files)
            validate_direct_read(file, 0, {static_cast<std::byte*>(buffer.data), buffer.bytes});
        const auto begin = reinterpret_cast<std::uintptr_t>(buffer.data);
        for (std::size_t j = 0; j < i; ++j) {
            const auto other = reinterpret_cast<std::uintptr_t>(buffers[j].data);
            if (begin < other + buffers[j].bytes && other < begin + buffer.bytes)
                throw std::invalid_argument("read queue buffers overlap");
        }
    }
    p.port = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 1);
    if (!p.port) fail("initialize direct read queue");
    p.handles.reserve(files.size());
    for (std::size_t i = 0; i < files.size(); ++i) {
        // DuplicateHandle shares the file object and its IOCP association. A fresh file object
        // lets independent Programs borrow the same immutable source without sharing queues.
        const auto file = ReOpenFile(reinterpret_cast<HANDLE>(files[i].descriptor), GENERIC_READ,
                                      FILE_SHARE_READ | FILE_SHARE_DELETE,
                                      FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED);
        if (file == INVALID_HANDLE_VALUE) fail("reopen direct read file");
        p.handles.push_back(file);
        if (!CreateIoCompletionPort(file, p.port, i + 1, 0)) fail("associate direct read file");
    }
}

DirectReadQueue::~DirectReadQueue() {
    try { close(); drain(); } catch (...) { std::terminate(); }
}

ReadAdmission DirectReadQueue::try_enqueue(DirectRead request) {
    auto& p = *impl_;
    if (p.closed) return ReadAdmission::Closed;
    if (p.pending == p.capacity) return ReadAdmission::Full;
    if (request.file_index >= p.files.size() || request.buffer_index >= p.buffers.size() || !request.bytes)
        throw std::invalid_argument("invalid direct read request");
    auto& slot = p.slots[request.buffer_index];
    const auto buffer = p.buffers[request.buffer_index];
    if (slot.state != Impl::State::Free || request.bytes > buffer.bytes)
        throw std::invalid_argument("direct read buffer is busy or too small");
    validate_direct_read(p.files[request.file_index], request.file_offset,
                          {static_cast<std::byte*>(buffer.data), request.bytes});
    slot.operation = {};
    slot.operation.Offset = static_cast<DWORD>(request.file_offset);
    slot.operation.OffsetHigh = static_cast<DWORD>(request.file_offset >> 32);
    slot.request = request;
    slot.state = Impl::State::Prepared;
    ++p.pending;
    return ReadAdmission::Accepted;
}

void DirectReadQueue::submit() {
    auto& p = *impl_;
    for (std::uint32_t i = 0; i < p.slots.size(); ++i) {
        auto& slot = p.slots[i];
        if (slot.state != Impl::State::Prepared) continue;
        const auto& request = slot.request;
        const bool ok = windows::issue_read(p.handles[request.file_index],
                                             p.buffers[i].data, request.bytes, &slot.operation);
        const auto error = ok ? ERROR_SUCCESS : GetLastError();
        // These failures did not issue a writer. Retain prepared work so submit can be retried.
        if (error == ERROR_NOT_ENOUGH_MEMORY || error == ERROR_NO_SYSTEM_RESOURCES)
            fail("submit direct reads", error);
        slot.state = Impl::State::Issued;
        if (!ok && error != ERROR_IO_PENDING) p.completed(i, 0, error);
    }
}

std::size_t DirectReadQueue::poll(std::span<ReadCompletion> output) {
    auto& p = *impl_;
    std::size_t count = 0;
    while (count < output.size()) {
        if (p.ready.empty()) {
            const auto wake = p.receive(0);
            if (!wake) break;
            if (*wake == ReadWake::Signalled) {
                p.signalled = true;
                continue;
            }
        }
        output[count++] = p.ready.front();
        p.slots[p.ready.front().buffer_index].state = Impl::State::Free;
        p.ready.pop_front();
        --p.pending;
    }
    return count;
}

ReadWake DirectReadQueue::wait() {
    auto& p = *impl_;
    if (std::exchange(p.signalled, false)) return ReadWake::Signalled;
    if (!p.ready.empty()) return ReadWake::Completion;
    return *p.receive(INFINITE);
}
void DirectReadQueue::wake() {
    if (!windows::signal(impl_->port)) fail("wake direct read queue");
}
void DirectReadQueue::close() { impl_->closed = true; submit(); }
void DirectReadQueue::drain() {
    submit();
    std::array<ReadCompletion, 128> completions;
    while (impl_->pending) if (!poll(completions)) (void)wait();
}
std::size_t DirectReadQueue::outstanding() const noexcept { return impl_->pending; }
} // namespace ninfer::io
