#include "core/io/file.h"

#include <windows.h>

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace ninfer::io {
namespace {
HANDLE handle(FileView file) { return reinterpret_cast<HANDLE>(file.descriptor); }

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation,
                       DWORD error = GetLastError()) {
    throw std::system_error(static_cast<int>(error), std::system_category(),
                            path.string() + ": " + operation);
}

void position(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<LONGLONG>::max()))
        throw std::invalid_argument("file offset exceeds the Windows file domain");
}

std::size_t read(FileView file, std::uint64_t offset, std::span<std::byte> destination,
                 const std::filesystem::path& path) {
    if (destination.empty()) return 0;
    position(offset);
    OVERLAPPED operation{};
    operation.Offset     = static_cast<DWORD>(offset);
    operation.OffsetHigh = static_cast<DWORD>(offset >> 32);
    operation.hEvent     = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!operation.hEvent) fail(path, "create read event");
    DWORD received = 0;
    const bool issued = ReadFile(handle(file), destination.data(),
                                 static_cast<DWORD>(destination.size()), &received, &operation);
    DWORD error = issued ? ERROR_SUCCESS : GetLastError();
    if (error == ERROR_IO_PENDING) {
        error = GetOverlappedResult(handle(file), &operation, &received, TRUE)
                    ? ERROR_SUCCESS : GetLastError();
    }
    CloseHandle(operation.hEvent);
    if (error == ERROR_HANDLE_EOF) return 0;
    if (error != ERROR_SUCCESS) fail(path, "overlapped read", error);
    return received;
}
} // namespace

ReadOnlyFile::ReadOnlyFile(const std::filesystem::path& path, ReadMode mode) : path_(path) {
    if (mode != ReadMode::Buffered && mode != ReadMode::Direct)
        throw std::invalid_argument("invalid file read mode");
    const auto flags = FILE_FLAG_OVERLAPPED | (mode == ReadMode::Direct ? FILE_FLAG_NO_BUFFERING : 0);
    const auto file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE,
                                   nullptr, OPEN_EXISTING, flags, nullptr);
    if (file == INVALID_HANDLE_VALUE) fail(path, "open");
    view_.descriptor = reinterpret_cast<std::intptr_t>(file);
    view_.mode = mode;
    try {
        FILE_STANDARD_INFO status{};
        if (GetFileType(file) != FILE_TYPE_DISK ||
            !GetFileInformationByHandleEx(file, FileStandardInfo, &status, sizeof(status)))
            fail(path, "query regular file");
        if (status.Directory || status.EndOfFile.QuadPart < 0)
            fail(path, "expected regular file", ERROR_INVALID_DATA);
        view_.bytes = static_cast<std::uint64_t>(status.EndOfFile.QuadPart);
        if (mode == ReadMode::Direct) {
            FILE_STORAGE_INFO storage{};
            FILE_ALIGNMENT_INFO alignment{};
            if (!GetFileInformationByHandleEx(file, FileStorageInfo, &storage, sizeof(storage)) ||
                !GetFileInformationByHandleEx(file, FileAlignmentInfo, &alignment, sizeof(alignment)))
                fail(path, "query direct I/O alignment");
            view_.offset_alignment = storage.LogicalBytesPerSector;
            view_.memory_alignment = std::max<std::size_t>(
                {view_.offset_alignment, storage.PhysicalBytesPerSectorForPerformance,
                 std::size_t(alignment.AlignmentRequirement) + 1});
            if (!std::has_single_bit(view_.memory_alignment) ||
                !std::has_single_bit(view_.offset_alignment))
                fail(path, "unsupported direct I/O alignment", ERROR_NOT_SUPPORTED);
        }
    } catch (...) {
        CloseHandle(file);
        view_.descriptor = -1;
        throw;
    }
}

ReadOnlyFile::~ReadOnlyFile() {
    if (view_.descriptor != -1) CloseHandle(handle(view_));
}
ReadOnlyFile::ReadOnlyFile(ReadOnlyFile&& other) noexcept
    : path_(std::move(other.path_)), view_(std::exchange(other.view_, {})) {}
ReadOnlyFile& ReadOnlyFile::operator=(ReadOnlyFile&& other) noexcept {
    if (this != &other) {
        if (view_.descriptor != -1) CloseHandle(handle(view_));
        path_ = std::move(other.path_);
        view_ = std::exchange(other.view_, {});
    }
    return *this;
}

void ReadOnlyFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (view_.mode != ReadMode::Buffered || view_.descriptor == -1)
        throw std::invalid_argument("read_exact requires a buffered file");
    if (offset > view_.bytes || destination.size() > view_.bytes - offset)
        fail(path_, "read exceeds file length", ERROR_HANDLE_EOF);
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto n = read(view_, offset, destination.first(count), path_);
        if (!n) fail(path_, "unexpected EOF", ERROR_HANDLE_EOF);
        offset += n;
        destination = destination.subspan(n);
    }
}

void validate_direct_read(FileView file, std::uint64_t offset, std::span<std::byte> destination) {
    if (file.descriptor == -1 || file.mode != ReadMode::Direct ||
        !std::has_single_bit(file.memory_alignment) || !std::has_single_bit(file.offset_alignment) ||
        offset % file.offset_alignment || destination.size() % file.offset_alignment ||
        (!destination.empty() && (!destination.data() ||
         reinterpret_cast<std::uintptr_t>(destination.data()) % file.memory_alignment)) ||
        destination.size() > std::numeric_limits<DWORD>::max())
        throw std::invalid_argument("invalid direct read range or alignment");
    position(offset);
}

std::size_t ReadOnlyFile::read_direct(std::uint64_t offset,
                                      std::span<std::byte> destination) const {
    validate_direct_read(view_, offset, destination);
    return read(view_, offset, destination, path_);
}

DirectReadWindow direct_read_window(FileView file, std::uint64_t offset, std::size_t bytes) {
    if (file.descriptor == -1 || file.mode != ReadMode::Direct || !bytes ||
        !std::has_single_bit(file.offset_alignment) || offset > file.bytes || bytes > file.bytes - offset)
        throw std::invalid_argument("invalid direct read window");
    const auto alignment = file.offset_alignment;
    const auto prefix = offset % alignment;
    if (bytes > std::numeric_limits<std::size_t>::max() - prefix - (alignment - 1))
        throw std::overflow_error("direct read window size overflow");
    const auto required = bytes + prefix;
    const auto count = (required + alignment - 1) / alignment * alignment;
    position(offset - prefix);
    return {offset - prefix, count, prefix, required};
}
} // namespace ninfer::io
