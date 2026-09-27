#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <system_error>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

constexpr std::size_t kMaximumReadBytes = 64ULL * 1024 * 1024;

#if defined(_WIN32)

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    const auto error = static_cast<int>(::GetLastError());
    throw ArtifactError(path.string() + ": " + operation + ": " +
                        std::system_category().message(error));
}

HANDLE handle(std::intptr_t file) noexcept { return reinterpret_cast<HANDLE>(file); }

// Full sharing keeps POSIX semantics: an open artifact does not lock the file against other
// writers, renames or deletion.
std::intptr_t open_file(const std::filesystem::path& path, DWORD flags, const char* operation) {
    const HANDLE file = ::CreateFileW(
        path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        OPEN_EXISTING, flags, nullptr);
    if (file == INVALID_HANDLE_VALUE) { fail(path, operation); }
    return reinterpret_cast<std::intptr_t>(file);
}

void close_file(std::intptr_t file) noexcept { ::CloseHandle(handle(file)); }

// One positional read of at most kMaximumReadBytes; 0 means end of file.
std::size_t read_at(std::intptr_t file, const std::filesystem::path& path, std::uint64_t offset,
                    std::span<std::byte> destination, const char* operation) {
    OVERLAPPED position{};
    position.Offset     = static_cast<DWORD>(offset);
    position.OffsetHigh = static_cast<DWORD>(offset >> 32U);
    DWORD read          = 0;
    const auto count    = static_cast<DWORD>(std::min(destination.size(), kMaximumReadBytes));
    if (!::ReadFile(handle(file), destination.data(), count, &read, &position)) {
        if (::GetLastError() == ERROR_HANDLE_EOF) { return 0; }
        fail(path, operation);
    }
    return read;
}

#else

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

std::intptr_t open_file(const std::filesystem::path& path, int flags, const char* operation) {
    const int file = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | flags);
    if (file < 0) { fail(path, operation); }
    return file;
}

void close_file(std::intptr_t file) noexcept { ::close(static_cast<int>(file)); }

// One positional read of at most kMaximumReadBytes; 0 means end of file.
std::size_t read_at(std::intptr_t file, const std::filesystem::path& path, std::uint64_t offset,
                    std::span<std::byte> destination, const char* operation) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    const auto count = std::min(destination.size(), kMaximumReadBytes);
    ssize_t read;
    do {
        read = ::pread(static_cast<int>(file), destination.data(), count,
                       static_cast<off_t>(offset));
    } while (read < 0 && errno == EINTR);
    if (read < 0) { fail(path, operation); }
    return static_cast<std::size_t>(read);
}

#endif

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
#if defined(_WIN32)
    file_ = open_file(path_, FILE_ATTRIBUTE_NORMAL, "open");
    LARGE_INTEGER size{};
    if (::GetFileType(handle(file_)) != FILE_TYPE_DISK || !::GetFileSizeEx(handle(file_), &size) ||
        size.QuadPart < 0) {
        close_file(file_);
        file_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(size.QuadPart);
#else
    file_ = open_file(path_, 0, "open");

    struct stat status {};

    if (::fstat(static_cast<int>(file_), &status) != 0) {
        const auto error = errno;
        close_file(file_);
        file_ = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        close_file(file_);
        file_ = -1;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
#endif
}

InputFile::~InputFile() {
    if (direct_file_ >= 0) { close_file(direct_file_); }
    if (file_ >= 0) { close_file(file_); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto read = read_at(file_, path_, offset, destination, "read");
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += read;
        destination = destination.subspan(read);
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_file_ < 0) {
#if defined(_WIN32)
        direct_file_ = open_file(path_, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
                                 "open direct");
#else
        direct_file_ = open_file(path_, O_DIRECT, "open direct");
#endif
    }
    // Aligned chunks keep every read on the unbuffered path; only end of file ends it early.
    std::size_t total = 0;
    while (total < destination.size()) {
        const auto read =
            read_at(direct_file_, path_, offset + total, destination.subspan(total), "direct read");
        total += read;
        if (read == 0 || read % kPayloadAlignment != 0) { break; }
    }
    return total;
}

} // namespace ninfer::artifact
