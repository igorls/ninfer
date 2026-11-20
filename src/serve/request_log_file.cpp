#include "serve/request_log_file.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <cstddef>
#include <cstring>
#include <memory>
#endif

namespace ninfer::serve {
namespace {

std::filesystem::path normalized_absolute_path(const std::filesystem::path& value) {
    std::error_code error;
    auto path = std::filesystem::weakly_canonical(value, error);
    if (!error) return path;
    path = std::filesystem::absolute(value, error);
    return error ? value.lexically_normal() : path.lexically_normal();
}

bool same_file(const std::filesystem::path& a, const std::filesystem::path& b) {
    std::error_code error;
    return std::filesystem::equivalent(a, b, error) ||
           normalized_absolute_path(a) == normalized_absolute_path(b);
}

bool replace_log(const std::filesystem::path& staged, const std::filesystem::path& path) {
#ifdef _WIN32
    // A dashboard tail may still be reading the old file. Replace its name while allowing
    // those delete-sharing handles to finish reading their own file generation.
    const auto& name = path.native();
    const auto name_bytes = name.size() * sizeof(wchar_t);
    const auto terminated_bytes = name_bytes + sizeof(wchar_t);
    const auto name_offset = offsetof(FILE_RENAME_INFO, FileName);
    const auto info_size = std::max(sizeof(FILE_RENAME_INFO), name_offset + terminated_bytes);
    const auto buffer = std::make_unique<std::byte[]>(info_size);
    auto* info = std::construct_at(reinterpret_cast<FILE_RENAME_INFO*>(buffer.get()));
    info->Flags = FILE_RENAME_FLAG_REPLACE_IF_EXISTS | FILE_RENAME_FLAG_POSIX_SEMANTICS;
    info->FileNameLength = static_cast<DWORD>(name_bytes);
    std::memcpy(buffer.get() + name_offset, name.c_str(), terminated_bytes);

    const HANDLE file = CreateFileW(staged.c_str(), DELETE,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                    nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    const bool ok = SetFileInformationByHandle(file, FileRenameInfoEx, info,
                                              static_cast<DWORD>(info_size)) != FALSE;
    CloseHandle(file);
    return ok;
#else
    std::error_code error;
    std::filesystem::rename(staged, path, error);
    return !error;
#endif
}

} // namespace

RequestLogFile::RequestLogFile(const std::filesystem::path& path, std::uint64_t max_bytes,
                               const std::filesystem::path& protected_artifact_path)
    : path_(path), max_bytes_(max_bytes) {
    if (path_.empty()) return;
    if (max_bytes_) {
        path_ = normalized_absolute_path(path_);
        staging_ = path_;
        staging_ += ".prune.tmp";
    }
    if (!protected_artifact_path.empty() &&
        (same_file(path_, protected_artifact_path) ||
         (max_bytes_ && same_file(staging_, protected_artifact_path)))) {
        throw std::invalid_argument("request JSONL log must not overwrite the model artifact");
    }
    output_.open(path_, max_bytes_ ? std::ios::binary | std::ios::app : std::ios::app);
    if (!output_) {
        throw std::runtime_error("failed to open request JSONL log for append: " + path.string());
    }
    if (!max_bytes_) return;

    size_ = std::filesystem::file_size(path_);
    if (size_) {
        std::ifstream input(path_, std::ios::binary);
        input.seekg(-1, std::ios::end);
        const int last = input.get();
        if (last == std::char_traits<char>::eof()) {
            throw std::runtime_error("failed to read request JSONL log: " + path.string());
        }
        incomplete_line_ = last != '\n';
    }
}

RequestLogWrite RequestLogFile::append(std::string_view record, bool startup) {
    const std::uint64_t record_size = record.size() + 1;
    if (max_bytes_) {
        const std::uint64_t available = max_bytes_ - (metadata_end_ - metadata_begin_);
        const bool oversized = record_size > available;
        const auto pending = oversized ? 0 : record_size;
        if ((incomplete_line_ || size_ > max_bytes_ - pending) &&
            !prune(std::min(max_bytes_ / 4, available - pending))) {
            return RequestLogWrite::Failed;
        }
        if (oversized) return RequestLogWrite::Oversized;
    }

    output_ << record << '\n';
    output_.flush();
    if (!output_) return RequestLogWrite::Failed;
    if (max_bytes_) {
        if (startup && metadata_end_ == 0) metadata_begin_ = size_;
        size_ += record_size;
        if (startup) metadata_end_ = size_;
    }
    return RequestLogWrite::Written;
}

bool RequestLogFile::prune(std::uint64_t tail_budget) {
    const auto begin = std::max(metadata_end_, size_ - std::min(size_, tail_budget));
    const bool trim_first = begin > metadata_end_ && begin < size_;
    const auto read_begin = trim_first ? begin - 1 : begin;
    std::string metadata(static_cast<std::size_t>(metadata_end_ - metadata_begin_), '\0');
    std::string bytes(static_cast<std::size_t>(size_ - read_begin), '\0');
    {
        std::ifstream input(path_, std::ios::binary);
        if (!metadata.empty()) {
            input.seekg(static_cast<std::streamoff>(metadata_begin_));
            input.read(metadata.data(), static_cast<std::streamsize>(metadata.size()));
        }
        if (!bytes.empty()) {
            input.seekg(static_cast<std::streamoff>(read_begin));
            input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        if (!input) return false;
    }

    // Read one extra byte so a newline-aligned cutoff keeps its first complete line.
    std::string_view tail(bytes);
    if (trim_first) {
        const auto newline = tail.find('\n');
        tail.remove_prefix(newline == tail.npos ? tail.size() : newline + 1);
    }
    const auto newline = tail.rfind('\n');
    tail = newline == tail.npos ? std::string_view{} : tail.substr(0, newline + 1);

    // The sibling is reserved for this writer. Reject links and other non-file entries.
    // Discard a stale regular file left by an interruption before staging new data.
    std::error_code error;
    const auto staging_status = std::filesystem::symlink_status(staging_, error);
    if (error && error != std::errc::no_such_file_or_directory) return false;
    if (std::filesystem::exists(staging_status) && !std::filesystem::is_regular_file(staging_status)) {
        return false;
    }
    std::filesystem::remove(staging_, error);
    if (error) return false;
    std::ofstream staged(staging_, std::ios::binary | std::ios::trunc);
    if (!staged) return false;
    staged << metadata << tail;
    staged.close();
    if (staged) {
        output_.close();
        if (replace_log(staging_, path_)) {
            metadata_begin_ = 0;
            metadata_end_ = metadata.size();
            size_ = metadata.size() + tail.size();
            incomplete_line_ = false;
            output_.open(path_, std::ios::binary | std::ios::app);
            return static_cast<bool>(output_);
        }
    }
    std::filesystem::remove(staging_, error);
    return false;
}

} // namespace ninfer::serve
