#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>

namespace ninfer::artifact {

// Read-only mapping of one file range. It stays valid after its InputFile closes: POSIX mappings
// outlive their descriptor and Windows keeps the file open until the last view is unmapped.
// Pages are backed by the OS page cache; nothing is copied, pinned or locked.
class FileMapping {
public:
    FileMapping() noexcept = default;
    ~FileMapping();
    FileMapping(FileMapping&& other) noexcept;
    FileMapping& operator=(FileMapping&& other) noexcept;
    FileMapping(const FileMapping&)            = delete;
    FileMapping& operator=(const FileMapping&) = delete;

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept { return {data_, bytes_}; }

    // Reads the range ahead and faults in every page, so first use does not pay synchronous
    // file faults. Reclaim remains possible under memory pressure.
    void warm() const noexcept;

private:
    friend class InputFile;
    void release() noexcept;

    void* view_            = nullptr;
    std::size_t length_    = 0;
    const std::byte* data_ = nullptr;
    std::size_t bytes_     = 0;
};

// Direct reads require aligned offsets and buffers. A short final direct block is allowed;
// read_exact always requires the complete requested byte range.
class InputFile {
public:
    explicit InputFile(std::filesystem::path path);
    ~InputFile();
    InputFile(const InputFile&)            = delete;
    InputFile& operator=(const InputFile&) = delete;

    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

    void read_exact(std::uint64_t offset, std::span<std::byte> destination) const;
    [[nodiscard]] std::size_t read_direct(std::uint64_t offset,
                                          std::span<std::byte> destination) const;
    [[nodiscard]] FileMapping map(std::uint64_t offset, std::uint64_t bytes) const;

private:
    // POSIX file descriptors or Windows HANDLEs; -1 is invalid on both.
    std::filesystem::path path_;
    std::intptr_t file_                 = -1;
    mutable std::intptr_t direct_file_ = -1;
    std::uint64_t bytes_                = 0;
};

} // namespace ninfer::artifact
