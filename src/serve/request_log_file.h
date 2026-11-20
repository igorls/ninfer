#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string_view>

namespace ninfer::serve {

enum class RequestLogWrite { Written, Oversized, Failed };

// The request logger serializes access. A nonzero limit retains complete recent lines
// and the running instance's startup metadata when pruning the file.
class RequestLogFile {
public:
    RequestLogFile(const std::filesystem::path& path, std::uint64_t max_bytes,
                   const std::filesystem::path& protected_artifact_path = {});

    [[nodiscard]] bool configured() const noexcept { return !path_.empty(); }
    // Both startup records are written consecutively, before any other events in an instance.
    RequestLogWrite append(std::string_view record, bool startup = false);

private:
    bool prune(std::uint64_t tail_budget);

    std::filesystem::path path_;
    std::filesystem::path staging_;
    std::ofstream output_;
    std::uint64_t max_bytes_;
    std::uint64_t size_ = 0;
    std::uint64_t metadata_begin_ = 0;
    std::uint64_t metadata_end_   = 0;
    bool incomplete_line_ = false;
};

} // namespace ninfer::serve
