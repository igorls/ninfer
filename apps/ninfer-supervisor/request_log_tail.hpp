#pragma once

// Follows the engine's request log as it grows, reading only what was appended since the last
// poll. The log is appended for the life of an installation and reaches gigabytes; reading it
// whole on every dashboard request took most of a minute and gigabytes of memory per call.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::supervisor {

// Which file a path named when it was opened. A log that was deleted and recreated, or replaced
// by a rename, has a different identity even after it has grown past the old read offset.
struct FileIdentity {
    std::uint64_t volume  = 0;
    std::uint64_t id_low  = 0;
    std::uint64_t id_high = 0;
    bool operator==(const FileIdentity&) const = default;
};

// One file generation: the first file seen, or one that replaced it. The tail window is the
// newest complete lines at the moment the generation was first seen, handed over before any
// history so recent figures are right immediately. seed_begin..seed_end is the byte range of
// those lines; a line that begins before seed_begin is older than the window, one that begins
// at or after seed_end was appended after the generation was first seen.
struct RequestLogGeneration {
    std::uint64_t number     = 0;
    std::uint64_t seed_begin = 0;
    std::uint64_t seed_end   = 0;
};

enum class RequestLogStatus { NotPolled, Missing, Unreadable, Ok };

struct RequestLogPoll {
    RequestLogStatus status = RequestLogStatus::NotPolled;
    unsigned long error     = 0;  // GetLastError() when Unreadable
    std::uint64_t size      = 0;  // the file's size when this poll opened it
    // Bytes the history reader has consumed: every complete line before this offset has been
    // handed over. At most size; an unfinished last line stays unconsumed until it ends.
    std::uint64_t history_offset = 0;
    // The history reader has caught up with `size` (an unfinished last line aside).
    bool history_at_end = false;
};

// Two readers over one file. The history reader starts at byte 0 and hands over every complete
// line in order, a bounded amount per poll, so an existing log is caught up in the background
// and then followed. The live reader starts where the file ended when it was first seen and hands
// over only what is appended after that, every poll, so live figures do not wait for the history.
//
// A sink provides:
//   void begin(const RequestLogGeneration&)  a new file: drop everything derived from the old one
//   void seed(std::string_view line)         the tail window's lines, oldest first, after begin
//   void live(std::string_view line)         lines appended after the generation was first seen
//   void history(std::string_view line, std::uint64_t begin_offset)  every line from byte 0
//   void gone()                              the file no longer exists
// Lines arrive without their "\n" or "\r\n".
class RequestLogTail {
public:
    static constexpr std::uint64_t kDefaultSeedWindow = 8ull << 20;
    // A line longer than this is not a request-log record; it is skipped, not buffered.
    static constexpr std::uint64_t kDefaultMaxLine = 16ull << 20;
    static constexpr std::size_t kDefaultBlock     = 1u << 20;

    explicit RequestLogTail(std::uint64_t seed_window = kDefaultSeedWindow,
                            std::uint64_t max_line    = kDefaultMaxLine,
                            std::size_t block         = kDefaultBlock)
        : seed_window_(seed_window), max_line_(max_line), block_(std::max<std::size_t>(block, 1)) {}

    // Opens the file once, checks that it is still the same one, hands over newly appended lines
    // to the live reader and up to about `history_budget` bytes of lines to the history reader.
    template <class Sink>
    RequestLogPoll poll(const std::wstring& path, std::uint64_t history_budget, Sink& sink) {
        RequestLogPoll out;
        const HANDLE h = CreateFileW(path.c_str(), GENERIC_READ,
                                     FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                     OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            const DWORD err = GetLastError();
            if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
                if (generation_.number != 0 && open_) {
                    open_ = false;
                    sink.gone();
                }
                out.status = RequestLogStatus::Missing;
                return out;
            }
            // Locked or denied for now: keep everything and try again next poll.
            out.status         = RequestLogStatus::Unreadable;
            out.error          = err;
            out.history_offset = history_.offset;
            return out;
        }
        struct Closer {
            HANDLE h;
            ~Closer() { CloseHandle(h); }
        } closer{h};

        FileIdentity id{};
        LARGE_INTEGER size_li{};
        if (!identity_of(h, id) || !GetFileSizeEx(h, &size_li)) {
            out.status         = RequestLogStatus::Unreadable;
            out.error          = GetLastError();
            out.history_offset = history_.offset;
            return out;
        }
        const auto size = static_cast<std::uint64_t>(size_li.QuadPart);
        // Replaced (new identity) or truncated (shorter than what was read): a new generation.
        // Reading on from a stale offset would splice records in half.
        if (open_ && (!(id == identity_) || size < std::max(history_.offset, live_.offset))) {
            open_ = false;
        }
        if (!open_) {
            identity_ = id;
            open_     = true;
            begin_generation(h, size, sink);
        }
        out.status = RequestLogStatus::Ok;
        out.size   = size;
        read_lines(h, live_, size, ~std::uint64_t{0},
                   [&](std::string_view line, std::uint64_t) { sink.live(line); });
        out.history_at_end = read_lines(h, history_, size, history_budget,
                                        [&](std::string_view line, std::uint64_t begin) {
                                            sink.history(line, begin);
                                        });
        out.history_offset = history_.offset;
        return out;
    }

    [[nodiscard]] const RequestLogGeneration& generation() const noexcept { return generation_; }
    // Lines over the length limit that were skipped, for diagnostics and tests.
    [[nodiscard]] std::uint64_t oversized_lines() const noexcept { return oversized_; }

private:
    struct Reader {
        std::uint64_t offset = 0;  // every complete line before this byte has been handed over
        bool discarding      = false;  // inside an oversized line: skip to its newline
    };

    static bool identity_of(HANDLE h, FileIdentity& id) {
        FILE_ID_INFO info{};
        if (GetFileInformationByHandleEx(h, FileIdInfo, &info, sizeof(info))) {
            id.volume = info.VolumeSerialNumber;
            std::memcpy(&id.id_low, info.FileId.Identifier, 8);
            std::memcpy(&id.id_high, info.FileId.Identifier + 8, 8);
            return true;
        }
        BY_HANDLE_FILE_INFORMATION bh{};
        if (!GetFileInformationByHandle(h, &bh)) { return false; }
        id.volume  = bh.dwVolumeSerialNumber;
        id.id_low  = (static_cast<std::uint64_t>(bh.nFileIndexHigh) << 32) | bh.nFileIndexLow;
        id.id_high = 0;
        return true;
    }

    // Reads up to `want` bytes at `offset` (fewer at end of file). Returns bytes read.
    static std::size_t read_at(HANDLE h, std::uint64_t offset, char* buf, std::size_t want) {
        std::size_t got = 0;
        while (got < want) {
            OVERLAPPED ov{};
            const std::uint64_t at = offset + got;
            ov.Offset     = static_cast<DWORD>(at & 0xFFFFFFFFu);
            ov.OffsetHigh = static_cast<DWORD>(at >> 32);
            DWORD n       = 0;
            const auto ask = static_cast<DWORD>(std::min<std::size_t>(want - got, 1u << 30));
            if (!ReadFile(h, buf + got, ask, &n, &ov) || n == 0) { break; }
            got += n;
        }
        return got;
    }

    static std::string_view strip_cr(std::string_view line) {
        if (!line.empty() && line.back() == '\r') { line.remove_suffix(1); }
        return line;
    }

    template <class Sink>
    void begin_generation(HANDLE h, std::uint64_t size, Sink& sink) {
        RequestLogGeneration g;
        g.number = generation_.number + 1;
        const std::uint64_t window_begin = size > seed_window_ ? size - seed_window_ : 0;
        std::string buf(static_cast<std::size_t>(size - window_begin), '\0');
        buf.resize(read_at(h, window_begin, buf.data(), buf.size()));
        // The window's first line is whole only when the window starts the file. With no newline
        // at all, a window that does not start the file lies inside one line begun before it.
        std::size_t first       = 0;
        bool inside_older_line  = false;
        if (window_begin > 0) {
            const auto nl     = buf.find('\n');
            inside_older_line = nl == std::string::npos;
            first             = inside_older_line ? buf.size() : nl + 1;
        }
        const auto last_nl     = buf.rfind('\n');
        const std::size_t last = (last_nl == std::string::npos || inside_older_line)
                                     ? first
                                     : std::max(first, last_nl + 1);
        g.seed_begin = window_begin + first;
        g.seed_end   = window_begin + last;
        generation_  = g;
        history_     = Reader{};
        // The live reader starts after the window. Inside an older line it skips to that line's
        // end: the line is history, not something appended now.
        live_ = Reader{g.seed_end, inside_older_line};
        sink.begin(g);
        std::string_view rest(buf.data() + first, last - first);
        while (!rest.empty()) {
            const auto nl = rest.find('\n');
            sink.seed(strip_cr(rest.substr(0, nl)));
            rest.remove_prefix(nl + 1);
        }
    }

    // Hands over complete lines from r.offset up to `size`, stopping at a line boundary once
    // about `budget` bytes were consumed. An unfinished last line is left for a later poll.
    // Returns true when everything up to `size` was consumed except such a line.
    template <class OnLine>
    bool read_lines(HANDLE h, Reader& r, std::uint64_t size, std::uint64_t budget, OnLine&& on_line) {
        const std::uint64_t start = r.offset;
        std::uint64_t pos         = r.offset;  // next byte to read
        std::string carry;                     // the unfinished line read so far, from r.offset
        std::vector<char> block(block_);
        while (pos < size) {
            const auto want = static_cast<std::size_t>(std::min<std::uint64_t>(block_, size - pos));
            const std::size_t got = read_at(h, pos, block.data(), want);
            if (got == 0) { return false; }  // a failed read is retried next poll
            std::string_view data(block.data(), got);
            std::size_t from = 0;
            while (from < data.size()) {
                const auto nl = data.find('\n', from);
                if (nl == std::string_view::npos) {
                    if (r.discarding) {
                        r.offset = pos + got;  // the skipped line continues past this block
                    } else {
                        carry.append(data.substr(from));
                        if (carry.size() > max_line_) {
                            ++oversized_;
                            carry.clear();
                            r.discarding = true;
                            r.offset     = pos + got;
                        }
                    }
                    break;
                }
                if (r.discarding) {
                    r.discarding = false;
                } else if (carry.empty()) {
                    on_line(strip_cr(data.substr(from, nl - from)), r.offset);
                } else {
                    carry.append(data.substr(from, nl - from));
                    if (carry.size() > max_line_) {
                        ++oversized_;
                    } else {
                        on_line(strip_cr(carry), r.offset);
                    }
                    carry.clear();
                }
                r.offset = pos + nl + 1;
                from     = nl + 1;
                // Budget spent: stop at this line boundary. The rest of the block is read
                // again by the next poll.
                if (r.offset - start >= budget) { return r.offset >= size; }
            }
            pos += got;
            if (r.discarding && r.offset - start >= budget) { return r.offset >= size; }
        }
        return true;
    }

    std::uint64_t seed_window_;
    std::uint64_t max_line_;
    std::size_t block_;
    FileIdentity identity_{};
    bool open_ = false;
    RequestLogGeneration generation_{};
    Reader history_{};
    Reader live_{};
    std::uint64_t oversized_ = 0;
};

} // namespace ninfer::supervisor
