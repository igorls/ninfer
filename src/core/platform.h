#pragma once

// Small host-platform queries shared by runtime, product and test code. Header-only, so any
// component can use them without a link dependency.

#include <cstdint>
#include <cstdio>
#include <ctime>

#if defined(_WIN32)
#include <io.h>
#include <process.h>
#else
#include <unistd.h>
#endif

namespace ninfer {

[[nodiscard]] inline std::uint64_t current_process_id() noexcept {
#if defined(_WIN32)
    return static_cast<std::uint64_t>(::_getpid());
#else
    return static_cast<std::uint64_t>(::getpid());
#endif
}

[[nodiscard]] inline bool stderr_is_terminal() noexcept {
#if defined(_WIN32)
    return ::_isatty(::_fileno(stderr)) != 0;
#else
    return ::isatty(STDERR_FILENO) == 1;
#endif
}

// Thread-safe calendar conversions (localtime_r/gmtime_r are POSIX; the CRT has *_s).
[[nodiscard]] inline std::tm local_calendar_time(std::time_t time) noexcept {
    std::tm result{};
#if defined(_WIN32)
    (void)::localtime_s(&result, &time);
#else
    (void)::localtime_r(&time, &result);
#endif
    return result;
}

[[nodiscard]] inline std::tm utc_calendar_time(std::time_t time) noexcept {
    std::tm result{};
#if defined(_WIN32)
    (void)::gmtime_s(&result, &time);
#else
    (void)::gmtime_r(&time, &result);
#endif
    return result;
}

} // namespace ninfer
