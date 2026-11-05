#pragma once

#include <string_view>

namespace ninfer::supervisor {
// Borrows the immutable RCDATA bytes embedded in this executable.
[[nodiscard]] std::string_view dashboard_html();
} // namespace ninfer::supervisor
