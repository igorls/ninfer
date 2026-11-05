#pragma once

#include <cstdint>
#include <string_view>

namespace ninfer::product {

// Exact budgets shared by startup configuration and the Supervisor's child editor.
// A bare integer means bytes or milliseconds; fractional units must resolve exactly.
[[nodiscard]] std::uint64_t parse_capacity_quantity(std::string_view text);
[[nodiscard]] std::uint32_t parse_duration_quantity(std::string_view text);

} // namespace ninfer::product
