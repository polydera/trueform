/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt {

constexpr auto make_morton_byte_expansion_table()
    -> std::array<std::uint16_t, 256> {
  std::array<std::uint16_t, 256> expansion{};
  for (std::size_t byte = 0; byte < expansion.size(); ++byte) {
    std::uint16_t expanded = 0;
    for (unsigned bit = 0; bit < 8; ++bit) {
      const auto present = (byte >> bit) & std::size_t(1);
      expanded |= static_cast<std::uint16_t>(present << (2U * bit));
    }
    expansion[byte] = expanded;
  }
  return expansion;
}

inline constexpr auto morton_byte_expansion =
    make_morton_byte_expansion_table();

inline constexpr auto morton_code(std::uint32_t x, std::uint32_t y)
    -> std::uint32_t {
  const auto x_low =
      std::uint32_t(morton_byte_expansion[std::size_t(x & 0xffU)]);
  const auto y_low =
      std::uint32_t(morton_byte_expansion[std::size_t(y & 0xffU)]);
  const auto x_high =
      std::uint32_t(morton_byte_expansion[std::size_t((x >> 8U) & 0xffU)]);
  const auto y_high =
      std::uint32_t(morton_byte_expansion[std::size_t((y >> 8U) & 0xffU)]);
  const auto low = x_low | (y_low << 1U);
  const auto high = x_high | (y_high << 1U);
  return low | (high << 16U);
}

} // namespace tf::topology::cdt
