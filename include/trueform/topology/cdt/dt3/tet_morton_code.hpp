/*
 * Copyright (c) 2026 XLAB
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

namespace tf::topology::cdt::dt3 {

/// The bits of a coordinate one code carries, so three of them occupy 63 of
/// the 64 a code is stated in.
inline constexpr unsigned tet_morton_bits = 21;

constexpr auto make_tet_morton_byte_spread() -> std::array<std::uint32_t, 256> {
  std::array<std::uint32_t, 256> spread{};
  for (std::size_t byte = 0; byte < spread.size(); ++byte) {
    std::uint32_t expanded = 0;
    for (unsigned bit = 0; bit < 8; ++bit) {
      const auto present = std::uint32_t((byte >> bit) & std::size_t(1));
      expanded |= present << (3U * bit);
    }
    spread[byte] = expanded;
  }
  return spread;
}

inline constexpr auto tet_morton_byte_spread = make_tet_morton_byte_spread();

/// The three coordinates interleaved, x in the lowest bit of every triple.
/// A coordinate past @ref tet_morton_bits is truncated, which a scheduling
/// order may be and an identity may not.
inline constexpr auto tet_morton_code(std::uint32_t x, std::uint32_t y,
                                      std::uint32_t z) -> std::uint64_t {
  std::uint64_t code = 0;
  for (unsigned byte = 0; byte < 3; ++byte) {
    const auto shift = 8U * byte;
    const auto spread_x = std::uint64_t(
        tet_morton_byte_spread[std::size_t((x >> shift) & 0xffU)]);
    const auto spread_y = std::uint64_t(
        tet_morton_byte_spread[std::size_t((y >> shift) & 0xffU)]);
    const auto spread_z = std::uint64_t(
        tet_morton_byte_spread[std::size_t((z >> shift) & 0xffU)]);
    code |= (spread_x | (spread_y << 1U) | (spread_z << 2U)) << (24U * byte);
  }
  return code;
}

} // namespace tf::topology::cdt::dt3
