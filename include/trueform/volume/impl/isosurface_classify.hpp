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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../volume.hpp"
#include "./field_value.hpp"
#include <array>
#include <cstdint>
#include <cstring>

namespace tf {
namespace volume_detail {

// The x-edge classification both isosurface extractors stand on: for every row
// of the volume, the 2-bit case of each x edge, the row's crossing count and
// the [min, max) span of its crossings. A corner is inside when `sample < iso`.
//
// 32 edges are classified per iteration: one inside-mask over the 33 samples
// they span, eight case bytes per store, and the crossing count and trim read
// off the mask instead of a branch per edge.

// Portable bit helpers. The library is C++17, so <bit> is unavailable; these
// idioms compile to the popcount/clz instructions where the target has them.
inline auto fe_popcount(std::uint32_t v) -> int {
  v = v - ((v >> 1) & 0x55555555u);
  v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
  return static_cast<int>((((v + (v >> 4)) & 0x0f0f0f0fu) * 0x01010101u) >> 24);
}

/// @brief Index of the highest set bit plus one; 0 when @p v is 0.
inline auto fe_bit_width(std::uint32_t v) -> int {
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return fe_popcount(v);
}

// Bit i of the index into byte i of the value, as 0 or 1. Small enough to state
// at compile time, so the inner loop reads it with no initialization guard.
constexpr auto make_spread_table() -> std::array<std::uint64_t, 256> {
  std::array<std::uint64_t, 256> t{};
  for (std::size_t m = 0; m < t.size(); ++m)
    for (int lane = 0; lane < 8; ++lane)
      t[m] |= static_cast<std::uint64_t>((m >> lane) & 1u) << (lane * 8);
  return t;
}

inline constexpr std::array<std::uint64_t, 256> k_spread_bits =
    make_spread_table();

/// @brief Bit i of @p b into byte i of the result, as 0 or 1.
inline auto fe_spread_bits(unsigned char b) -> std::uint64_t {
  return k_spread_bits[b];
}

// The eight-at-a-time classification writes its case bytes as one 64-bit word,
// which states the lane order only on a little-endian target. Elsewhere every
// row falls through to the same per-edge tail the chunk remainder uses.
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
inline constexpr bool fe_packed_stores =
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
#elif defined(_WIN32)
inline constexpr bool fe_packed_stores = true;
#else
inline constexpr bool fe_packed_stores = false;
#endif

/// @brief Classify the volume's x edges into @p xcase, one 2-bit case per edge,
/// and state each row's crossing count and trim.
///
/// @param meta Row metadata, @p meta_stride entries per row. The first four are
///        zeroed and the first receives the row's crossing count; the caller
///        owns the rest.
/// @param xmin,xmax The row's first crossing and one past its last, or
///        `nx > 0` and `0` when the row has none.
template <typename Real, typename Policy>
auto classify_x_edges(const tf::volume<Policy> &vol, Real iso,
                      unsigned char *xcase, std::int64_t *meta,
                      std::size_t meta_stride, int *xmin, int *xmax) -> void {
  const auto &dims = vol.dims();
  const int nx = dims[0], ny = dims[1], nz = dims[2];
  const int xedges = nx - 1;
  const auto samples = vol.begin();
  const auto row_index = [ny](int y, int z) {
    return static_cast<std::size_t>(z) * ny + y;
  };

  tf::parallel_for_each(tf::make_sequence_range(0, nz), [&](int z) {
    for (int y = 0; y < ny; ++y) {
      const auto row = row_index(y, z);
      unsigned char *xc = xcase + row * xedges;
      std::int64_t *m = meta + row * meta_stride;
      m[0] = m[1] = m[2] = m[3] = 0;
      int xl = nx; // no crossing sentinel (xl > xr)
      int xr = 0;
      std::int64_t crossings = 0;
      const auto s = samples + row * nx;
      int x = 0;
      if (fe_packed_stores) {
        for (; x + 32 <= xedges; x += 32) {
          std::uint32_t inside = 0;
          for (int lane = 0; lane < 32; ++lane)
            inside |= static_cast<std::uint32_t>(
                          field_value<Real>(s[x + lane]) < iso)
                      << lane;
          const std::uint32_t right =
              (inside >> 1) |
              (static_cast<std::uint32_t>(field_value<Real>(s[x + 32]) < iso)
               << 31);
          for (int lane = 0; lane < 32; lane += 8) {
            const std::uint64_t packed =
                fe_spread_bits(static_cast<unsigned char>(inside >> lane)) |
                (fe_spread_bits(static_cast<unsigned char>(right >> lane))
                 << 1);
            std::memcpy(xc + x + lane, &packed, sizeof(packed));
          }
          const std::uint32_t crossing = inside ^ right;
          if (crossing != 0) {
            crossings += fe_popcount(crossing);
            if (xl == nx)
              xl = x + fe_bit_width(crossing & (0u - crossing)) - 1;
            xr = x + fe_bit_width(crossing);
          }
        }
      }
      bool prev_in = field_value<Real>(s[x]) < iso;
      for (; x < xedges; ++x) {
        const bool next_in = field_value<Real>(s[x + 1]) < iso;
        const unsigned char cs =
            static_cast<unsigned char>((prev_in ? 1 : 0) | (next_in ? 2 : 0));
        xc[x] = cs;
        if (cs == 1 || cs == 2) {
          ++crossings;
          if (x < xl)
            xl = x;
          xr = x + 1;
        }
        prev_in = next_in;
      }
      m[0] = crossings;
      xmin[row] = xl;
      xmax[row] = xr;
    }
  });
}

} // namespace volume_detail
} // namespace tf
