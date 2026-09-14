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

#include "./exact_lane.hpp"
#include "./wide_to_double.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::exact::door::pool {

/// One exact direction's place in the FIXED projective chart: the dominant
/// axis it is charted in, the two rounded ratios against it, and the sign
/// the whole plane must be aligned by to state them.
///
/// `sign` is zero for a direction that has none — the degenerate normal —
/// and such a record joins no cell.
struct chart_cell {
  std::array<int, 3> at{};
  int sign = 0;

  auto operator<(const chart_cell &o) const -> bool {
    for (std::size_t k = 0; k < 3; ++k)
      if (at[k] != o.at[k])
        return at[k] < o.at[k];
    return false;
  }
  auto operator==(const chart_cell &o) const -> bool {
    return at == o.at;
  }
  auto operator!=(const chart_cell &o) const -> bool { return !(*this == o); }
};

/// How many cells the chart holds at resolution `K`, and where one of them
/// stands in it.
///
/// The key is an axis and two quotients bounded by `K`, so the whole chart
/// is `3 (2K + 1)^2` positions however wide the lattice is and however many
/// directions occupy it. The index is lexicographic in the key's own
/// components, so walking the domain in order IS walking the keys in
/// canonical order.
inline auto chart_domain(int resolution) -> std::int64_t {
  const std::int64_t span = 2 * std::int64_t(resolution) + 1;
  return 3 * span * span;
}

inline auto chart_index_of(const chart_cell &at, int resolution) -> int {
  const int span = 2 * resolution + 1;
  return (at.at[0] * span + (at.at[1] + resolution)) * span +
         (at.at[2] + resolution);
}

/// The cell of an exact primitive normal at chart resolution `K`.
///
/// The dominant axis is the largest magnitude, least index on ties; the
/// plane is aligned so that coordinate is positive, and the OFFSET follows
/// that same sign wherever the aligned plane is used. The remaining two
/// axes, in ascending index order, are rounded to nearest at `K` steps:
///
///     q = floor((2 K n[axis] + n[a]) / (2 n[a])).
///
/// The division is the mathematical floor of a signed numerator over a
/// positive denominator, so an exact half tie goes toward positive infinity
/// and the grid is not sheared across the origin. Scaling, endpoint order
/// and the canonical sign choice all leave the key alone: one unoriented
/// direction is one key.
///
/// The numerator `2 K n + n[a]` occupies the normal's own width plus the
/// resolution's, so the key is stated on the COEFFICIENT rung wherever that
/// width proof holds and on the product rung where it does not; the rung
/// moves the price and not the answer. Only the proven quotient, bounded by
/// `K`, is narrowed to the key.
template <typename Int>
auto chart_cell_of(
    const std::array<typename exact_lane<Int>::coefficient_type, 3> &normal,
    int resolution) -> chart_cell {
  using coefficient_type = typename exact_lane<Int>::coefficient_type;
  using product_type = typename exact_lane<Int>::product_type;

  chart_cell record;
  const auto magnitude = [](const coefficient_type &v) {
    return v < coefficient_type(0) ? -v : v;
  };
  std::size_t axis = 0;
  for (std::size_t k = 1; k < 3; ++k)
    if (magnitude(normal[k]) > magnitude(normal[axis]))
      axis = k;
  if (normal[axis] == coefficient_type(0))
    return record;

  record.sign = normal[axis] > coefficient_type(0) ? 1 : -1;
  const coefficient_type dominant = magnitude(normal[axis]);
  const coefficient_type scale =
      coefficient_type(2) * coefficient_type(resolution);
  const bool narrow =
      wide_bit_width(scale) + wide_bit_width(dominant) < exact_lane<Int>::bound_bits;
  record.at[0] = int(axis);
  std::size_t held = 1;
  for (std::size_t k = 0; k < 3; ++k) {
    if (k == axis)
      continue;
    if (narrow) {
      const coefficient_type over =
          scale * coefficient_type(record.sign) * normal[k] + dominant;
      const coefficient_type step = coefficient_type(2) * dominant;
      coefficient_type cell = over / step;
      if (cell * step != over && over < coefficient_type(0))
        cell = cell - coefficient_type(1);
      record.at[held++] = static_cast<int>(cell);
      continue;
    }
    const product_type step = product_type(2) * product_type(dominant);
    const product_type over = product_type(scale) *
                                  product_type(record.sign) *
                                  product_type(normal[k]) +
                              product_type(dominant);
    product_type cell = over / step;
    if (cell * step != over && over < product_type(0))
      cell = cell - product_type(1);
    record.at[held++] = static_cast<int>(cell);
  }
  return record;
}

} // namespace tf::exact::door::pool
