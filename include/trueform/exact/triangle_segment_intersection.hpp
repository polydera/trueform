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

#include "./orient3d.hpp"
#include <array>
#include <optional>

namespace tf::exact {

/// Triangle ABC (indices 0,1,2) and segment DE (indices 3,4), every point
/// carried in one common positive multiple of the `Int` lattice.
///
/// Each orientation verdict is scale-invariant
/// (@ref tf::exact::orient3d_sos_scaled) and the barycentric weights are a
/// ratio of two volumes that carry the same cube of the scale, so the
/// crossing and its position along DE are the unscaled segment's. The
/// point comes back in the scaled space it was asked in, which leaves the
/// ORDER of several crossings of one segment — the only thing a caller
/// compares distances for — the unscaled order.
template <typename Int, typename Index, typename Coord>
auto triangle_segment_intersect_point_scaled_sos(
    const std::array<vertex<Index, Coord>, 5> &vs)
    -> std::optional<pt3<Coord>> {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;

  auto orient = [&](int p, int q, int r, int s) -> bool {
    const std::array<vertex<Index, Coord>, 4> t{vs[p], vs[q], vs[r], vs[s]};
    return orient3d_sos_scaled<Int>(t.data());
  };

  constexpr int a = 0, b = 1, c = 2, d = 3, e = 4;

  // Same-side test
  auto abcd = orient(a, b, c, d);
  auto abce = orient(a, b, c, e);
  if (abcd == abce)
    return std::nullopt;

  // Edge orientation checks
  auto dabe = orient(a, b, d, e);
  auto dbce = orient(b, c, d, e);
  if (dabe != dbce)
    return std::nullopt;

  auto dcae = !orient(a, c, d, e);
  if (dbce != dcae)
    return std::nullopt;

  // Intersection confirmed — compute point using volume ratios
  auto vol_d =
      orient3d_value_scaled<Int>(vs[0].pt, vs[1].pt, vs[2].pt, vs[3].pt);
  auto vol_e =
      orient3d_value_scaled<Int>(vs[0].pt, vs[1].pt, vs[2].pt, vs[4].pt);
  auto abs_d = vol_d < 0 ? -vol_d : vol_d;
  auto abs_e = vol_e < 0 ? -vol_e : vol_e;
  auto sum = abs_d + abs_e;

  pt3<Coord> point;
  if (sum != 0) {
    for (int i = 0; i < 3; ++i)
      point[i] = static_cast<Coord>(
          div_round(abs_e * T2(vs[3].pt[i]) + abs_d * T2(vs[4].pt[i]), sum));
  } else {
    for (int i = 0; i < 3; ++i)
      point[i] = static_cast<Coord>((T1(vs[3].pt[i]) + T1(vs[4].pt[i])) / 2);
  }

  return point;
}

/// Triangle ABC (indices 0,1,2) and segment DE (indices 3,4).
/// Uses SoS orient3d for all intersection tests (sign convention must be
/// consistent). On intersection, computes orient3d volumes for barycentric
/// weights (absolute values only, so sign convention is irrelevant).
template <typename Index, typename Int>
auto triangle_segment_intersect_point_sos(
    const std::array<vertex<Index, Int>, 5> &vs) -> std::optional<pt3<Int>> {
  return triangle_segment_intersect_point_scaled_sos<Int>(vs);
}

} // namespace tf::exact
