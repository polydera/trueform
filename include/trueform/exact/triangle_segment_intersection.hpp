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

/// Triangle ABC (indices 0,1,2) and segment DE (indices 3,4).
/// Uses SoS orient3d for all intersection tests (sign convention must be
/// consistent). On intersection, computes orient3d volumes for barycentric
/// weights (absolute values only, so sign convention is irrelevant).
template <typename Index>
auto triangle_segment_intersect_point_sos(
    const std::array<vertex<Index>, 5> &vs) -> std::optional<pt3> {

  auto orient = [&](int p, int q, int r, int s) -> bool {
    return orient3d_sos(
        std::array<vertex<Index>, 4>{vs[p], vs[q], vs[r], vs[s]});
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
  auto vol_d = orient3d_value(vs[0].pt, vs[1].pt, vs[2].pt, vs[3].pt);
  auto vol_e = orient3d_value(vs[0].pt, vs[1].pt, vs[2].pt, vs[4].pt);
  auto abs_d = vol_d < 0 ? -vol_d : vol_d;
  auto abs_e = vol_e < 0 ? -vol_e : vol_e;
  auto sum = abs_d + abs_e;

  pt3 point;
  if (sum != 0) {
    for (int i = 0; i < 3; ++i)
      point[i] = static_cast<int32_t>(
          div_round(abs_e * int128(vs[3].pt[i]) + abs_d * int128(vs[4].pt[i]),
                    sum));
  } else {
    for (int i = 0; i < 3; ++i)
      point[i] = static_cast<int32_t>(
          (int64_t(vs[3].pt[i]) + int64_t(vs[4].pt[i])) / 2);
  }

  return point;
}

} // namespace tf::exact
