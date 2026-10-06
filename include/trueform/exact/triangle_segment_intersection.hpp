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

#include "./meta.hpp"
#include "./orient3d.hpp"
#include <array>
#include <optional>

namespace tf::exact {

/// Where segment DE crosses triangle ABC, and which side of ABC the D end
/// stands on — `true` the side @ref tf::exact::orient3d_sos_scaled calls
/// positive. The side is the same-side test's own verdict, so a caller
/// reading it recomputes nothing; a raw determinant sign is not a
/// substitute, an accepted crossing having a zero one.
template <typename Coord> struct triangle_segment_crossing {
  pt3<Coord> point;
  bool d_on_positive_side;
};

/// Whether segment DE (indices 3,4) crosses triangle ABC (indices 0,1,2),
/// every point carried in one common positive multiple of the `Int`
/// lattice: the side D stands on when it does, `std::nullopt` when it does
/// not. Five SoS orientation verdicts decide it, each scale-invariant
/// (@ref tf::exact::orient3d_sos_scaled), so the answer is the unscaled
/// segment's and no volume is evaluated.
template <typename Int, typename Index, typename Coord>
auto triangle_segment_intersect_side_scaled_sos(
    const std::array<vertex<Index, Coord>, 5> &vs) -> std::optional<bool> {
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

  return abcd;
}

/// The crossing @ref tf::exact::triangle_segment_intersect_side_scaled_sos
/// confirms, with its point. The barycentric weights are a ratio of two
/// volumes that carry the same cube of the scale, so the position along DE
/// is the unscaled segment's. The point comes back in the scaled space it
/// was asked in, which leaves the order of several crossings of one
/// segment — the only thing a caller compares distances for — the unscaled
/// order.
template <typename Int, typename Index, typename Coord>
auto triangle_segment_intersect_point_scaled_sos(
    const std::array<vertex<Index, Coord>, 5> &vs)
    -> std::optional<triangle_segment_crossing<Coord>> {
  using T1 = typename meta<Int>::T1;
  using T3 = typename meta<Int>::T3;

  const auto side = triangle_segment_intersect_side_scaled_sos<Int>(vs);
  if (!side)
    return std::nullopt;

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
      point[i] = static_cast<Coord>(div_round(
          T3(abs_e) * T3(vs[3].pt[i]) + T3(abs_d) * T3(vs[4].pt[i]), T3(sum)));
  } else {
    for (int i = 0; i < 3; ++i)
      point[i] = static_cast<Coord>((T1(vs[3].pt[i]) + T1(vs[4].pt[i])) / 2);
  }

  return triangle_segment_crossing<Coord>{point, *side};
}

/// @ref tf::exact::triangle_segment_intersect_side_scaled_sos on the
/// lattice itself.
template <typename Index, typename Int>
auto triangle_segment_intersect_side_sos(
    const std::array<vertex<Index, Int>, 5> &vs) -> std::optional<bool> {
  return triangle_segment_intersect_side_scaled_sos<Int>(vs);
}

/// @ref tf::exact::triangle_segment_intersect_point_scaled_sos on the
/// lattice itself.
template <typename Index, typename Int>
auto triangle_segment_intersect_point_sos(
    const std::array<vertex<Index, Int>, 5> &vs)
    -> std::optional<triangle_segment_crossing<Int>> {
  return triangle_segment_intersect_point_scaled_sos<Int>(vs);
}

} // namespace tf::exact
