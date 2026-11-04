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
#include "../../../core/config/verify.hpp"
#include "../../../core/coordinate_type.hpp"
#include "../../../core/point_like.hpp"
#include "../../../exact/vertex.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// Where a site or a point to be minted stands: its floor, the lattice point
/// of the frame at or below it on every axis, and below the frame its
/// `depth` and the `residual` there, the position being
/// `floor + residual / 2^depth`. A position is canonical: at depth zero its
/// residual is zero, below it some axis of the residual is odd, so two
/// positions are equal exactly when their parts are. A read that takes the
/// floor for the position, exact at depth zero, is called shallow.
template <typename Coord> struct tet_position {
  tf::exact::pt3<Coord> floor;
  int depth = 0;
  std::array<std::int64_t, 3> residual{};
};

template <typename Coord>
auto operator==(const tet_position<Coord> &a, const tet_position<Coord> &b)
    -> bool {
  return a.depth == b.depth && a.floor == b.floor && a.residual == b.residual;
}

/// The order positions are sorted and welded in: floor, depth, residual.
template <typename Coord>
auto operator<(const tet_position<Coord> &a, const tet_position<Coord> &b)
    -> bool {
  if (!(a.floor == b.floor))
    return a.floor < b.floor;
  if (a.depth != b.depth)
    return a.depth < b.depth;
  return a.residual < b.residual;
}

/// The position `p` read shallow.
template <typename Coord>
auto tet_shallow_position(const tet_position<Coord> &p)
    -> tf::exact::pt3<Coord> {
  if constexpr (tf::core::verification)
    tf::core::verify(!p.depth, "a shallow read names a position on the frame");
  return p.floor;
}

/// `x`, standing `frame_bits` above the frame and read `depth` below it,
/// minted once at `x 2^(frame_bits - depth)`: a tier's point by the tier
/// before it walks, lands or appends it, a caller's by the build or append
/// that takes it, which stores a handed position as given. Its depth is
/// decided here and nowhere else, the least that names it.
template <typename Coord, typename Policy>
auto tet_minted_position(const tf::point_like<3, Policy> &x, int frame_bits = 0,
                         int depth = 0) -> tet_position<Coord> {
  using Wide = tf::coordinate_type<Policy>;
  tet_position<Coord> p;
  if (depth <= frame_bits) {
    const Coord frame = Coord(1) << (frame_bits - depth);
    for (std::size_t k = 0; k < 3; ++k)
      p.floor[k] = Coord(x[k]) * frame;
    return p;
  }
  depth -= frame_bits;
  const Wide unit = Wide(1) << depth;
  std::array<Wide, 3> rest{};
  for (std::size_t k = 0; k < 3; ++k) {
    Wide below = x[k] / unit;
    if (x[k] - below * unit < Wide(0))
      below = below - Wide(1);
    p.floor[k] = Coord(below);
    rest[k] = x[k] - below * unit;
  }
  while (depth && ((rest[0] | rest[1] | rest[2]) & Wide(1)) == Wide(0)) {
    for (auto &r : rest)
      r = r / Wide(2);
    --depth;
  }
  if constexpr (tf::core::verification)
    tf::core::verify(depth < 63, "a residual stands in 64 bits");
  p.depth = depth;
  for (std::size_t k = 0; k < 3; ++k)
    p.residual[k] = std::int64_t(rest[k]);
  return p;
}

} // namespace tf::topology::cdt::dt3
