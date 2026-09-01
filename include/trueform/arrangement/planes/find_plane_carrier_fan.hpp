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

#include "../../exact/meta.hpp"
#include "../../exact/orient2d.hpp"
#include "../../exact/vertex.hpp"

#include <array>
#include <cstddef>

namespace tf::arrangement {

/// One carrier's answer without a triangulation: the boundary ring in loop
/// order, the corner to fan it from, and whether that ring turns against the
/// carrier's own winding. `size == 0` is the whole statement that this carrier
/// is not one of the family.
template <typename Index> struct plane_carrier_fan {
  std::array<Index, 4> ring{};
  std::size_t size = 0;
  std::size_t apex = 0;
  bool reversed = false;
};

/// CORE. THE CONVEX FAMILY'S PREDICATE, stated once, on the PREPARED
/// constraint set — which is the one place both world states have already
/// resolved their carrier to the same thing.
///
/// A carrier belongs to the family when its constraint set is ONE SIMPLE
/// CLOSED RING over its own point table and nothing else: three or four
/// points, as many constraints, every one a boundary statement, every point of
/// degree two, and the walk closing on its start having visited each once.
/// Then the ring IS the carrier's boundary, and a fan from the right corner is
/// the whole triangulation of it — @ref
/// tf::topology::for_each_convex_chain_fan_triangle.
///
/// THE GUARD IS THE TURN. Every corner's exact turn must be non-zero: that is
/// simultaneously the proof that the carrier bounds area, that no two of its
/// identities stand at one projected point (so no weld is possible), and that
/// no emitted triangle is degenerate. A zero turn declines, and the
/// triangulation answers for that carrier on the tier's ordinary terms.
///
/// The apex follows from the turns. All of one sign is convex and either
/// diagonal answers; exactly one of the other sign is the reflex corner, and
/// the fan from it is the only one that stays inside; two and two is a
/// crossing loop, which is nobody's fan.
///
/// THIS PREDICATE IS TRUE OF SOME BOOLEAN-PATH CARRIERS TOO — a cut face whose
/// every statement coincides with its own boundary has exactly such a block —
/// and firing there is correct: the answer is the same triangulation over the
/// same identities, and a later statement that changes the block puts the
/// carrier back in the wave's frontier, where this predicate is asked again of
/// the new constraint set and declines.
template <typename Index, typename Int, typename Local>
auto find_plane_carrier_fan(const Local &local) -> plane_carrier_fan<Index> {
  using T2 = typename tf::exact::meta<Int>::T2;
  plane_carrier_fan<Index> fan;
  const auto n = local.ends.size();
  if (n < 3 || n > fan.ring.size() || local.cons.size() != 2 * n)
    return fan;
  for (const auto boundary : local.bnd)
    if (boundary == char(0))
      return fan;

  std::array<std::array<Index, 2>, 4> peers{};
  std::array<int, 4> degree{};
  for (std::size_t at = 0; at < n; ++at) {
    const auto a = local.cons[2 * at];
    const auto b = local.cons[2 * at + 1];
    if (a < Index(0) || b < Index(0) || std::size_t(a) >= n ||
        std::size_t(b) >= n || a == b)
      return fan;
    if (degree[std::size_t(a)] == 2 || degree[std::size_t(b)] == 2)
      return fan;
    peers[std::size_t(a)][std::size_t(degree[std::size_t(a)]++)] = b;
    peers[std::size_t(b)][std::size_t(degree[std::size_t(b)]++)] = a;
  }

  std::array<char, 4> seen{};
  auto previous = Index(-1);
  auto current = Index(0);
  seen[0] = char(1);
  for (std::size_t at = 1; at < n; ++at) {
    const auto &pair = peers[std::size_t(current)];
    const auto next = pair[0] == previous ? pair[1] : pair[0];
    if (next == previous || seen[std::size_t(next)] != char(0))
      return fan;
    fan.ring[at] = next;
    seen[std::size_t(next)] = char(1);
    previous = current;
    current = next;
  }
  const auto &closing = peers[std::size_t(current)];
  if (closing[0] != Index(0) && closing[1] != Index(0))
    return fan;

  const auto projected = [&local](Index id) -> tf::exact::pt2<Int> {
    const auto point = local.pts2[std::size_t(id)];
    return {point[0], point[1]};
  };
  std::array<int, 4> turn{};
  int positive = 0;
  for (std::size_t at = 0; at < n; ++at) {
    const auto value = tf::exact::orient2d<Int>(
        projected(fan.ring[(at + n - 1) % n]), projected(fan.ring[at]),
        projected(fan.ring[(at + 1) % n]));
    if (value == T2(0))
      return fan;
    turn[at] = value > T2(0) ? 1 : -1;
    positive += turn[at] > 0 ? 1 : 0;
  }
  const auto negative = int(n) - positive;
  if (positive != int(n) && negative != int(n)) {
    if (positive != 1 && negative != 1)
      return fan;
    for (std::size_t at = 0; at < n; ++at)
      if ((turn[at] > 0) == (positive == 1))
        fan.apex = at;
  }
  fan.reversed = (positive > negative ? 1 : -1) != local.face_orientation;
  fan.size = n;
  return fan;
}

} // namespace tf::arrangement
