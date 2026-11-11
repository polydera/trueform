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
#include "../../core/buffer.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/vertex.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// PER SIDE, ONE DIMENSION: the rows of `edge_points` on one side, at their
/// parameters `t` (@ref state_tet_side_parameters), ordered along it.
/// Adjacent rows closer than `quantum` along the side are one coincidence,
/// and a row within one lattice unit of an end is that end
/// (`end_identity(slot)`): welded through `identities` — a created point
/// into an original, the input being the authority on its corners, and two
/// created points by the election; two originals stand apart. Transitive
/// by the order, never chaining in space. `slot_point(slot)` reads an end;
/// `order` is scratch. O(rows log rows).
template <typename Int, typename Index, typename SlotPoint,
          typename EndIdentity, typename Identities>
auto weld_tet_side_points(
    const tf::buffer<std::array<Index, 3>> &edge_points,
    const tf::buffer<typename tf::exact::meta<Int>::param_type> &t,
    const SlotPoint &slot_point, const EndIdentity &end_identity, Int quantum,
    Identities &identities, tf::buffer<std::size_t> &order) -> void {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T3 = typename tf::exact::meta<Int>::T3;
  constexpr int bits = tf::exact::meta<Int>::param_bits;
  const auto one = [&identities](Index x, Index y) {
    x = identities.standing(x);
    y = identities.standing(y);
    if (x == y)
      return;
    const bool ox = identities.is_original(x), oy = identities.is_original(y);
    if (ox && oy)
      return;
    if (ox)
      identities.retire(y, x);
    else if (oy)
      identities.retire(x, y);
    else
      identities.weld(x, y);
  };
  const std::size_t n = edge_points.size();
  for (std::size_t i = 0; i < n;) {
    std::size_t j = i + 1;
    while (j < n && edge_points[j][0] == edge_points[i][0] &&
           edge_points[j][1] == edge_points[i][1])
      ++j;
    const tf::exact::pt3<Int> a = slot_point(edge_points[i][0]);
    const tf::exact::pt3<Int> b = slot_point(edge_points[i][1]);
    T3 length2(0);
    for (std::size_t d = 0; d < 3; ++d) {
      const T1 side = T1(b[d]) - T1(a[d]);
      length2 = length2 + T3(side) * T3(side);
    }
    const T3 full = T3(1) << bits;
    // a gap along the side, in parameter units, within `limit` lattice units
    const auto within = [&](T3 gap, T3 limit) {
      return gap * gap * length2 <= limit * limit * full * full;
    };
    order.clear();
    for (std::size_t r = i; r < j; ++r)
      order.push_back(r);
    std::sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) {
      return t[x] < t[y] || (!(t[y] < t[x]) && x < y);
    });
    for (std::size_t k = 0; k < order.size(); ++k) {
      const std::size_t r = order[k];
      const Index x = edge_points[r][2];
      if (within(T3(t[r]), T3(1)))
        one(end_identity(edge_points[r][0]), x);
      if (within(full - T3(t[r]), T3(1)))
        one(end_identity(edge_points[r][1]), x);
      if (k && within(T3(t[r]) - T3(t[order[k - 1]]), T3(quantum)))
        one(edge_points[order[k - 1]][2], x);
    }
    i = j;
  }
}

} // namespace tf::tetmesh
