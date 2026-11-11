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
#include "../../exact/dyadic_blend.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/vertex.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>

namespace tf::tetmesh {

/// The scratch and product of @ref state_tet_lane_occurrences: per lane
/// and identity `{lane, identity, occurrence}` sorted, `-1` where the lane
/// charts the identity at its own position, and the occurrences' positions.
template <typename Index, typename Int> struct tet_lane_occurrences {
  tf::buffer<std::array<Index, 6>> entries;
  tf::buffer<std::array<Index, 2>> corners;
  tf::buffer<std::array<Index, 3>> key;
  tf::buffer<tf::exact::pt3<Int>> position;

  /// Where lane `lane` charts identity `x`, whose one position is `own`.
  auto chart(Index lane, Index x, const tf::exact::pt3<Int> &own) const
      -> tf::exact::pt3<Int> {
    const auto at = std::lower_bound(key.begin(), key.end(),
                                     std::array<Index, 3>{lane, x, Index(-1)});
    return at != key.end() && (*at)[0] == lane && (*at)[1] == x &&
                   (*at)[2] >= Index(0)
               ? position[std::size_t((*at)[2])]
               : own;
  }
};

/// WITHIN ONE COPLANAR LANE A POINT IS ONE POINT OF ONE PLANE: an identity
/// a lane's sides state has one chart position there, and every side of
/// the lane that states it reads it there. AN IDENTITY'S OCCURRENCE IN A
/// LANE IS ITS OWN POSITION WHERE THE LANE STATES IT ON TWO NON-COLLINEAR
/// SIDES — the crossing of those lines, which its stated position rounds —
/// ELSE ITS PARAMETRIC POSITION ON ITS FIRST SIDE in canonical order (sides
/// by slots, then the parameter `t`, @ref state_tet_side_parameters). A
/// corner of one of the lane's members is at its own position. The identity
/// stays one flat point everywhere, so a side two lanes share is one set of
/// definition rows; only a lane's chart reads it at its occurrence
/// (@ref tf::arrangement::states_chart_point). `lanes` offsets the members
/// `members` (their corners) of `triangle`, `rows_on_edge(t, k, visit)`
/// visits the rows of `side_rows` on a member's side and `slot_point(slot)`
/// reads a side's end. O(rows log rows).
template <typename Int, typename Index, typename RowsOnEdge, typename SlotPoint>
auto state_tet_lane_occurrences(
    const tf::buffer<Index> &lanes,
    const tf::buffer<std::array<Index, 3>> &members,
    const tf::buffer<Index> &triangle, const RowsOnEdge &rows_on_edge,
    const tf::buffer<std::array<Index, 3>> &side_rows,
    const tf::buffer<typename tf::exact::meta<Int>::param_type> &t,
    const SlotPoint &slot_point, tet_lane_occurrences<Index, Int> &out)
    -> void {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  auto &entries = out.entries;
  auto &corners = out.corners;
  entries.clear();
  corners.clear();
  const std::size_t n_lanes = lanes.size() - 1;
  for (std::size_t c = 0; c < n_lanes; ++c)
    for (Index f = lanes[c]; f < lanes[c + 1]; ++f) {
      for (auto v : members[std::size_t(f)])
        corners.push_back({Index(c), v});
      for (std::size_t k = 0; k < 3; ++k)
        rows_on_edge(triangle[std::size_t(f)], k, [&](std::size_t r) {
          const auto &row = side_rows[r];
          entries.push_back(
              {Index(c), row[2], row[0], row[1], Index(r), Index(0)});
        });
    }
  std::sort(corners.begin(), corners.end());
  corners.erase_till_end(std::unique(corners.begin(), corners.end()));
  std::sort(entries.begin(), entries.end(),
            [&t](const std::array<Index, 6> &x, const std::array<Index, 6> &y) {
              return std::make_tuple(x[0], x[1], x[2], x[3],
                                     t[std::size_t(x[4])], x[4]) <
                     std::make_tuple(y[0], y[1], y[2], y[3],
                                     t[std::size_t(y[4])], y[4]);
            });
  // two sides are one line when both ends of the second stand on the first
  const auto on_line = [](const tf::exact::pt3<Int> &a,
                          const tf::exact::pt3<Int> &b,
                          const tf::exact::pt3<Int> &p) {
    T1 u[3], w[3];
    for (std::size_t d = 0; d < 3; ++d) {
      u[d] = T1(b[d]) - T1(a[d]);
      w[d] = T1(p[d]) - T1(a[d]);
    }
    return T2(u[1]) * w[2] - T2(u[2]) * w[1] == T2(0) &&
           T2(u[2]) * w[0] - T2(u[0]) * w[2] == T2(0) &&
           T2(u[0]) * w[1] - T2(u[1]) * w[0] == T2(0);
  };
  out.key.clear();
  out.position.clear();
  for (std::size_t i = 0; i < entries.size();) {
    std::size_t j = i + 1;
    while (j < entries.size() && entries[j][0] == entries[i][0] &&
           entries[j][1] == entries[i][1])
      ++j;
    const Index lane = entries[i][0], x = entries[i][1];
    const tf::exact::pt3<Int> a = slot_point(entries[i][2]),
                              b = slot_point(entries[i][3]);
    bool crossing = false;
    for (std::size_t k = i + 1; k < j && !crossing; ++k)
      crossing = !on_line(a, b, slot_point(entries[k][2])) ||
                 !on_line(a, b, slot_point(entries[k][3]));
    Index occurrence = Index(-1);
    if (!crossing && !std::binary_search(corners.begin(), corners.end(),
                                         std::array<Index, 2>{lane, x})) {
      const auto tr = t[std::size_t(entries[i][4])];
      tf::exact::pt3<Int> p;
      for (std::size_t d = 0; d < 3; ++d)
        p[d] = tf::exact::dyadic_blend(a[d], b[d], tr);
      occurrence = Index(out.position.size());
      out.position.push_back(p);
    }
    out.key.push_back({lane, x, occurrence});
    i = j;
  }
}

} // namespace tf::tetmesh
