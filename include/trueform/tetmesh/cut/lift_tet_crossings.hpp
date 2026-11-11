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
#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>

namespace tf::tetmesh {

/// One triangle's box in a sweep, and the side of the sweep it stands on.
template <typename Index, typename Int> struct tet_sweep_box {
  std::array<std::array<Int, 3>, 2> box;
  Index id;
  Index side;
};

/// The scratch @ref lift_tet_crossings walks on.
template <typename Index, typename Int> struct tet_lift_workspace {
  tf::buffer<std::array<Index, 4>> sides;
  tf::buffer<Index> side_a;
  tf::buffer<Index> side_b;
  tf::buffer<tet_sweep_box<Index, Int>> sweep;
  tf::buffer<Index> open_a;
  tf::buffer<Index> open_b;
};

/// The triangle pairs of the crossings `named` into `seeds`: a side is a
/// segment's live triangles on both its ends (`holders_of`) or a facet's
/// live triangles (`facet_triangles`), each pair of sides is lifted once,
/// and its pairs are those whose boxes (`box_of`) meet, by one sweep along
/// the first axis. O(named log named + Σ over distinct side pairs of
/// (a + b) log (a + b) + the pairs met).
template <typename Index, typename Int, typename Named, typename HoldersOf,
          typename FacetTriangles, typename BoxOf>
auto lift_tet_crossings(const Named &named, const HoldersOf &holders_of,
                        const FacetTriangles &facet_triangles,
                        const BoxOf &box_of,
                        tf::buffer<std::array<Index, 2>> &seeds,
                        tet_lift_workspace<Index, Int> &work) -> void {
  seeds.clear();
  // a side is `{kind, id}`: 0 a segment, 1 a facet
  auto &sides = work.sides;
  sides.clear();
  for (const auto &pair : named) {
    const std::array<Index, 2> a =
        pair.segment >= 0 ? std::array<Index, 2>{Index(0), pair.segment}
                          : std::array<Index, 2>{Index(1), pair.face};
    if (pair.crosses.segment >= 0)
      sides.push_back({a[0], a[1], Index(0), pair.crosses.segment});
    if (pair.crosses.face >= 0)
      sides.push_back({a[0], a[1], Index(1), pair.crosses.face});
  }
  std::sort(sides.begin(), sides.end());
  sides.erase_till_end(std::unique(sides.begin(), sides.end()));
  const auto triangles_of = [&](Index kind, Index id, tf::buffer<Index> &out) {
    if (kind == 0)
      holders_of(id, out);
    else
      facet_triangles(id, out);
  };
  const auto meet = [](const auto &a, const auto &b) {
    for (std::size_t d = 0; d < 3; ++d)
      if (a[1][d] < b[0][d] || b[1][d] < a[0][d])
        return false;
    return true;
  };
  auto &sweep = work.sweep;
  auto &open_a = work.open_a;
  auto &open_b = work.open_b;
  for (const auto &side : sides) {
    triangles_of(side[0], side[1], work.side_a);
    triangles_of(side[2], side[3], work.side_b);
    sweep.clear();
    for (auto x : work.side_a)
      sweep.push_back({box_of(x), x, Index(0)});
    for (auto y : work.side_b)
      sweep.push_back({box_of(y), y, Index(1)});
    std::sort(sweep.begin(), sweep.end(), [](const auto &u, const auto &v) {
      return std::tie(u.box[0][0], u.side, u.id) <
             std::tie(v.box[0][0], v.side, v.id);
    });
    open_a.clear();
    open_b.clear();
    for (std::size_t i = 0; i < sweep.size(); ++i) {
      const auto &e = sweep[i];
      auto &other = e.side ? open_a : open_b;
      std::size_t w = 0;
      for (std::size_t k = 0; k < other.size(); ++k) {
        const auto &o = sweep[std::size_t(other[k])];
        if (o.box[1][0] < e.box[0][0])
          continue;
        other[w++] = other[k];
        if (e.id != o.id && meet(e.box, o.box))
          seeds.push_back(e.side ? std::array<Index, 2>{o.id, e.id}
                                 : std::array<Index, 2>{e.id, o.id});
      }
      other.erase_till_end(other.begin() + std::ptrdiff_t(w));
      (e.side ? open_b : open_a).push_back(Index(i));
    }
  }
}

} // namespace tf::tetmesh
