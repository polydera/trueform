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
#include "../../../../core/buffer.hpp"
#include "../../../../core/config/verify.hpp"
#include "../../../winding/flood_triangle_winding.hpp"
#include "./tet_face_cover_scratch.hpp"
#include "./tet_orient2d_sign.hpp"
#include "./tet_projected_point.hpp"
#include "./tet_rim_piece.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// State into `layers`, aligned with the cover's facets, how many of a
/// face's own triangles lie over each facet a spread from its `rim` read
/// (@ref spread_tet_face_cover): the winding of the rim around it
/// (@ref tf::topology::flood_triangle_winding). A rim piece is a wall run as
/// often each way as the face's triangles holding it on either side do, so
/// entering a facet across it steps by the triangles on the facet's side less
/// those on the other; a piece no facet of the cover stands across faces the
/// outside, which winds nothing. Every site is read in the decision's
/// `sites` on the projection `axes`.
template <typename Sites, typename Index>
auto state_tet_cover_layers(const Sites &sites,
                            const std::pair<int, int> &axes,
                            const tf::buffer<tet_rim_piece<Index>> &rim,
                            tf::buffer<Index> &layers,
                            tet_face_cover_scratch<Index> &scratch) -> void {
  using Int = typename Sites::int_type;
  const auto &due = scratch.due;
  auto &adjacent = scratch.adjacent;
  const std::size_t n = due.size();
  layers.allocate(n);
  const auto flat = [&sites, &axes](Index s) {
    return tet_projected_point(sites[std::size_t(s)].pt, axes);
  };
  const auto side_ends = [&due](std::size_t t, std::size_t e) {
    const auto &f = due[t];
    const std::array<Index, 3> corners{f.u, f.v, f.x};
    return std::array<Index, 3>{corners[e], corners[(e + 1) % 3],
                                corners[(e + 2) % 3]};
  };
  // per side of the cover the step entering its facet across it; a rim side
  // is matched by its edge to the one facet across it, or to the outside
  auto &steps = scratch.steps;
  steps.allocate_and_initialize(n * 3, static_cast<signed char>(0));
  auto &rim_sides = scratch.rim_sides;
  rim_sides.clear();
  for (std::size_t t = 0; t < n; ++t)
    for (std::size_t e = 0; e < 3; ++e)
      if (adjacent[t][e] == Index(-2)) {
        const auto ends = side_ends(t, e);
        const Index lo = std::min(ends[0], ends[1]);
        const Index hi = std::max(ends[0], ends[1]);
        const auto run = std::equal_range(
            rim.begin(), rim.end(), tet_rim_piece<Index>{lo, hi, 0});
        int plus = 0, minus = 0;
        for (auto at = run.first; at != run.second; ++at)
          (at->side > 0 ? plus : minus) += 1;
        const int side = tet_orient2d_sign<Int>(flat(lo), flat(hi),
                                                flat(ends[2]));
        steps[3 * t + e] =
            static_cast<signed char>(side > 0 ? plus - minus : minus - plus);
        rim_sides.push_back({lo, hi, Index(t), Index(e)});
      }
  std::sort(rim_sides.begin(), rim_sides.end());
  for (std::size_t i = 0; i < rim_sides.size();) {
    std::size_t end = i + 1;
    while (end < rim_sides.size() && rim_sides[end][0] == rim_sides[i][0] &&
           rim_sides[end][1] == rim_sides[i][1])
      ++end;
    if constexpr (tf::core::verification)
      tf::core::verify(end - i <= 2,
                       "a piece holds one facet of a plane on either side");
    const auto &a = rim_sides[i];
    if (end - i == 2) {
      const auto &b = rim_sides[i + 1];
      adjacent[std::size_t(a[2])][std::size_t(a[3])] = b[2];
      adjacent[std::size_t(b[2])][std::size_t(b[3])] = a[2];
    } else
      adjacent[std::size_t(a[2])][std::size_t(a[3])] = Index(-1);
    i = end;
  }
  const auto neighbor = [&adjacent](Index t, int e) {
    return adjacent[std::size_t(t)][std::size_t(e)];
  };
  const auto entering = [&steps](Index t, int e) {
    return int(steps[3 * std::size_t(t) + std::size_t(e)]);
  };
  tf::topology::flood_triangle_winding(n, neighbor, entering, scratch.stack,
                                       scratch.winding);
  for (std::size_t t = 0; t < n; ++t) {
    layers[t] = Index(scratch.winding[t]);
    if constexpr (tf::core::verification) {
      tf::core::verify(layers[t] > Index(0),
                       "every facet of a cover lies under the face");
      for (int e = 0; e < 3; ++e) {
        const Index q = neighbor(Index(t), e);
        const int beyond = q == Index(-1) ? 0 : int(scratch.winding[q]);
        tf::core::verify(int(scratch.winding[t]) ==
                             beyond + entering(Index(t), e),
                         "every piece moves the layers by the triangles it "
                         "bounds");
      }
    }
  }
}

} // namespace tf::topology::cdt::dt3
