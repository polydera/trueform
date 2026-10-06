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
#include "../../core/range.hpp"
#include "../hole_split.hpp"
#include "./hole_metric.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::fill {

/// The scratch one carrying face's plan is built on.
template <typename Index> struct hole_plan_scratch {
  tf::buffer<std::array<Index, 3>> triangles;
  tf::buffer<std::array<Index, 2>> chain;
};

/// The canonical sequential subdivision of a carrying triangle.
///
/// Its splits arrive in the one order that defines the result: original
/// edges by ascending canonical key, and each edge's parameters ascending in
/// that key's own direction. Each split is inserted into the boundary
/// subedge that contains it, replacing that subedge's unique incident
/// triangle with the two the insertion makes, so the block covers the face,
/// keeps its winding, preserves every boundary subedge and states no flat
/// triangle. `s` splits give `s + 1` triangles.
///
/// The operation consumes resolved point tickets and mints nothing. The
/// finished block is normalized — every triangle rotated onto its smallest
/// corner, the block then sorted — so two carriers of one face state it
/// identically.
template <typename Index, typename Iterator>
auto subdivide_carrier_triangle(
    const std::array<Index, 3> &corners,
    const tf::range<Iterator, tf::dynamic_size> &splits,
    tf::fill::hole_plan_scratch<Index> &scratch,
    tf::buffer<std::array<Index, 3>> &out) -> void {
  scratch.triangles.clear();
  scratch.triangles.push_back(corners);
  Index low = Index(-1), high = Index(-1);

  for (const auto &split : splits) {
    if (split.v0 != low || split.v1 != high) {
      low = split.v0;
      high = split.v1;
      scratch.chain.clear();
      scratch.chain.push_back({low, Index(0)});
      scratch.chain.push_back({high, Index(tf::hole_split_scale)});
    }
    const Index parameter = Index(split.parameter);
    Index before = Index(-1), after = Index(-1);
    Index before_at = Index(-1), after_at = Index(tf::hole_split_scale + 1);
    for (auto entry : scratch.chain) {
      if (entry[1] < parameter && entry[1] > before_at) {
        before = entry[0];
        before_at = entry[1];
      }
      if (entry[1] > parameter && entry[1] < after_at) {
        after = entry[0];
        after_at = entry[1];
      }
    }
    scratch.chain.push_back({split.point, parameter});

    for (std::size_t i = 0; i < scratch.triangles.size(); ++i) {
      const auto triangle = scratch.triangles[i];
      int rotation = -1;
      for (int r = 0; r < 3; ++r) {
        const Index a = triangle[std::size_t(r)];
        const Index b = triangle[std::size_t((r + 1) % 3)];
        if ((a == before && b == after) || (a == after && b == before))
          rotation = r;
      }
      if (rotation < 0)
        continue;
      const Index a = triangle[std::size_t(rotation)];
      const Index b = triangle[std::size_t((rotation + 1) % 3)];
      const Index z = triangle[std::size_t((rotation + 2) % 3)];
      scratch.triangles[i] = {a, split.point, z};
      scratch.triangles.push_back({split.point, b, z});
      break;
    }
  }

  const auto base = out.size();
  for (auto triangle : scratch.triangles) {
    const int first = tf::fill::hole_canonical_rotation(
        triangle[0], triangle[1], triangle[2]);
    out.push_back({triangle[std::size_t(first)],
                   triangle[std::size_t((first + 1) % 3)],
                   triangle[std::size_t((first + 2) % 3)]});
  }
  std::sort(out.begin() + std::ptrdiff_t(base), out.end());
}

} // namespace tf::fill
