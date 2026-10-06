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
#include "../../topology/boundary_rims.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::fill {

/// Name, per rim, the lowest edge it shares with another rim, or `-1` where
/// it shares none.
///
/// A rim edge belongs to one rim, so an edge two rims both claim makes the
/// supply itself ambiguous; every rim involved refuses, and the ambiguity
/// stays the caller's to resolve rather than being settled by an election
/// between them.
template <typename Index>
auto reject_overlapping_hole_rims(const tf::boundary_rims<Index> &rims,
                                  tf::buffer<Index> &rejected) -> void {
  rejected.allocate(rims.size());
  for (std::size_t rim = 0; rim < rims.size(); ++rim)
    rejected[rim] = Index(-1);

  tf::buffer<std::array<Index, 4>> claims;
  for (std::size_t rim = 0; rim < rims.size(); ++rim) {
    const auto vertices = rims.vertices[rim];
    const Index n = Index(vertices.size());
    const Index edges = Index(rims.closed[rim] ? n : n - 1);
    for (Index k = 0; k < edges; ++k) {
      const Index a = vertices[std::size_t(k)];
      const Index b = vertices[std::size_t(k + 1 == n ? 0 : k + 1)];
      claims.push_back(
          {std::min(a, b), std::max(a, b), Index(rim), k});
    }
  }
  tbb::parallel_sort(claims.begin(), claims.end());

  for (std::size_t k = 0; k + 1 < claims.size();) {
    std::size_t run = k + 1;
    while (run < claims.size() && claims[run][0] == claims[k][0] &&
           claims[run][1] == claims[k][1])
      ++run;
    if (run - k > 1)
      for (std::size_t claim = k; claim < run; ++claim) {
        auto &ticket = rejected[std::size_t(claims[claim][2])];
        if (ticket == Index(-1) || claims[claim][3] < ticket)
          ticket = claims[claim][3];
      }
    k = run;
  }
}

} // namespace tf::fill
