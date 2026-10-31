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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
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
  tf::parallel_fill(rejected, Index(-1));

  const auto &vertex_offsets = rims.vertices.offsets_buffer();
  tf::buffer<Index> claim_offsets;
  claim_offsets.allocate(rims.size() + 1);
  claim_offsets[0] = 0;
  for (std::size_t rim = 0; rim < rims.size(); ++rim) {
    const Index n = Index(vertex_offsets[rim + 1] - vertex_offsets[rim]);
    const Index edges = rims.closed[rim] ? n : std::max(Index(0), Index(n - 1));
    claim_offsets[rim + 1] = Index(claim_offsets[rim] + edges);
  }

  tf::buffer<std::array<Index, 4>> claims;
  claims.allocate(std::size_t(claim_offsets[rims.size()]));
  const auto &corners = rims.vertices.data_buffer();
  tf::parallel_for_each(
      tf::make_sequence_range(corners.size()),
      [&](std::size_t at) {
        const std::size_t rim = std::size_t(
            std::upper_bound(vertex_offsets.begin() + 1, vertex_offsets.end(),
                             Index(at)) -
            (vertex_offsets.begin() + 1));
        const Index first = vertex_offsets[rim];
        const Index n = Index(vertex_offsets[rim + 1] - first);
        const Index k = Index(Index(at) - first);
        if (k >= Index(claim_offsets[rim + 1] - claim_offsets[rim]))
          return;
        const Index a = corners[at];
        const Index b = corners[std::size_t(first + (k + 1 == n ? 0 : k + 1))];
        claims[std::size_t(claim_offsets[rim] + k)] = {
            std::min(a, b), std::max(a, b), Index(rim), k};
      },
      tf::checked);
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
