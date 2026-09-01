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
#include <array>
#include <cstddef>

namespace tf::arrangement {

/// Per region of a pooled plane, the member that speaks for it: the
/// minimal-tag member covering it, first in members order on ties. Every
/// other covering member emits a coplanar duplicate of that member's
/// triangles; a region no member covers elects nobody and is not emitted.
///
/// A region's own cover names its candidates, ascending, so the election reads
/// what exists and nothing else.
template <typename Index, typename Graph, typename Members>
auto elect_plane_region_survivors(
    const Graph &g, const Members &members,
    const tf::buffer<std::array<Index, 2>> &cover_range,
    const tf::buffer<Index> &cover, Index n_regions,
    tf::buffer<Index> &survivors) -> void {
  survivors.allocate(std::size_t(n_regions));
  for (Index r = 0; r < n_regions; ++r) {
    const auto block = cover_range[std::size_t(r)];
    auto elected = Index(-1);
    auto elected_tag = Index(0);
    for (auto at = block[0]; at != block[1]; ++at) {
      const auto mi = cover[std::size_t(at)];
      const auto tag =
          Index(g.descriptors()[std::size_t(members[std::size_t(mi)])].tag);
      if (elected == Index(-1) || tag < elected_tag) {
        elected = mi;
        elected_tag = tag;
      }
    }
    survivors[std::size_t(r)] = elected;
  }
}

} // namespace tf::arrangement
