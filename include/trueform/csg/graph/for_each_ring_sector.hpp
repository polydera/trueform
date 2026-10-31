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
#include "../../core/algorithm/circular_increment.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "./make_plane_radial_fans.hpp"
#include <cstddef>

namespace tf::csg::graph {

/// The sectors of ring `fan`: for every two cyclic-adjacent pages, every
/// occurrence of the one meets every occurrence of the other, and `f`
/// receives `(node_a, node_b, frag_a, frag_b)` — the two component-side
/// nodes the sector joins and their components. Sides come from the
/// occurrence's own traversal direction. An occurrence the exposure did not
/// carry meets nothing.
template <typename Index, typename Labels, typename ExposedOfRow, typename F>
auto for_each_ring_sector(const Labels &labels,
                          const tf::csg::graph::plane_radial_fans<Index> &fans,
                          const ExposedOfRow &exposed_of_row, Index fan,
                          const F &f) -> void {
  auto dirs_view = tf::make_offset_block_range(fans.rows.offsets_buffer(),
                                               tf::make_range(fans.dirs));
  const auto fragment = [&](Index row, Index &frag) {
    if (row < Index(0)) {
      frag = Index(-row - 1);
      return true;
    }
    const Index exposed = exposed_of_row[std::size_t(row / Index(3))];
    if (exposed == Index(-1))
      return false;
    frag = labels[exposed];
    return true;
  };
  const Index first = fans.page_offsets[std::size_t(fan)];
  const Index K = fans.page_offsets[std::size_t(fan) + 1] - first;
  for (Index r = 0; r < K; ++r) {
    const Index rn = tf::circular_increment(r, K);
    auto page = fans.rows[std::size_t(first + r)];
    auto page_dirs = dirs_view[std::size_t(first + r)];
    auto next = fans.rows[std::size_t(first + rn)];
    auto next_dirs = dirs_view[std::size_t(first + rn)];
    for (std::size_t a = 0; a < page.size(); ++a) {
      Index frag_a;
      if (!fragment(page[a], frag_a))
        continue;
      const Index sa = page_dirs[a] ? Index(0) : Index(1);
      for (std::size_t b = 0; b < next.size(); ++b) {
        Index frag_b;
        if (!fragment(next[b], frag_b))
          continue;
        const Index sb = (next_dirs[b] ? Index(0) : Index(1)) ^ Index(1);
        f(Index(2 * frag_a + sa), Index(2 * frag_b + sb), frag_a, frag_b);
      }
    }
  }
}

} // namespace tf::csg::graph
