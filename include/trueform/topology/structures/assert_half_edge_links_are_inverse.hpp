/*
 * Copyright (c) 2025 XLAB
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

#include "../../core/algorithm/parallel_contains.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/config/verify.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../half_edge.hpp"

namespace tf::topology {

/// Every live half-edge keeps the loop-link law @ref tf::half_edge states.
template <typename Index>
auto assert_half_edge_links_are_inverse(
    const tf::buffer<tf::half_edge<Index>> &half_edges) -> void {
  const Index n = Index(half_edges.size());
  bool broken = tf::parallel_contains(
      tf::make_sequence_range(n),
      [&half_edges, n](Index id) {
        const auto &h = half_edges[id];
        if (h.is_removed())
          return false;
        if (h.has_next() && (h.next >= n || half_edges[h.next].is_removed() ||
                             half_edges[h.next].prev != id))
          return true;
        return h.prev >= 0 && (h.prev >= n || half_edges[h.prev].is_removed() ||
                               half_edges[h.prev].next != id);
      },
      tf::checked);
  tf::core::verify(!broken,
                   "a half-edge's loop links name it back in both directions");
}

} // namespace tf::topology
