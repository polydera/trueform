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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/unsafe.hpp"
#include "../../topology/half_edges.hpp"

namespace tf::remesh {

/// @ingroup remesh
/// @brief How many rings past the partition separators the sequential cleanup
/// may collapse.
inline constexpr int cleanup_rings = 3;

/// @ingroup remesh
/// @brief Grow a parallel collapse's separator marks into the region its
/// sequential cleanup works.
///
/// Takes @p frozen as the separators the partitioning left (0 free, 1 marked)
/// and leaves every vertex within @ref cleanup_rings rings of one carrying the
/// ring that reached it, the rest still 0. A ring reads only the stamps
/// earlier rings wrote, so the region is the partition's alone.
template <typename Index>
auto grow_cleanup_region(const tf::half_edges<Index> &he,
                         tf::buffer<char> &frozen) -> void {
  static_assert(cleanup_rings + 1 <= 127, "the ring stamp must fit a char");
  for (int ring = 0; ring < cleanup_rings; ++ring) {
    const char reached = char(ring + 1);
    tf::parallel_for_each(he.edge_handles(), [&, reached](const auto &eh) {
      if (!eh.is_valid())
        return;
      auto h0 = he.half_edge_handle(tf::unsafe, eh, false);
      auto v0 = he.start_vertex_handle(tf::unsafe, h0).id();
      auto v1 = he.end_vertex_handle(tf::unsafe, h0).id();
      char s0 = frozen[v0];
      char s1 = frozen[v1];
      if ((s0 == 0) == (s1 == 0))
        return;
      if ((s0 != 0 ? s0 : s1) > reached)
        return;
      frozen[s0 != 0 ? v1 : v0] = char(reached + 1);
    });
  }
}

} // namespace tf::remesh
