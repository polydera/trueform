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
#include "./compact_plane_weld_triangles.hpp"
#include "./plane_arrangement_store.hpp"
#include "./rewrite_plane_flat_triangles.hpp"

#include <array>

namespace tf::arrangement {

/// THE SPAN WELD: a weld is a substitution over the published store — every
/// retained corner speaks the closed table `merges` (@ref
/// rewrite_plane_flat_triangles), and a triangle the weld collapsed leaves
/// with every table it carries (@ref compact_plane_weld_triangles).
template <typename Index>
auto weld_plane_triangles(const tf::buffer<std::array<Index, 3>> &merges,
                          Index n_flat, plane_arrangement_store<Index> &store)
    -> void {
  rewrite_plane_flat_triangles(store.triangles, merges, n_flat);
  if (merges.size() == 0)
    return;
  compact_plane_weld_triangles(
      store.triangles, store.slot_parents, store.corner_subs, store.coplanar_of,
      store.coplanar, store.stacked, store.triangle_cells, store.plane_offsets,
      store.face_range);
}

} // namespace tf::arrangement
