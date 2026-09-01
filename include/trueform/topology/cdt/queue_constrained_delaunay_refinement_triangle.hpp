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

namespace tf::topology::cdt {

template <typename Owner>
auto queue_constrained_delaunay_refinement_triangle(
    Owner &owner, typename Owner::index_type face) -> void {
  using Index = typename Owner::index_type;

  if (owner._t[face].seg[0] == Owner::none &&
      owner._t[face].seg[1] == Owner::none &&
      owner._t[face].seg[2] == Owner::none &&
      owner.quality(face) >= owner._min_quality)
    return;
  owner._queue.push_back({face, Index(owner._t[face].stamp)});
}

} // namespace tf::topology::cdt
