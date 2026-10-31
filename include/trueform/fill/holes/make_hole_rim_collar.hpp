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
#include "./hole_collar.hpp"
#include <cstddef>

namespace tf::fill {

/// The collar of a rim read alone: the rim in its own order and no ring,
/// which is what a chart of the rim's own plane stands on.
template <typename RimRange, typename Index>
auto make_hole_rim_collar(const RimRange &rim,
                          tf::fill::hole_collar<Index> &collar) -> void {
  const Index n = Index(rim.size());
  collar.vertices.clear();
  for (Index k = 0; k < n; ++k)
    collar.vertices.push_back(Index(rim[std::size_t(k)]));
  collar.vertex_offsets = {Index(0), n, n, n};
  collar.faces.clear();
  collar.face_offsets = {Index(0), Index(0), Index(0)};
  collar.ring_triangles.clear();
}

} // namespace tf::fill
