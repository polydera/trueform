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

#include "../core/angle.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/points.hpp"
#include "../topology/half_edges.hpp"
#include "./flip/admits_min_angle_flip.hpp"

namespace tf::remesh {

/// @ingroup remesh
/// @brief Delaunay-style edge flips that raise the minimum angle.
///
/// Flips a (non-feature, flippable) edge when doing so raises the smaller of the
/// two triangles' minimum angles -- the criterion that removes slivers/caps that
/// a valence flip leaves behind. Every flip is fold-guarded (it must not invert
/// a triangle), so it is safe on uneven or near-flat regions where is_flip_ok's
/// topology-only check is not enough. A flip whose quad touches a feature/crease
/// vertex is only taken when the result clears `angle_floor`, so it never trades
/// one sliver for a marginally-less-bad one against the locked feature net.
///
/// @param mask  feature_mask (skip feature edges/floor near feature vertices) or
///              tf::none (no features).
/// @param max_deviation  When > 0, reject flips that displace the surface by
///              more than this: the displacement of a flip is the skew
///              distance between the old and new diagonal.
template <typename Index, typename PointsPolicy, typename MaskOrNone>
auto flip_min_angle(tf::half_edges<Index> &he,
                    const tf::points<PointsPolicy> &points,
                    const MaskOrNone &mask, int iterations,
                    tf::rad<tf::coordinate_type<PointsPolicy>> angle_floor,
                    tf::coordinate_type<PointsPolicy> max_deviation = 0)
    -> Index {
  Index n_flipped = 0;
  for (int it = 0; it < iterations; ++it) {
    Index flipped = 0;
    for (auto eh : he.edge_handles()) {
      if (!tf::remesh::admits_min_angle_flip(he, points, mask, eh, angle_floor,
                                             max_deviation))
        continue;
      he.flip(eh);
      ++flipped;
    }
    n_flipped += flipped;
    if (flipped == 0)
      break;
  }
  return n_flipped;
}

} // namespace tf::remesh
