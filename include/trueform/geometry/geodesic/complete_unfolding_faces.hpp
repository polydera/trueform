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
#include "../../core/algorithm/parallel_transform.hpp"
#include "../../core/checked.hpp"
#include "../../core/polygons.hpp"
#include "./unfolding_faces.hpp"

namespace tf::geometry {

/// Form every image of `unfolding` in one parallel pass, so fronts that run
/// at once only read it. An image a pop already formed is formed again to the
/// same bits, which keeps the pass one transform over the triangles.
template <typename Policy>
auto complete_unfolding_faces(const tf::polygons<Policy> &polygons,
                              tf::geometry::unfolding_faces &unfolding)
    -> void {
  tf::parallel_transform(
      polygons, unfolding.images,
      [](const auto &polygon) {
        return tf::geometry::make_unfolding_face(polygon);
      },
      tf::checked);
  tf::parallel_fill(unfolding.formed, char(1));
}

} // namespace tf::geometry
