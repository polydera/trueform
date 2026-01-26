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
#include "../core/algorithm/parallel_transform.hpp"
#include "../core/normal.hpp"
#include "../core/polygons.hpp"
#include "../core/unit_vectors_buffer.hpp"

namespace tf {

/// @ingroup geometry_normals
/// @brief Compute cell normals for polygons.
/// @tparam Policy The policy type of the polygons.
/// @param polygons The input polygons (must be 3D).
/// @return A unit_vectors_buffer containing one normal per polygon.
template <typename Policy>
auto compute_normals(const tf::polygons<Policy> &polygons)
    -> tf::unit_vectors_buffer<tf::coordinate_type<Policy>, 3> {
  static_assert(tf::coordinate_dims_v<Policy> == 3,
                "compute_normals requires 3D polygons");

  using T = tf::coordinate_type<Policy>;

  tf::unit_vectors_buffer<T, 3> normals;
  normals.allocate(polygons.size());

  tf::parallel_transform(polygons, normals.unit_vectors(),
                         [](const auto &poly) {
                           return tf::make_normal(poly[0], poly[1], poly[2]);
                         });

  return normals;
}

} // namespace tf
