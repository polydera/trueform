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
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/angle.hpp"
#include "../core/buffer.hpp"
#include "../core/checked.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/polygons.hpp"
#include "../core/views/enumerate.hpp"
#include "./quality/face_quality_of.hpp"

namespace tf {

/// @ingroup geometry
/// @brief One flat buffer per measure, each indexed by face.
///
/// The corner angles and the aspect ratio — the longest side over the
/// shortest, infinite where a side has no length — are stated for a face of
/// any arity. An angle is unsigned and lies in `[0, pi]`, so a reflex corner
/// of a non-convex face reads its explement. `quality` is the triangle
/// measure, `1` for equilateral and approaching `0` for a sliver; a face that
/// is not a triangle has none and reads `-1`.
template <typename T> struct face_quality {
  tf::buffer<T> quality;
  tf::buffer<tf::rad<T>> min_angle;
  tf::buffer<tf::rad<T>> max_angle;
  tf::buffer<T> aspect_ratio;
};

/// @ingroup geometry
/// @brief Measure every face of a mesh.
/// @tparam Policy The policy type of the polygons.
/// @param polygons The input polygons (must be 3D).
/// @return A @ref tf::face_quality holding one value per face in each buffer.
template <typename Policy>
auto compute_face_quality(const tf::polygons<Policy> &polygons)
    -> tf::face_quality<tf::coordinate_type<Policy>> {
  static_assert(tf::coordinate_dims_v<Policy> == 3,
                "compute_face_quality requires 3D polygons");

  tf::face_quality<tf::coordinate_type<Policy>> values;
  values.quality.allocate(polygons.size());
  values.min_angle.allocate(polygons.size());
  values.max_angle.allocate(polygons.size());
  values.aspect_ratio.allocate(polygons.size());

  tf::parallel_for_each(
      tf::enumerate(polygons),
      [&values](const auto &pair) {
        const auto &[face_id, face] = pair;
        const auto measured = tf::geometry::face_quality_of(face);
        values.quality[face_id] = measured.quality;
        values.min_angle[face_id] = measured.min_angle;
        values.max_angle[face_id] = measured.max_angle;
        values.aspect_ratio[face_id] = measured.aspect_ratio;
      },
      tf::checked);

  return values;
}

} // namespace tf
