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
#include "./algorithm/reduce.hpp"
#include "./cross.hpp"
#include "./dot.hpp"
#include "./frame_of.hpp"
#include "./polygons.hpp"
#include "./transformed.hpp"

namespace tf {

/// @ingroup core_queries
/// @brief Compute signed volume of a closed 3D polygon mesh.
///
/// Uses the divergence theorem to compute the volume enclosed by a
/// closed mesh. Positive volume indicates outward-facing normals
/// (right-hand rule), negative indicates inward-facing normals.
///
/// @tparam Policy The polygon mesh policy type.
/// @param polygons The closed polygon mesh.
/// @return The signed volume (positive for outward normals).
template <typename Policy>
auto signed_volume(const tf::polygons<Policy> &polygons) {
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;
  static_assert(Dims == 3, "signed_volume requires 3D polygons");
  using T = tf::coordinate_type<Policy>;

  auto frame = tf::frame_of(polygons);

  // Use first vertex as reference for numerical stability.
  // Volume is translation-invariant for closed meshes, so
  // V = (1/6) * sum(dot(p0-r, cross(p1-r, p2-r))) for any r.
  auto ref = tf::transformed(polygons.points()[0], frame);

  auto polygon_volumes =
      tf::make_mapped_range(polygons, [&frame, &ref](const auto &poly) {
        T sum = 0;
        auto v0 = tf::transformed(poly[0], frame) - ref;
        auto size = poly.size();
        for (decltype(size) i = 1; i + 1 < size; ++i) {
          auto v1 = tf::transformed(poly[i], frame) - ref;
          auto v2 = tf::transformed(poly[i + 1], frame) - ref;
          sum += tf::dot(v0, tf::cross(v1, v2));
        }
        return sum;
      });

  T result = tf::reduce(polygon_volumes, std::plus<>{}, T{0}, tf::checked);
  return result / T{6};
}
} // namespace tf
