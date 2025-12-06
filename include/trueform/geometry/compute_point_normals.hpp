/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/parallel_apply.hpp"
#include "../core/policy/normals.hpp"
#include "../core/polygons.hpp"
#include "../core/unit_vector.hpp"
#include "../core/unit_vectors_buffer.hpp"
#include "../core/views/zip.hpp"
#include "../topology/policy/face_membership.hpp"

namespace tf {

/// @brief Compute point normals for polygons by averaging adjacent face
/// normals.
/// @tparam Policy The policy type of the polygons (must have face_membership
/// and normals).
/// @param polygons The input polygons (must be 3D, must have face_membership
/// and normals tagged).
/// @return A unit_vectors_buffer containing one normal per point.
template <typename Policy>
auto compute_point_normals(const tf::polygons<Policy> &polygons)
    -> tf::unit_vectors_buffer<tf::coordinate_type<Policy>, 3> {
  static_assert(tf::coordinate_dims_v<Policy> == 3,
                "compute_point_normals requires 3D polygons");
  static_assert(tf::has_face_membership_policy<Policy>,
                "compute_point_normals requires face_membership to be tagged");
  static_assert(tf::has_normals_policy<Policy>,
                "compute_point_normals requires normals to be tagged");

  using T = tf::coordinate_type<Policy>;

  tf::unit_vectors_buffer<T, 3> normals;
  normals.allocate(polygons.points().size());

  const auto &fm = polygons.face_membership();
  const auto &poly_normals = polygons.normals();

  tf::parallel_apply(
      tf::zip(normals.unit_vectors(), fm), [&poly_normals](auto pair) {
        auto &&[normal, poly_ids] = pair;
        tf::vector<T, 3> sum{T(0), T(0), T(0)};
        for (auto normal : tf::make_indirect_range(poly_ids, poly_normals)) {
          sum += normal;
        }

        normal = tf::make_unit_vector(sum);
      });

  return normals;
}

} // namespace tf
