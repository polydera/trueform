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
#include "../core/policy/normals.hpp"
#include "../core/polygons.hpp"
#include "../core/unit_vector.hpp"
#include "../core/unit_vectors_buffer.hpp"
#include "../core/views/zip.hpp"
#include "../topology/face_membership.hpp"
#include "../topology/policy/face_membership.hpp"
#include "./compute_normals.hpp"

namespace tf {

/// @ingroup geometry_normals
/// @brief Compute point normals for polygons by averaging adjacent face
/// normals.
/// @tparam Policy The policy type of the polygons.
/// @param polygons The input polygons (must be 3D). If face_membership or
/// normals are not tagged, they will be computed automatically.
/// @return A unit_vectors_buffer containing one normal per point.
template <typename Policy>
auto compute_point_normals(const tf::polygons<Policy> &polygons)
    -> tf::unit_vectors_buffer<tf::coordinate_type<Policy>, 3> {
  static_assert(tf::coordinate_dims_v<Policy> == 3,
                "compute_point_normals requires 3D polygons");

  if constexpr (!tf::has_normals_policy<Policy>) {
    auto poly_normals = tf::compute_normals(polygons);
    return compute_point_normals(
        polygons | tf::tag_normals(poly_normals.unit_vectors()));
  } else if constexpr (!tf::has_face_membership_policy<Policy>) {
    using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
    tf::face_membership<Index> fm;
    fm.build(polygons);
    return compute_point_normals(polygons | tf::tag(fm));
  } else {
    using T = tf::coordinate_type<Policy>;

    tf::unit_vectors_buffer<T, 3> normals;
    normals.allocate(polygons.points().size());

    const auto &fm = polygons.face_membership();
    const auto &poly_normals = polygons.normals();

    tf::parallel_for_each(
        tf::zip(normals.unit_vectors(), fm), [&poly_normals](auto pair) {
          auto &&[normal, poly_ids] = pair;
          tf::vector<T, 3> sum{T(0), T(0), T(0)};
          for (auto n : tf::make_indirect_range(poly_ids, poly_normals)) {
            sum += n;
          }
          normal = tf::make_unit_vector(sum);
        });

    return normals;
  }
}

} // namespace tf
