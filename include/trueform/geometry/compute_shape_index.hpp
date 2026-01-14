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

#include "./compute_principal_curvatures.hpp"
#include <cmath>

namespace tf {

/// @ingroup geometry
/// @brief Compute shape index for all vertices.
///
/// Shape index S = (2/π) * arctan((k1 + k2) / (k1 - k2)) maps principal
/// curvatures to [-1, 1]:
///   - S = -1: spherical cup (concave)
///   - S = -0.5: cylindrical cup
///   - S = 0: saddle
///   - S = 0.5: cylindrical cap
///   - S = 1: spherical cap (convex)
///
/// @tparam PolygonsPolicy The polygons policy type.
/// @tparam OutputRange Output range for shape index values.
/// @param polygons The input polygons.
/// @param output Output range of T for shape index per vertex.
/// @param k Number of rings for neighborhood (default 2).
template <typename PolygonsPolicy, typename OutputRange>
void compute_shape_index(const tf::polygons<PolygonsPolicy> &polygons,
                         OutputRange &&output, std::size_t k = 2) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  if constexpr (!tf::has_vertex_link_policy<PolygonsPolicy>) {
    if constexpr (!tf::has_face_membership_policy<PolygonsPolicy>) {
      tf::face_membership<Index> fm;
      fm.build(polygons);
      tf::vertex_link<Index> vlink;
      vlink.build(polygons.faces(), fm);
      return compute_shape_index(polygons | tf::tag(fm) | tf::tag(vlink),
                                 output, k);
    } else {
      tf::vertex_link<Index> vlink;
      vlink.build(polygons.faces(), polygons.face_membership());
      return compute_shape_index(polygons | tf::tag(vlink), output, k);
    }
  } else {
    using T = tf::coordinate_type<PolygonsPolicy>;
    constexpr T two_over_pi = T(2) / T(3.14159265358979323846);

    const auto &points = polygons.points();

    auto compute = [&output, &polygons, &points, k,
                    two_over_pi](const auto &normals) {
      const auto n_vertices = points.size();
      const auto &vlink = polygons.vertex_link();

      struct State {
        topology::k_ring_applier<Index> applier;
        geometry::curvature_work_state<T, Index> curvature;
      };

      tf::parallel_for_each(
          tf::make_sequence_range(static_cast<Index>(n_vertices)),
          [&](Index vid, State &state) {
            state.curvature.neighbor_ids.clear();

            state.applier(vlink, vid, k, false, [&](Index n) {
              state.curvature.neighbor_ids.push_back(n);
            });

            auto [k1, k2] = geometry::compute_principal_curvatures<false>(
                state.curvature, points, vid, normals[vid],
                state.curvature.neighbor_ids);

            // Shape index: S = (2/π) * arctan((k1 + k2) / (k1 - k2))
            T diff = k1 - k2;
            if (diff == T(0)) {
              // Umbilical point: k1 == k2
              // S = 1 if k1 > 0, S = -1 if k1 < 0, S = 0 if flat
              output[vid] = k1 > T(0) ? T(1) : (k1 < T(0) ? T(-1) : T(0));
            } else {
              output[vid] = two_over_pi * std::atan((k1 + k2) / diff);
            }
          },
          State{});
    };

    if constexpr (!tf::has_normals_policy<std::decay_t<decltype(points)>>) {
      compute(tf::compute_point_normals(polygons));
    } else {
      compute(polygons.points().normals());
    }
  }
}

/// @ingroup geometry
/// @brief Compute shape index for all vertices, returning the result buffer.
///
/// Convenience wrapper that allocates the output buffer internally.
///
/// @tparam PolygonsPolicy The polygons policy type.
/// @param polygons The input polygons.
/// @param k Number of rings for neighborhood (default 2).
/// @return Buffer of shape index values, one per vertex.
template <typename PolygonsPolicy>
auto make_shape_index(const tf::polygons<PolygonsPolicy> &polygons,
                      std::size_t k = 2) {
  using T = tf::coordinate_type<PolygonsPolicy>;
  tf::buffer<T> output;
  output.allocate(polygons.points().size());
  compute_shape_index(polygons, output, k);
  return output;
}

} // namespace tf
