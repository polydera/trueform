/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (www.trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
#pragma once

#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/basis.hpp"
#include "../core/buffer.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/linalg/least_squares.hpp"
#include "../core/points.hpp"
#include "../core/sqrt.hpp"
#include "../core/unit_vector_like.hpp"
#include "../core/views/sequence_range.hpp"
#include "../topology/face_membership.hpp"
#include "../topology/make_k_ring.hpp"
#include "../topology/policy/vertex_link.hpp"
#include "../topology/vertex_link.hpp"
#include "./compute_point_normals.hpp"

namespace tf::geometry {

/// @ingroup geometry
/// @brief Workspace state for principal curvature computation.
///
/// Holds reusable buffers for least squares fitting. Use as thread-local
/// state with parallel_for_each for efficient inline curvature computation.
///
/// @tparam T The scalar type (float, double).
/// @tparam Index The vertex index type.
template <typename T, typename Index> struct curvature_work_state {
  tf::buffer<T> A;
  tf::buffer<T> b_vec;
  tf::buffer<T> work;
  tf::buffer<Index> neighbor_ids;
};

/// @ingroup geometry
/// @brief Compute principal curvatures at a vertex from its neighborhood.
///
/// Fits a quadric z = ax² + bxy + cy² to the neighborhood projected onto
/// the tangent plane, then computes shape operator eigenvalues.
///
/// @tparam PointsPolicy The points policy type.
/// @tparam NormalPolicy The unit vector policy type.
/// @tparam NeighborRange Range of neighbor vertex indices.
/// @param state Reusable workspace buffers.
/// @param points Mesh vertex positions.
/// @param vid The vertex index to compute curvatures for.
/// @param normal The unit normal at vid.
/// @param neighbors Range of neighbor vertex indices.
/// @return Array of {k1, k2} principal curvatures, or {0, 0} if < 5 neighbors.
template <typename PointsPolicy, typename Index, typename NormalPolicy,
          typename NeighborRange>
auto compute_principal_curvatures(
    curvature_work_state<tf::coordinate_type<PointsPolicy>, Index> &state,
    const tf::points<PointsPolicy> &points, std::size_t vid,
    const tf::unit_vector_like<3, NormalPolicy> &normal,
    NeighborRange &&neighbors) {
  using T = tf::coordinate_type<PointsPolicy>;

  const std::size_t n = neighbors.size();

  // Need at least 5 neighbors for robust 3-parameter fit
  if (n < 5)
    return std::array<T, 2>{T(0), T(0)};

  // Build local coordinate frame
  auto [t0, t1] = tf::make_basis_from_normal(normal);
  auto origin = points[vid];

  constexpr std::size_t cols = 3;

  // Resize workspace
  state.A.allocate(n * cols);
  state.b_vec.allocate(n);
  const auto work_size = tf::linalg::least_squares_workspace_size<T>(n, cols);
  state.work.allocate(work_size);

  // Build system: project neighbors to local coords
  std::size_t i = 0;
  for (auto neighbor_id : neighbors) {
    auto p = points[neighbor_id];
    auto diff = p - origin;

    T x = tf::dot(diff, t0);
    T y = tf::dot(diff, t1);
    T z = tf::dot(diff, normal);

    // Column-major: A[i + j*n]
    state.A[i + 0 * n] = x * x;
    state.A[i + 1 * n] = x * y;
    state.A[i + 2 * n] = y * y;
    // Negate z: shape operator S = -Hessian for z = f(x,y)
    state.b_vec[i] = -z;
    ++i;
  }

  // Solve least squares
  std::array<T, cols> coeffs;
  tf::linalg::solve_least_squares(state.A.data(), state.b_vec.data(),
                                  coeffs.data(), n, cols, state.work.data());

  T a = coeffs[0];
  T b_coef = coeffs[1];
  T c = coeffs[2];

  // Shape operator eigenvalues (principal curvatures)
  T trace = T(2) * (a + c);
  T det = T(4) * a * c - b_coef * b_coef;
  T disc = trace * trace - T(4) * det;
  if (disc < T(0))
    disc = T(0);
  T sqrt_disc = tf::sqrt(disc);

  return std::array<T, 2>{(trace + sqrt_disc) / T(2),
                          (trace - sqrt_disc) / T(2)};
}
} // namespace tf::geometry
namespace tf {

/// @ingroup geometry
/// @brief Compute principal curvatures for all vertices.
///
/// Uses inline k-ring traversal with parallel_for_each for efficiency.
/// Neighborhoods are computed on-the-fly using k-ring BFS.
///
/// @tparam PolygonsPolicy The polygons policy type.
/// @tparam OutputRange Output range for curvature pairs.
/// @param polygons The input polygons.
/// @param output Output range of std::array<T, 2> for {k1, k2} per vertex.
/// @param k Number of rings for neighborhood (default 2).
template <typename PolygonsPolicy, typename OutputRange>
void compute_principal_curvatures(const tf::polygons<PolygonsPolicy> &polygons,
                                  OutputRange &&output, std::size_t k = 2) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  if constexpr (!tf::has_vertex_link_policy<PolygonsPolicy>) {
    if constexpr (!tf::has_face_membership_policy<PolygonsPolicy>) {
      tf::face_membership<Index> fm;
      fm.build(polygons);
      tf::vertex_link<Index> vlink;
      vlink.build(polygons.faces(), fm);
      return compute_principal_curvatures(
          polygons | tf::tag(fm) | tf::tag(vlink), output, k);
    } else {
      tf::vertex_link<Index> vlink;
      vlink.build(polygons.faces(), polygons.face_membership());
      return compute_principal_curvatures(polygons | tf::tag(vlink), output, k);
    }
  } else {
    const auto &points = polygons.points();

    auto compute = [&output, &polygons, &points, k](const auto &normals) {
      using T = tf::coordinate_type<PolygonsPolicy>;
      const auto n_vertices = points.size();
      const auto &vlink = polygons.vertex_link();

      // Thread-local state
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

            output[vid] = geometry::compute_principal_curvatures(
                state.curvature, points, vid, normals[vid],
                state.curvature.neighbor_ids);
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

} // namespace tf
