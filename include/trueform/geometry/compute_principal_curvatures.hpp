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
#include "../core/basis.hpp"
#include "../core/buffer.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/epsilon.hpp"
#include "../core/linalg/least_squares.hpp"
#include "../core/none.hpp"
#include "../core/points.hpp"
#include "../core/sqrt.hpp"
#include "../core/unit_vector.hpp"
#include "../core/unit_vector_like.hpp"
#include "../core/views/sequence_range.hpp"
#include "../topology/face_membership.hpp"
#include "../topology/make_k_rings.hpp"
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
/// @brief Result of principal curvature computation (values only).
template <typename T> struct curvature_values {
  T k0;
  T k1;
};

/// @ingroup geometry
/// @brief Result of principal curvature computation (values + directions).
template <typename T> struct curvature_full {
  T k0;
  T k1;
  tf::unit_vector<T, 3> d0;
  tf::unit_vector<T, 3> d1;
};

/// @ingroup geometry
/// @brief Compute principal curvatures at a vertex from its neighborhood.
///
/// Fits a quadric z = ax² + bxy + cy² to the neighborhood projected onto
/// the tangent plane, then computes shape operator eigenvalues.
/// When WithDirections is true, also computes eigenvectors (principal
/// directions).
///
/// @tparam WithDirections If true, compute principal directions as well.
/// @tparam PointsPolicy The points policy type.
/// @tparam NormalPolicy The unit vector policy type.
/// @tparam NeighborRange Range of neighbor vertex indices.
/// @param state Reusable workspace buffers.
/// @param points Mesh vertex positions.
/// @param vid The vertex index to compute curvatures for.
/// @param normal The unit normal at vid.
/// @param neighbors Range of neighbor vertex indices.
/// @return curvature_values{k0, k1} or curvature_full{k0, k1, d0, d1} depending
/// on WithDirections.
template <bool WithDirections, typename PointsPolicy, typename Index,
          typename NormalPolicy, typename NeighborRange>
auto compute_principal_curvatures(
    curvature_work_state<tf::coordinate_type<PointsPolicy>, Index> &state,
    const tf::points<PointsPolicy> &points, std::size_t vid,
    const tf::unit_vector_like<3, NormalPolicy> &normal,
    NeighborRange &&neighbors) {
  using T = tf::coordinate_type<PointsPolicy>;

  const std::size_t n = neighbors.size();

  // Need at least 5 neighbors for robust 3-parameter fit
  if (n < 5) {
    if constexpr (WithDirections) {
      return curvature_full<T>{T(0), T(0),
                               tf::make_unit_vector(tf::unsafe, 1, 0, 0),
                               tf::make_unit_vector(tf::unsafe, 0, 1, 0)};
    } else {
      return curvature_values<T>{T(0), T(0)};
    }
  }

  // Build local coordinate frame
  auto [t0, t1] = tf::make_basis_from_normal(normal);
  auto origin = points[vid];

  constexpr std::size_t cols = 5;

  // Resize workspace
  state.A.allocate(n * cols);
  state.b_vec.allocate(n);
  const auto work_size = tf::linalg::least_squares_workspace_size<T>(n, cols);
  state.work.allocate(work_size);

  // Build system: project neighbors to local coords
  // Fit z = ax² + bxy + cy² + dx + ey (linear terms improve robustness)
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
    state.A[i + 3 * n] = x;
    state.A[i + 4 * n] = y;
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
  // Shape operator S = [[2a, b], [b, 2c]]
  T trace = T(2) * (a + c);
  T det = T(4) * a * c - b_coef * b_coef;
  T disc = trace * trace - T(4) * det;
  if (disc < T(0))
    disc = T(0);
  T sqrt_disc = tf::sqrt(disc);

  T k0 = (trace + sqrt_disc) / T(2);
  T k1 = (trace - sqrt_disc) / T(2);

  if constexpr (WithDirections) {
    // Compute eigenvectors of shape operator S = [[2a, b], [b, 2c]]
    // For eigenvalue k, solve (S - kI)v = 0
    // Row 1: (2a - k)v0 + b*v1 = 0  =>  v = (b, k - 2a) or (-b, 2a - k)
    // We use v = (b, k - 2a) and normalize

    tf::vector<T, 3> d0_world, d1_world;

    T denom0 = b_coef * b_coef + (k0 - T(2) * a) * (k0 - T(2) * a);
    if (denom0 > tf::epsilon2<T>) {
      T inv_len = T(1) / tf::sqrt(denom0);
      T local_d0_x = b_coef * inv_len;
      T local_d0_y = (k0 - T(2) * a) * inv_len;
      // Transform to world space: d0 = local_d0_x * t0 + local_d0_y * t1
      d0_world = tf::make_vector(local_d0_x * t0[0] + local_d0_y * t1[0],
                                 local_d0_x * t0[1] + local_d0_y * t1[1],
                                 local_d0_x * t0[2] + local_d0_y * t1[2]);
    } else {
      // Degenerate case (isotropic curvature) - use t0 as d0
      d0_world = tf::make_vector(t0[0], t0[1], t0[2]);
    }

    // d1 is perpendicular to d0 in tangent plane: d1 = normal × d0
    d1_world =
        tf::make_vector(normal[1] * d0_world[2] - normal[2] * d0_world[1],
                        normal[2] * d0_world[0] - normal[0] * d0_world[2],
                        normal[0] * d0_world[1] - normal[1] * d0_world[0]);

    return curvature_full<T>{k0, k1, d0_world, d1_world};
  } else {
    return curvature_values<T>{k0, k1};
  }
}

template <typename PolygonsPolicy, typename Range0, typename Range1,
          typename Range2, typename Range3>
void compute_principal_curvatures(const tf::polygons<PolygonsPolicy> &polygons,
                                  Range0 &&ks0, Range1 &&ks1, Range2 &&dirs0,
                                  Range3 &&dirs1, std::size_t k) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  if constexpr (!tf::has_vertex_link_policy<PolygonsPolicy>) {
    if constexpr (!tf::has_face_membership_policy<PolygonsPolicy>) {
      tf::face_membership<Index> fm;
      fm.build(polygons);
      tf::vertex_link<Index> vlink;
      vlink.build(polygons.faces(), fm);
      return geometry::compute_principal_curvatures(
          polygons | tf::tag(fm) | tf::tag(vlink), ks0, ks1, dirs0, dirs1, k);
    } else {
      tf::vertex_link<Index> vlink;
      vlink.build(polygons.faces(), polygons.face_membership());
      return geometry::compute_principal_curvatures(polygons | tf::tag(vlink),
                                                    ks0, ks1, dirs0, dirs1, k);
    }
  } else {
    const auto &points = polygons.points();

    auto compute = [&, k](const auto &normals) {
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

            constexpr bool with_directions =
                !std::is_same_v<std::decay_t<Range2>, tf::none_t>;

            auto res = geometry::compute_principal_curvatures<with_directions>(
                state.curvature, points, vid, normals[vid],
                state.curvature.neighbor_ids);
            ks0[vid] = res.k0;
            ks1[vid] = res.k1;
            if constexpr (with_directions) {
              dirs0[vid] = res.d0;
              dirs1[vid] = res.d1;
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
template <typename PolygonsPolicy, typename Range0, typename Range1>
auto compute_principal_curvatures(const tf::polygons<PolygonsPolicy> &polygons,
                                  Range0 &&ks0, Range1 &&ks1,
                                  std::size_t k = 2) {
  static_assert(tf::coordinate_dims_v<PolygonsPolicy> == 3,
                "3D dimensionality required");
  return geometry::compute_principal_curvatures(polygons, ks0, ks1, tf::none,
                                                tf::none, k);
}

template <typename PolygonsPolicy, typename Range0, typename Range1,
          typename Range2, typename Range3>
auto compute_principal_curvatures(const tf::polygons<PolygonsPolicy> &polygons,
                                  Range0 &&ks0, Range1 &&ks1, Range2 &&dirs0,
                                  Range3 &&dirs1, std::size_t k = 2) {
  static_assert(tf::coordinate_dims_v<PolygonsPolicy> == 3,
                "3D dimensionality required");
  return geometry::compute_principal_curvatures(polygons, ks0, ks1, dirs0,
                                                dirs1, k);
}

/// @ingroup geometry
/// @brief Compute principal curvatures for all vertices, returning result
/// buffers.
///
/// Convenience wrapper that allocates output buffers internally.
/// Principal curvatures (k0, k1) characterize surface curvature at each vertex,
/// where k0 is the maximum curvature and k1 is the minimum curvature.
///
/// @tparam PolygonsPolicy The polygons policy type.
/// @param polygons The input polygons.
/// @param k Number of rings for neighborhood (default 2).
/// @return Pair of buffers {k0, k1} containing principal curvatures per vertex.
template <typename PolygonsPolicy>
auto make_principal_curvatures(const tf::polygons<PolygonsPolicy> &polygons,
                               std::size_t k = 2) {
  static_assert(tf::coordinate_dims_v<PolygonsPolicy> == 3,
                "3D dimensionality required");
  using T = tf::coordinate_type<PolygonsPolicy>;
  tf::buffer<T> ks0;
  ks0.allocate(polygons.points().size());
  tf::buffer<T> ks1;
  ks1.allocate(polygons.points().size());
  compute_principal_curvatures(polygons, ks0, ks1, k);
  return std::make_pair(std::move(ks0), std::move(ks1));
}

/// @ingroup geometry
/// @brief Compute principal curvatures and directions for all vertices.
///
/// Convenience wrapper that allocates output buffers internally.
/// Returns both principal curvatures (k0, k1) and their corresponding
/// directions (d0, d1) as unit vectors in the tangent plane.
///
/// @tparam PolygonsPolicy The polygons policy type.
/// @param polygons The input polygons.
/// @param k Number of rings for neighborhood (default 2).
/// @return Tuple of {k0, k1, d0, d1} where k0/k1 are curvature buffers and
///         d0/d1 are unit_vectors_buffers containing principal directions.
template <typename PolygonsPolicy>
auto make_principal_directions(const tf::polygons<PolygonsPolicy> &polygons,
                               std::size_t k = 2) {
  static_assert(tf::coordinate_dims_v<PolygonsPolicy> == 3,
                "3D dimensionality required");
  using T = tf::coordinate_type<PolygonsPolicy>;
  tf::buffer<T> ks0;
  ks0.allocate(polygons.points().size());
  tf::buffer<T> ks1;
  ks1.allocate(polygons.points().size());
  tf::unit_vectors_buffer<T, tf::coordinate_dims_v<PolygonsPolicy>> dirs0;
  dirs0.allocate(polygons.points().size());
  tf::unit_vectors_buffer<T, tf::coordinate_dims_v<PolygonsPolicy>> dirs1;
  dirs1.allocate(polygons.points().size());
  compute_principal_curvatures(polygons, ks0, ks1, dirs0, dirs1, k);
  return std::make_tuple(std::move(ks0), std::move(ks1), std::move(dirs0),
                         std::move(dirs1));
}

} // namespace tf
