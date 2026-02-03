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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/coordinate_type.hpp"
#include "../../core/cross.hpp"
#include "../../core/dot.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/linalg/least_squares_parallel.hpp"
#include "../../core/make_rotation.hpp"
#include "../../core/points.hpp"
#include "../../core/policy/unwrap.hpp"
#include "../../core/transformation.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/sequence_range.hpp"

namespace tf::geometry {

/// @brief Workspace state for point-to-plane alignment.
///
/// Holds reusable buffers for the least squares system. Use to avoid
/// repeated allocations when calling fit_rigid_alignment in a loop
/// (e.g., ICP iterations).
///
/// @tparam T The scalar type (float, double).
template <typename T> struct plane_alignment_state {
  tf::buffer<T> A;     ///< N×6 matrix (column-major)
  tf::buffer<T> b_vec; ///< N×1 vector
  tf::linalg::parallel_least_squares_state<T> solver_state;
};

/// @brief Point-to-plane rigid alignment using linearized rotation.
///
/// Computes a rigid transformation T such that T(X) ≈ Y by minimizing
/// the point-to-plane distance: sum_i ((T(x_i) - y_i) · n_i)².
///
/// The rotation is linearized as R ≈ I + [r]× where r = (rx, ry, rz) is
/// the Rodrigues vector. This gives a 6-unknown linear system solved via
/// parallel TSQR least squares.
///
/// Point-to-plane ICP typically converges ~3x faster than point-to-point
/// on smooth surfaces.
///
/// @note This method assumes small rotations per iteration. For large
/// rotations, use multiple ICP iterations.
///
/// @tparam Policy0 The policy type for the source point set.
/// @tparam Policy1 The policy type for the target point set (must have normals).
/// @param X_ The source point set.
/// @param Y_ The target point set with normals.
/// @param state Reusable workspace buffers.
/// @return A transformation that aligns X to Y.
template <typename Policy0, typename Policy1>
auto fit_rigid_alignment_point_to_plane(
    const tf::points<Policy0> &X_, const tf::points<Policy1> &Y_,
    plane_alignment_state<tf::coordinate_type<Policy0, Policy1>> &state)
    -> tf::transformation<tf::coordinate_type<Policy0, Policy1>, 3> {
  using T = tf::coordinate_type<Policy0, Policy1>;
  constexpr std::size_t Dims = tf::coordinate_dims_v<Policy0>;
  static_assert(Dims == 3, "Point-to-plane alignment requires 3D points");
  static_assert(Dims == tf::coordinate_dims_v<Policy1>,
                "Point sets must have the same dimensionality");

  // Extract plain points, frames, and normals
  const auto &X = X_ | tf::plain();
  const auto &Y = Y_ | tf::plain();
  const auto &normals = Y_.normals();
  const auto &fX = tf::frame_of(X_);
  const auto &fY = tf::frame_of(Y_);

  const std::size_t n = X.size();
  constexpr std::size_t cols = 6;

  // Allocate buffers
  state.A.allocate(n * cols);
  state.b_vec.allocate(n);

  // Build system in parallel: A * [rx, ry, rz, tx, ty, tz]^T = b
  //
  // Linearized point-to-plane: for each correspondence (x_i, y_i, n_i)
  //   ((R*x_i + t) - y_i) · n_i ≈ 0
  //
  // With R ≈ I + [r]× (small angle):
  //   (x_i + [r]× x_i + t - y_i) · n_i = 0
  //   (x_i × n_i) · r + n_i · t = (y_i - x_i) · n_i
  //
  // Each row of A: [cross_x, cross_y, cross_z, nx, ny, nz]
  // Each element of b: dot(y - x, n)

  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&](std::size_t i) {
        // Transform points to world space
        auto x = tf::transformed(X[i], fX);
        auto y = tf::transformed(Y[i], fY);

        // Transform normal to world space (rotation only)
        auto normal = tf::transformed_normal(normals[i], fY);

        // cross = x × n
        auto cross = tf::cross(x.as_vector_view(), normal);

        // Column-major: A[i + col*n]
        state.A[i + 0 * n] = cross[0];
        state.A[i + 1 * n] = cross[1];
        state.A[i + 2 * n] = cross[2];
        state.A[i + 3 * n] = normal[0];
        state.A[i + 4 * n] = normal[1];
        state.A[i + 5 * n] = normal[2];

        // b[i] = (y - x) · n
        state.b_vec[i] = tf::dot(y - x, normal);
      },
      tf::checked);

  // Solve least squares using parallel TSQR
  std::array<T, cols> coeffs;
  tf::linalg::solve_least_squares_parallel(state.A.data(), state.b_vec.data(),
                                           coeffs.data(), n, cols,
                                           state.solver_state);

  // Build transformation from Rodrigues vector and translation
  // coeffs = [rx, ry, rz, tx, ty, tz]
  auto result =
      tf::make_rotation_from_rodrigues(coeffs[0], coeffs[1], coeffs[2]);

  // Set translation
  result(0, 3) = coeffs[3];
  result(1, 3) = coeffs[4];
  result(2, 3) = coeffs[5];

  return result;
}

/// @brief Point-to-plane rigid alignment (allocates internally).
///
/// Convenience overload that allocates workspace internally.
/// For repeated calls (e.g., ICP loop), prefer the overload with
/// explicit state to avoid repeated allocations.
///
/// @tparam Policy0 The policy type for the source point set.
/// @tparam Policy1 The policy type for the target point set (must have normals).
/// @param X The source point set.
/// @param Y The target point set with normals.
/// @return A transformation that aligns X to Y.
template <typename Policy0, typename Policy1>
auto fit_rigid_alignment_point_to_plane(const tf::points<Policy0> &X,
                                        const tf::points<Policy1> &Y)
    -> tf::transformation<tf::coordinate_type<Policy0, Policy1>, 3> {
  plane_alignment_state<tf::coordinate_type<Policy0, Policy1>> state;
  return fit_rigid_alignment_point_to_plane(X, Y, state);
}

} // namespace tf::geometry
