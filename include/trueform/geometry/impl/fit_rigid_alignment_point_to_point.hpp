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

#include "../../core/cross_covariance_of.hpp"
#include "../../core/epsilon.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/policy/unwrap.hpp"
#include "../../core/sqrt.hpp"
#include "../../core/svd_of_symmetric.hpp"
#include "../../core/transformation.hpp"
#include "../../core/transformed.hpp"
#include "../../core/transformed_cross_covariance.hpp"
#include "../../core/vector.hpp"

namespace tf::geometry {

/// @brief Point-to-point rigid alignment using Kabsch/SVD algorithm.
///
/// Computes the optimal rigid transformation T such that T(X) ≈ Y
/// by minimizing the point-to-point distance: sum_i ||T(x_i) - y_i||².
///
/// @tparam Policy0 The policy type for the source point set.
/// @tparam Policy1 The policy type for the target point set.
/// @param X_ The source point set.
/// @param Y_ The target point set (must have same size as X).
/// @return A transformation that best aligns X to Y.
template <typename Policy0, typename Policy1>
auto fit_rigid_alignment_point_to_point(const tf::points<Policy0> &X_,
                                        const tf::points<Policy1> &Y_) {
  using T = tf::coordinate_type<Policy0, Policy1>;
  constexpr std::size_t Dims = tf::coordinate_dims_v<Policy0>;
  static_assert(Dims == tf::coordinate_dims_v<Policy1>,
                "Point sets must have the same dimensionality");
  static_assert(Dims == 2 || Dims == 3,
                "Only 2D and 3D point sets are supported");

  // Extract plain points and frames
  const auto &X = X_ | tf::plain();
  const auto &Y = Y_ | tf::plain();
  const auto &tX = tf::frame_of(X_).transformation();
  const auto &tY = tf::frame_of(Y_).transformation();

  // Compute cross-covariance on plain points
  auto [cx, cy, H] = tf::cross_covariance_of(X, Y);

  // Transform H and centroids to world space
  H = tf::core::transformed_cross_covariance(H, tX, tY);
  auto cx_world = tf::transformed(cx, tX);
  auto cy_world = tf::transformed(cy, tY);

  // HtH = H^T * H
  std::array<std::array<T, Dims>, Dims> HtH{};
  for (std::size_t i = 0; i < Dims; ++i)
    for (std::size_t j = 0; j < Dims; ++j)
      for (std::size_t k = 0; k < Dims; ++k)
        HtH[i][j] += H[k][i] * H[k][j];

  // sigma_sq: squared singular values of H (descending)
  // V: right singular vectors of H (columns), from eig(HtH)
  auto [sigma_sq, _, V] = tf::svd_of_symmetric(HtH);

  // Build R = V * U^T
  tf::transformation<T, Dims> out;
  for (std::size_t i = 0; i < Dims; ++i)
    for (std::size_t j = 0; j < Dims; ++j)
      out(i, j) = T(0);

  std::array<T, Dims> inv_sigma{};
  // Use relative threshold based on largest singular value for scale invariance
  const T threshold = sigma_sq[0] * tf::epsilon2<T>;
  for (std::size_t col = 0; col < Dims; ++col) {
    inv_sigma[col] = (sigma_sq[col] > threshold)
                         ? T(1) / tf::sqrt(sigma_sq[col])
                         : T(0);
  }

  // First compute R = V * U^T without reflection handling
  // For H = U Σ V^T, the Kabsch rotation is R = V * U^T
  for (std::size_t col = 0; col < Dims; ++col) {
    // u = H * v_col / sigma = left singular vector
    tf::vector<T, Dims> u;
    for (std::size_t i = 0; i < Dims; ++i) {
      u[i] = T(0);
      for (std::size_t k = 0; k < Dims; ++k)
        u[i] += H[i][k] * V[col][k];
    }

    // R += v_col * (inv_sigma[col] * u^T) = outer(v, u) → V * U^T
    const T a = inv_sigma[col];
    for (std::size_t i = 0; i < Dims; ++i)
      for (std::size_t j = 0; j < Dims; ++j)
        out(i, j) += V[col][i] * (a * u[j]);
  }

  // det(R)
  T det;
  if constexpr (Dims == 2) {
    det = out(0, 0) * out(1, 1) - out(0, 1) * out(1, 0);
  } else {
    det = out(0, 0) * (out(1, 1) * out(2, 2) - out(1, 2) * out(2, 1)) -
          out(0, 1) * (out(1, 0) * out(2, 2) - out(1, 2) * out(2, 0)) +
          out(0, 2) * (out(1, 0) * out(2, 1) - out(1, 1) * out(2, 0));
  }

  // Reflection fix: if det<0, flip the smallest singular direction (last col)
  if (det < T(0)) {
    // Rebuild R = V * U^T with a sign flip on the last column contribution
    for (std::size_t i = 0; i < Dims; ++i)
      for (std::size_t j = 0; j < Dims; ++j)
        out(i, j) = T(0);

    for (std::size_t col = 0; col < Dims; ++col) {
      const T flip = (col + 1 == Dims) ? T(-1) : T(1);

      tf::vector<T, Dims> u;
      for (std::size_t i = 0; i < Dims; ++i) {
        u[i] = T(0);
        for (std::size_t k = 0; k < Dims; ++k)
          u[i] += H[i][k] * V[col][k];
      }

      const T a = inv_sigma[col] * flip;
      for (std::size_t i = 0; i < Dims; ++i)
        for (std::size_t j = 0; j < Dims; ++j)
          out(i, j) += V[col][i] * (a * u[j]);
    }
  }

  // t = cy_world - R * cx_world
  for (std::size_t i = 0; i < Dims; ++i) {
    out(i, Dims) = cy_world[i];
    for (std::size_t j = 0; j < Dims; ++j)
      out(i, Dims) -= out(i, j) * cx_world[j];
  }

  return out;
}

} // namespace tf::geometry
