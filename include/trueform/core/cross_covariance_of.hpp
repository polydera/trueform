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
#include "./centroid.hpp"
#include "./coordinate_type.hpp"
#include "./points.hpp"
#include "./views/zip.hpp"

namespace tf {

/// @ingroup core_properties
/// @brief Compute the cross-covariance matrix between two point sets.
///
/// Computes H = Σ (x_i - centroid_x) ⊗ (y_i - centroid_y)^T
/// This is useful for point cloud registration (Procrustes/Kabsch algorithm).
///
/// @tparam Policy0 The policy type for the first point set.
/// @tparam Policy1 The policy type for the second point set.
/// @param X The source point set.
/// @param Y The target point set.
/// @return A tuple of (centroid_x, centroid_y, cross_covariance_matrix).
template <typename Policy0, typename Policy1>
auto cross_covariance_of(const tf::points<Policy0> &X,
                         const tf::points<Policy1> &Y) {
  using T = tf::coordinate_type<Policy0, Policy1>;
  constexpr std::size_t Dims = tf::coordinate_dims_v<Policy0>;
  static_assert(Dims == tf::coordinate_dims_v<Policy1>,
                "Point sets must have the same dimensionality");

  auto cx = tf::centroid(X);
  auto cy = tf::centroid(Y);

  std::array<std::array<T, Dims>, Dims> H{};
  for (std::size_t i = 0; i < Dims; ++i)
    for (std::size_t j = 0; j < Dims; ++j)
      H[i][j] = T(0);

  H = tf::reduce(
      tf::zip(X, Y),
      [&cx, &cy](auto acc, const auto &element) {
        constexpr std::size_t Dims = tf::coordinate_dims_v<Policy0>;
        using ElementType = std::decay_t<decltype(element)>;
        using AccType = std::decay_t<decltype(acc)>;

        if constexpr (std::is_same_v<ElementType, AccType>) {
          // Merging two partial matrices
          for (std::size_t i = 0; i < Dims; ++i)
            for (std::size_t j = 0; j < Dims; ++j)
              acc[i][j] += element[i][j];
        } else {
          // Adding a point pair's contribution
          const auto &[x, y] = element;
          for (std::size_t i = 0; i < Dims; ++i)
            for (std::size_t j = 0; j < Dims; ++j)
              acc[i][j] += (x[i] - cx[i]) * (y[j] - cy[j]);
        }

        return acc;
      },
      H, tf::checked);

  auto n = X.size();
  for (std::size_t i = 0; i < Dims; ++i)
    for (std::size_t j = 0; j < Dims; ++j)
      H[i][j] /= n + (n == 0);

  return std::make_tuple(cx, cy, H);
}

} // namespace tf
