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

#include "./laplacian_smoothed.hpp"

namespace tf {

/// @ingroup geometry_processing
/// @brief Apply Taubin smoothing to a point set.
///
/// Taubin smoothing alternates between shrinking (positive lambda) and
/// inflating (negative mu) passes to smooth without significant volume loss.
/// This avoids the shrinkage problem of standard Laplacian smoothing.
///
/// The mu parameter is computed internally from lambda and the pass-band
/// frequency kpb (default 0.1):
///   mu = 1 / (kpb - 1/lambda)
///
/// With lambda=0.5 and kpb=0.1, mu ≈ -0.526.
///
/// @param pts Point set with vertex_link policy attached.
/// @param iterations Number of smoothing iterations (each iteration is one
///                   shrink + one inflate pass).
/// @param lambda Smoothing factor for shrink pass. Default: 0.5.
/// @param kpb Pass-band frequency. Default: 0.1.
/// @return New points buffer with smoothed positions.
template <typename Policy>
auto taubin_smoothed(const tf::points<Policy> &pts, std::size_t iterations,
                     tf::coordinate_type<Policy> lambda = 0.5,
                     tf::coordinate_type<Policy> kpb = 0.1) {
  static_assert(tf::has_vertex_link_policy<Policy>,
                "Points must have vertex_link policy attached");

  using T = tf::coordinate_type<Policy>;
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;

  // Compute mu from lambda and pass-band frequency
  // mu = 1 / (kpb - 1/lambda)
  T mu = T(1) / (kpb - T(1) / lambda);

  const auto &vlink = pts.vertex_link();

  tf::points_buffer<T, Dims> current;
  current.allocate(pts.size());
  tf::parallel_copy(pts, current.points());

  tf::points_buffer<T, Dims> next;
  next.allocate(pts.size());

  auto smooth_pass = [&](T weight) {
    tf::parallel_for_each(
        tf::zip(current.points(), next.points(),
                tf::make_block_indirect_range(vlink, current.points())),
        [&](auto tup) {
          auto [curr, out, neighbors] = tup;
          out = laplacian_smoothed(curr, tf::make_points(neighbors), weight);
        },
        tf::checked);
    std::swap(current, next);
  };

  for (std::size_t iter = 0; iter < iterations; ++iter) {
    smooth_pass(lambda); // Shrink
    smooth_pass(mu);     // Inflate
  }

  return current;
}

} // namespace tf
