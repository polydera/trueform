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
#include "./distance.hpp"
#include "./rss_like.hpp"

namespace tf {

/// @ingroup core_queries
/// @brief Compute distance metrics between two RSS bounding volumes.
///
/// Returns the squared minimum distance and squared maximum distance
/// between the two RSS volumes. Useful for BVH traversal decisions.
///
/// @tparam Dims The coordinate dimensions.
/// @tparam Policy0 The first RSS policy type.
/// @tparam Policy1 The second RSS policy type.
/// @param rss0 The first RSS bounding volume.
/// @param rss1 The second RSS bounding volume.
/// @return A pair of (min distance squared, max distance squared).
template <std::size_t Dims, typename Policy0, typename Policy1>
auto rss_metrics(const tf::rss_like<Dims, Policy0> &rss0,
                 const tf::rss_like<Dims, Policy1> &rss1) {
  using T = tf::coordinate_type<Policy0, Policy1>;
  static_assert(Dims == 2 || Dims == 3,
                "rss_metrics is implemented for 2D and 3D only.");

  using std::max;
  auto mind2 = distance2(rss0, rss1);

  // Compute centers (loops over Dims-1 axes for the base shape)
  auto center0 = rss0.center();
  auto center1 = rss1.center();
  auto center_diff = center1 - center0;
  T center_dist2 = center_diff.length2();
  T maxd2 = max(mind2, center_dist2);

  return std::make_pair(mind2, maxd2);
}
} // namespace tf
