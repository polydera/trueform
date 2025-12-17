/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./distance.hpp"
#include "./rss_like.hpp"

namespace tf {
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
