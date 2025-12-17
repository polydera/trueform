/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../iter/point_iterator.hpp"
#include "../range.hpp"

namespace tf::views {
template <std::size_t Dims, typename Range> auto make_points(Range &&r) {
  auto begin = tf::iter::make_point_iterator<Dims>(r.begin());
  auto end = tf::iter::make_point_iterator<Dims>(r.end());
  return tf::make_range(std::move(begin), std::move(end));
}
} // namespace tf::views
