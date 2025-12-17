/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../reindex/points.hpp"
#include "../reindex/return_index_map.hpp"
#include "./index_map/points.hpp"

namespace tf {
template <typename Index, typename Policy>
auto cleaned(const tf::points<Policy> &points,
             tf::coordinate_type<Policy> tolerance, tf::return_index_map_t) {
  auto im = tf::make_clean_index_map<Index>(points, tolerance);
  auto out = tf::reindexed(points, im);
  return std::make_pair(std::move(out), std::move(im));
}

template <typename Index, typename Policy>
auto cleaned(const tf::points<Policy> &points, tf::return_index_map_t) {
  auto im = tf::make_clean_index_map<Index>(points);
  auto out = tf::reindexed(points, im);
  return std::make_pair(std::move(out), std::move(im));
}

template <typename Index, typename Policy>
auto cleaned(const tf::points<Policy> &points,
             tf::coordinate_type<Policy> tolerance) {
  auto im = tf::make_clean_index_map<Index>(points, tolerance);
  auto out = tf::reindexed(points, im);
  return out;
}

template <typename Index, typename Policy>
auto cleaned(const tf::points<Policy> &points) {
  auto im = tf::make_clean_index_map<Index>(points);
  auto out = tf::reindexed(points, im);
  return out;
}
} // namespace tf
