/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
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

template <typename Policy>
auto cleaned(const tf::points<Policy> &points,
             tf::coordinate_type<Policy> tolerance, tf::return_index_map_t) {
  using index_t = typename Policy::size_type;
  return cleaned<index_t>(points, tolerance, tf::return_index_map);
}
template <typename Index, typename Policy>
auto cleaned(const tf::points<Policy> &points, tf::return_index_map_t) {
  auto im = tf::make_clean_index_map<Index>(points);
  auto out = tf::reindexed(points, im);
  return std::make_pair(std::move(out), std::move(im));
}

template <typename Policy>
auto cleaned(const tf::points<Policy> &points, tf::return_index_map_t) {
  using index_t = typename Policy::size_type;
  return cleaned<index_t>(points, tf::return_index_map);
}

template <typename Index, typename Policy>
auto cleaned(const tf::points<Policy> &points,
             tf::coordinate_type<Policy> tolerance) {
  auto im = tf::make_clean_index_map<Index>(points, tolerance);
  auto out = tf::reindexed(points, im);
  return out;
}

template <typename Policy>
auto cleaned(const tf::points<Policy> &points,
             tf::coordinate_type<Policy> tolerance) {
  using index_t = typename Policy::size_type;
  return cleaned<index_t>(points, tolerance);
}

template <typename Index, typename Policy>
auto cleaned(const tf::points<Policy> &points) {
  auto im = tf::make_clean_index_map<Index>(points);
  auto out = tf::reindexed(points, im);
  return out;
}

template <typename Policy> auto cleaned(const tf::points<Policy> &points) {
  using index_t = typename Policy::size_type;
  return cleaned<index_t>(points);
}
} // namespace tf
