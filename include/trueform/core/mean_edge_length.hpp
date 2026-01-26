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
#include "./coordinate_type.hpp"
#include "./distance.hpp"
#include "./polygons.hpp"
#include "./views/mapped_range.hpp"
#include "./segments.hpp"

namespace tf {
/// @ingroup core_queries
/// @brief Computes the mean edge length of a polygon collection.
/// @tparam Policy The polygons policy.
/// @param polygons The polygon collection.
/// @return The mean edge length.
template <typename Policy>
auto mean_edge_length(const tf::polygons<Policy> &polygons) {
  auto [total_edge_length, n_edges] = tf::reduce(
      tf::make_mapped_range(
          polygons,
          [&](const auto &polygon) {
            std::pair<tf::coordinate_type<Policy>, std::size_t> out{
                0, polygon.size()};
            std::size_t prev = out.second;
            for (std::size_t i = 0; i < out.second; prev = i++)
              out.first += distance(polygon[prev], polygon[i]);
            return out;
          }),
      [](const auto &x, const auto &y) {
        return std::make_pair(x.first + y.first, x.second + y.second);
      },
      std::pair<tf::coordinate_type<Policy>, std::size_t>{0, 0}, tf::checked);
  return total_edge_length / n_edges;
}

/// @ingroup core_queries
/// @brief Computes the mean edge length of a segment collection.
/// @tparam Policy The segments policy.
/// @param segments The segment collection.
/// @return The mean edge length.
template <typename Policy>
auto mean_edge_length(const tf::segments<Policy> &segments) {
  return tf::reduce(
             tf::make_mapped_range(segments,
                                   [&](const auto &segment) {
                                     return distance(segment[0], segment[1]);
                                   }),
             std::plus<>{}, tf::coordinate_type<Policy>{}, tf::checked) /
         segments.size();
}

} // namespace tf
