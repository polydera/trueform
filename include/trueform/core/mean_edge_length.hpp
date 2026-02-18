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
#include "./frame_of.hpp"
#include "./polygons.hpp"
#include "./segments.hpp"
#include "./transformed.hpp"
#include "./views/mapped_range.hpp"

namespace tf {
/// @ingroup core_queries
/// @brief Computes the mean edge length of a polygon collection.
/// @tparam Policy The polygons policy.
/// @param polygons The polygon collection.
/// @return The mean edge length.
template <typename Policy>
auto mean_edge_length(const tf::polygons<Policy> &polygons) {
  auto frame = tf::frame_of(polygons);
  auto [total_edge_length, n_edges] = tf::reduce(
      tf::make_mapped_range(
          polygons,
          [&frame](const auto &polygon) {
            std::pair<tf::coordinate_type<Policy>, std::size_t> out{
                0, polygon.size()};
            std::size_t prev = out.second - 1;
            for (std::size_t i = 0; i < out.second; prev = i++)
              out.first +=
                  tf::transformed(polygon[i] - polygon[prev], frame).length();
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
  auto frame = tf::frame_of(segments);
  return tf::reduce(
             tf::make_mapped_range(
                 segments,
                 [&frame](const auto &segment) {
                   return tf::transformed(segment[1] - segment[0], frame)
                       .length();
                 }),
             std::plus<>{}, tf::coordinate_type<Policy>{}, tf::checked) /
         segments.size();
}

} // namespace tf
