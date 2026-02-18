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
#include "./sqrt.hpp"
#include "./transformed.hpp"
#include "./views/mapped_range.hpp"

namespace tf {
/// @ingroup core_queries
/// @brief Computes the maximum edge length of a polygon collection.
/// @tparam Policy The polygons policy.
/// @param polygons The polygon collection.
/// @return The maximum edge length.
template <typename Policy>
auto max_edge_length(const tf::polygons<Policy> &polygons) {
  auto frame = tf::frame_of(polygons);
  return tf::sqrt(tf::reduce(
      tf::make_mapped_range(
          polygons,
          [&frame](const auto &polygon) {
            tf::coordinate_type<Policy> max_len2 = 0;
            std::size_t n = polygon.size();
            std::size_t prev = n - 1;
            for (std::size_t i = 0; i < n; prev = i++)
              max_len2 = std::max(
                  max_len2,
                  tf::transformed(polygon[i] - polygon[prev], frame).length2());
            return max_len2;
          }),
      [](const auto &x, const auto &y) { return std::max(x, y); },
      tf::coordinate_type<Policy>{}, tf::checked));
}

/// @ingroup core_queries
/// @brief Computes the maximum edge length of a segment collection.
/// @tparam Policy The segments policy.
/// @param segments The segment collection.
/// @return The maximum edge length.
template <typename Policy>
auto max_edge_length(const tf::segments<Policy> &segments) {
  auto frame = tf::frame_of(segments);
  return tf::sqrt(tf::reduce(
      tf::make_mapped_range(
          segments,
          [&frame](const auto &segment) {
            return tf::transformed(segment[1] - segment[0], frame).length2();
          }),
      [](const auto &x, const auto &y) { return std::max(x, y); },
      tf::coordinate_type<Policy>{}, tf::checked));
}

} // namespace tf
