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
#include "./point.hpp"
#include "./point_like.hpp"
#include "./points.hpp"
#include "./polygon.hpp"
#include "./polygons.hpp"
#include "./segment.hpp"
#include "./segments.hpp"

namespace tf {

/// @ingroup core_properties
/// @brief Compute the centroid of a point (identity).
///
/// Returns a copy of the input point. Provided for generic code
/// that computes centroids of arbitrary primitives.
template <std::size_t Dims, typename Policy>
auto centroid(const tf::point_like<Dims, Policy> &point) {
  return tf::point<tf::coordinate_type<Policy>, Dims>{point};
}

/// @ingroup core_properties
/// @brief Compute the centroid of a polygon.
///
/// Returns the arithmetic mean of all vertices.
template <std::size_t Dims, typename Policy>
auto centroid(const tf::polygon<Dims, Policy> &poly) {
  tf::point<tf::coordinate_type<Policy>, Dims> out = tf::zero;
  auto out_v = out.as_vector_view();
  for (const auto &pt : poly)
    out_v += pt.as_vector_view();
  out_v /= poly.size();
  return out;
}

/// @ingroup core_properties
/// @brief Compute the centroid (midpoint) of a segment.
template <std::size_t Dims, typename Policy>
auto centroid(const tf::segment<Dims, Policy> &seg) {
  tf::point<tf::coordinate_type<Policy>, Dims> out = seg[0];
  auto out_v = out.as_vector_view();
  out_v += seg[1];
  out_v /= 2;
  return out;
}

/// @ingroup core_properties
/// @brief Compute the centroid of a range of points.
///
/// Returns the arithmetic mean of all points in the range.
template <typename Policy> auto centroid(const tf::points<Policy> &pts) {
  constexpr auto Dims = tf::static_size_v<typename Policy::value_type>;
  tf::vector<tf::coordinate_type<Policy>, Dims> out_v = tf::zero;
  tf::point<tf::coordinate_type<Policy>, Dims> out;
  out.as_vector_view() =
      tf::reduce(pts.as_vector_view(), std::plus<>{}, out_v, tf::checked) /
      pts.size();
  return out;
}

/// @ingroup core_properties
/// @brief Compute the mean of a range of vectors.
template <typename Policy> auto centroid(const tf::vectors<Policy> &vcs) {
  constexpr auto Dims = tf::static_size_v<typename Policy::value_type>;
  tf::vector<tf::coordinate_type<Policy>, Dims> out_v = tf::zero;
  return tf::reduce(vcs, std::plus<>{}, out_v, tf::checked) / vcs.size();
}

/// @ingroup core_properties
/// @brief Compute the centroid of all vertices across a range of polygons.
///
/// Computes the weighted average where each vertex contributes equally,
/// regardless of which polygon it belongs to.
template <typename Policy> auto centroid(const tf::polygons<Policy> &polygons) {
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;
  using T = tf::coordinate_type<Policy>;

  // Map each polygon to (vertex_count, sum_of_vertices)
  auto polygon_data = tf::make_mapped_range(polygons, [](const auto &poly) {
    constexpr auto Dims = tf::coordinate_dims_v<Policy>;
    tf::vector<T, Dims> sum = tf::zero;
    for (const auto &pt : poly)
      sum += pt.as_vector_view();
    return std::pair{poly.size(), sum};
  });

  // Reduce to get (total_vertex_count, total_sum)
  std::pair<std::size_t, tf::vector<T, Dims>> init;
  init.first = 0;
  init.second = tf::zero;

  auto result = tf::reduce(
      polygon_data,
      [](auto acc, const auto &data) {
        acc.first += data.first;
        acc.second += data.second;
        return acc;
      },
      init, tf::checked);

  // Compute centroid
  tf::point<T, Dims> out;
  out.as_vector_view() = result.second / result.first;

  return out;
}

/// @ingroup core_properties
/// @brief Compute the centroid of all endpoints across a range of segments.
template <typename Policy> auto centroid(const tf::segments<Policy> &segments) {
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;
  using T = tf::coordinate_type<Policy>;

  auto segment_data = tf::make_mapped_range(segments, [](const auto &seg) {
    tf::vector<T, Dims> sum = seg[0].as_vector_view() + seg[1].as_vector_view();
    return sum;
  });

  // Reduce to get (total_vertex_count, total_sum)
  tf::vector<T, Dims> init = tf::zero;

  auto result = tf::reduce(segment_data, std::plus<>{}, init, tf::checked);

  // Compute centroid
  tf::point<T, Dims> out;
  out.as_vector_view() = result / (segments.size() * 2);

  return out;
}
} // namespace tf
