/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
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
template <std::size_t Dims, typename Policy>
auto centroid(const tf::point_like<Dims, Policy> &point) {
  return tf::point<tf::coordinate_type<Policy>, Dims>{point};
}

template <std::size_t Dims, typename Policy>
auto centroid(const tf::polygon<Dims, Policy> &poly) {
  tf::point<tf::coordinate_type<Policy>, Dims> out;
  auto out_v = out.as_vector_view();
  for (std::size_t i = 0; i < Dims; ++i)
    out[i] = 0;
  for (const auto &pt : poly)
    out_v += pt.as_vector_view();
  out_v /= poly.size();
  return out;
}

template <std::size_t Dims, typename Policy>
auto centroid(const tf::segment<Dims, Policy> &seg) {
  tf::point<tf::coordinate_type<Policy>, Dims> out = seg[0];
  auto out_v = out.as_vector_view();
  out_v += seg[1];
  out_v /= 2;
  return out;
}

template <typename Policy> auto centroid(const tf::points<Policy> &pts) {
  constexpr auto Dims = tf::static_size_v<typename Policy::value_type>;
  tf::vector<tf::coordinate_type<Policy>, Dims> out_v;
  for (std::size_t i = 0; i < Dims; ++i)
    out_v[i] = 0;
  tf::point<tf::coordinate_type<Policy>, Dims> out;
  out.as_vector_view() =
      tf::reduce(pts.as_vector_view(), std::plus<>{}, out_v, tf::checked) /
      pts.size();
  return out;
}

template <typename Policy> auto centroid(const tf::vectors<Policy> &vcs) {
  constexpr auto Dims = tf::static_size_v<typename Policy::value_type>;
  tf::vector<tf::coordinate_type<Policy>, Dims> out_v;
  for (std::size_t i = 0; i < Dims; ++i)
    out_v[i] = 0;
  return tf::reduce(vcs, std::plus<>{}, out_v, tf::checked) / vcs.size();
}

template <typename Policy> auto centroid(const tf::polygons<Policy> &polygons) {
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;
  using T = tf::coordinate_type<Policy>;

  // Map each polygon to (vertex_count, sum_of_vertices)
  auto polygon_data = tf::make_mapped_range(polygons, [](const auto &poly) {
    tf::vector<T, Dims> sum;
    for (std::size_t i = 0; i < Dims; ++i)
      sum[i] = 0;
    for (const auto &pt : poly)
      sum += pt.as_vector_view();
    return std::pair{poly.size(), sum};
  });

  // Reduce to get (total_vertex_count, total_sum)
  std::pair<std::size_t, tf::vector<T, Dims>> init;
  init.first = 0;
  for (std::size_t i = 0; i < Dims; ++i)
    init.second[i] = 0;

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

template <typename Policy> auto centroid(const tf::segments<Policy> &segments) {
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;
  using T = tf::coordinate_type<Policy>;

  auto segment_data = tf::make_mapped_range(segments, [](const auto &seg) {
    tf::vector<T, Dims> sum = seg[0].as_vector_view() + seg[1].as_vector_view();
    return sum;
  });

  // Reduce to get (total_vertex_count, total_sum)
  tf::vector<T, Dims> init;
  for (std::size_t i = 0; i < Dims; ++i)
    init[i] = 0;

  auto result = tf::reduce(segment_data, std::plus<>{}, init, tf::checked);

  // Compute centroid
  tf::point<T, Dims> out;
  out.as_vector_view() = result / (segments.size() * 2);

  return out;
}
} // namespace tf
