/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./algorithm/reduce.hpp"
#include "./point.hpp"
#include "./point_like.hpp"
#include "./points.hpp"
#include "./polygon.hpp"
#include "./segment.hpp"

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
      tf::reduce(pts.as_vector_view(), std::plus<>{}, out_v) / pts.size();
  return out;
}
} // namespace tf
