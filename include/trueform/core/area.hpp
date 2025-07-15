/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./algorithm/circular_increment.hpp"
#include "./algorithm/reduce.hpp"
#include "./polygon.hpp"
#include "./polygons.hpp"
#include "./sqrt.hpp"
#include "./vector.hpp"
#include "./views/mapped_range.hpp"

namespace tf {
template <typename Policy>
auto signed_area(const tf::polygon<2, Policy> &_poly) {
  auto size = _poly.size();
  tf::coordinate_type<Policy> area = 0;
  decltype(size) prev = size - 1;

  for (decltype(size) i = 0; i < size; prev = i++) {
    auto &&point0 = _poly[prev];
    auto &&point1 = _poly[i];
    area += (point1[1] + point0[1]) * (point0[0] - point1[0]);
  }
  return area / 2;
}

template <typename Policy> auto area(const tf::polygon<2, Policy> &_poly) {
  return std::abs(tf::signed_area(_poly));
}

template <typename Policy> auto area2(const tf::polygon<2, Policy> &_poly) {
  auto sa = tf::signed_area(_poly);
  return sa * sa;
}

template <std::size_t N, typename Policy>
auto area2(const tf::polygon<N, Policy> &_poly) {
  using scalar_t = tf::coordinate_type<Policy>;
  using vec_t = tf::vector<scalar_t, N>;

  const auto size = _poly.size();

  vec_t normal{}; // Newell's method accumulator

  for (std::size_t i = 0, prev = size - 1; i < size; prev = i++) {
    const auto &p0 = _poly[prev];
    const auto &p1 = _poly[i];

    for (std::size_t j = 0; j < N; ++j) {
      const auto jp1 = tf::circular_increment(j, N);   // (j+1) % N
      const auto jp2 = tf::circular_increment(jp1, N); // (j+2) % N
      normal[j] += (p0[jp1] - p1[jp1]) * (p0[jp2] + p1[jp2]);
    }
  }

  return normal.length2() / 4;
}

template <std::size_t N, typename Policy>
auto area(const tf::polygon<N, Policy> &_poly) {
  return tf::sqrt(tf::area2(_poly));
}

template <typename Policy> auto area(const tf::polygons<Policy> &polys) {
  return tf::reduce(tf::make_mapped_range(
                        polys, [](const auto &poly) { return tf::area(poly); }),
                    std::plus<>{}, tf::coordinate_type<Policy>(0));
}

template <typename Policy> auto area2(const tf::polygons<Policy> &polys) {
  auto area = tf::area(polys);
  return area * area;
}
} // namespace tf
