/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
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

/// @ingroup core_properties
/// @brief Compute the signed area of a 2D polygon.
///
/// Uses the shoelace formula. The sign indicates winding order:
/// positive for counter-clockwise, negative for clockwise.
///
/// @tparam Policy The polygon's storage policy.
/// @param _poly A 2D polygon.
/// @return The signed area.
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

/// @ingroup core_properties
/// @brief Compute the area of a 2D polygon.
///
/// Returns the absolute value of the signed area.
template <typename Policy> auto area(const tf::polygon<2, Policy> &_poly) {
  return std::abs(tf::signed_area(_poly));
}

/// @ingroup core_properties
/// @brief Compute the squared area of a 2D polygon.
template <typename Policy> auto area2(const tf::polygon<2, Policy> &_poly) {
  auto sa = tf::signed_area(_poly);
  return sa * sa;
}

/// @ingroup core_properties
/// @brief Compute the squared area of an N-dimensional polygon.
///
/// Uses Newell's method to compute the normal vector and derives
/// the squared area from its squared length.
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

/// @ingroup core_properties
/// @brief Compute the area of an N-dimensional polygon.
template <std::size_t N, typename Policy>
auto area(const tf::polygon<N, Policy> &_poly) {
  return tf::sqrt(tf::area2(_poly));
}

/// @ingroup core_properties
/// @brief Compute the total area of a range of polygons.
template <typename Policy> auto area(const tf::polygons<Policy> &polys) {
  return tf::reduce(tf::make_mapped_range(
                        polys, [](const auto &poly) { return tf::area(poly); }),
                    std::plus<>{}, tf::coordinate_type<Policy>(0));
}

/// @ingroup core_properties
/// @brief Compute the squared total area of a range of polygons.
template <typename Policy> auto area2(const tf::polygons<Policy> &polys) {
  auto area = tf::area(polys);
  return area * area;
}
} // namespace tf
