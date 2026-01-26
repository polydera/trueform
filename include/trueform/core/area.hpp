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
#include "./algorithm/circular_increment.hpp"
#include "./algorithm/reduce.hpp"
#include "./cross.hpp"
#include "./parallelogram_area.hpp"
#include "./polygon.hpp"
#include "./polygons.hpp"
#include "./sqrt.hpp"
#include "./static_size.hpp"
#include "./vector.hpp"
#include "./views/mapped_range.hpp"

namespace tf {

/// @ingroup core_properties
/// @brief Compute the signed area of a 2D polygon.
///
/// Uses the shoelace formula. The sign indicates winding order:
/// positive for counter-clockwise, negative for clockwise.
/// Optimized for triangles when the size is known at compile time.
///
/// @tparam Policy The polygon's storage policy.
/// @param _poly A 2D polygon.
/// @return The signed area.
template <typename Policy>
auto signed_area(const tf::polygon<2, Policy> &_poly) {
  if constexpr (tf::static_size_v<Policy> == 3) {
    const auto &p0 = _poly[0];
    const auto &p1 = _poly[1];
    const auto &p2 = _poly[2];
    return ((p1[0] - p0[0]) * (p2[1] - p0[1]) -
            (p2[0] - p0[0]) * (p1[1] - p0[1])) /
           2;
  } else {
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
}

/// @ingroup core_properties
/// @brief Compute the squared area of an N-dimensional polygon.
///
/// For 2D, uses signed_area squared. For higher dimensions, uses Newell's
/// method. Optimized for triangles when the size is known at compile time.
template <std::size_t N, typename Policy>
auto area2(const tf::polygon<N, Policy> &_poly) {
  if constexpr (N == 2) {
    auto sa = tf::signed_area(_poly);
    return sa * sa;
  } else {
    if constexpr (tf::static_size_v<Policy> == 3) {
      auto e0 = _poly[1] - _poly[0];
      auto e1 = _poly[2] - _poly[0];
      if constexpr (N == 3) {
        return tf::cross(e0, e1).length2() / 4;
      } else {
        return tf::parallelogram_area2(e0, e1) / 4;
      }
    } else {
      using scalar_t = tf::coordinate_type<Policy>;
      using vec_t = tf::vector<scalar_t, N>;

      const auto size = _poly.size();

      vec_t normal = tf::zero; // Newell's method accumulator

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
  }
}

/// @ingroup core_properties
/// @brief Compute the area of an N-dimensional polygon.
///
/// For 2D, returns absolute value of signed area. For higher dimensions,
/// returns sqrt of area2 (Newell's method).
template <std::size_t N, typename Policy>
auto area(const tf::polygon<N, Policy> &_poly) {
  if constexpr (N == 2) {
    return std::abs(tf::signed_area(_poly));
  } else {
    return tf::sqrt(tf::area2(_poly));
  }
}

/// @ingroup core_properties
/// @brief Compute the total area of a range of polygons.
template <typename Policy> auto area(const tf::polygons<Policy> &polys) {
  return tf::reduce(tf::make_mapped_range(
                        polys, [](const auto &poly) { return tf::area(poly); }),
                    std::plus<>{}, tf::coordinate_type<Policy>(0), tf::checked);
}

/// @ingroup core_properties
/// @brief Compute the squared total area of a range of polygons.
template <typename Policy> auto area2(const tf::polygons<Policy> &polys) {
  auto area = tf::area(polys);
  return area * area;
}
} // namespace tf
