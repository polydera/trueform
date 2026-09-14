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

#include "nd_array.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace tf::cpp::test {

namespace detail {

template <typename Real>
auto radical_inverse(std::size_t index, std::size_t base) -> Real {
  auto inverse_base = Real{1} / static_cast<Real>(base);
  auto factor = inverse_base;
  auto value = Real{0};
  while (index != 0) {
    value += static_cast<Real>(index % base) * factor;
    index /= base;
    factor *= inverse_base;
  }
  return value;
}

inline auto checked_grid_size(int x, int y, int z) -> std::size_t {
  if (x < 0 || y < 0 || z < 0)
    throw std::invalid_argument("grid dimensions must be nonnegative");
  const auto sx = static_cast<std::size_t>(x);
  const auto sy = static_cast<std::size_t>(y);
  const auto sz = static_cast<std::size_t>(z);
  if (sy != 0 && sx > std::numeric_limits<std::size_t>::max() / sy)
    throw std::overflow_error("grid point count overflow");
  const auto xy = sx * sy;
  if (sz != 0 && xy > std::numeric_limits<std::size_t>::max() / sz)
    throw std::overflow_error("grid point count overflow");
  return xy * sz;
}

} // namespace detail

template <typename Real>
auto grid_points(int x_count, int y_count, int z_count, Real spacing = Real{1},
                 Real origin_x = Real{0}, Real origin_y = Real{0},
                 Real origin_z = Real{0}) -> tf::cpp::nd_array<Real> {
  static_assert(std::is_floating_point<Real>::value,
                "deterministic points require floating-point coordinates");
  const auto count = detail::checked_grid_size(x_count, y_count, z_count);
  if (count > std::numeric_limits<std::size_t>::max() / 3)
    throw std::overflow_error("grid coordinate count overflow");
  if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("grid point count exceeds array shape capacity");

  std::vector<Real> values;
  values.reserve(count * 3);
  for (int z = 0; z < z_count; ++z)
    for (int y = 0; y < y_count; ++y)
      for (int x = 0; x < x_count; ++x) {
        values.push_back(origin_x + static_cast<Real>(x) * spacing);
        values.push_back(origin_y + static_cast<Real>(y) * spacing);
        values.push_back(origin_z + static_cast<Real>(z) * spacing);
      }
  return make_nd_array(values, {static_cast<int>(count), 3});
}

/// Points sampled deterministically along a three-dimensional circular spiral.
template <typename Real>
auto spiral_points(int count, Real radius = Real{1}, Real height = Real{1},
                   Real turns = Real{3}) -> tf::cpp::nd_array<Real> {
  static_assert(std::is_floating_point<Real>::value,
                "deterministic points require floating-point coordinates");
  if (count < 0)
    throw std::invalid_argument("point count must be nonnegative");

  constexpr auto pi =
      static_cast<Real>(3.141592653589793238462643383279502884L);
  std::vector<Real> values;
  values.reserve(static_cast<std::size_t>(count) * 3);
  for (int point = 0; point < count; ++point) {
    const auto parameter =
        count > 1 ? static_cast<Real>(point) / static_cast<Real>(count - 1)
                  : Real{0};
    const auto angle = Real{2} * pi * turns * parameter;
    values.push_back(radius * static_cast<Real>(std::cos(angle)));
    values.push_back(radius * static_cast<Real>(std::sin(angle)));
    values.push_back(height * parameter);
  }
  return make_nd_array(values, {count, 3});
}

/// Three-dimensional Halton points using bases 2, 3, and 5.
template <typename Real>
auto low_discrepancy_points(int count) -> tf::cpp::nd_array<Real> {
  static_assert(std::is_floating_point<Real>::value,
                "deterministic points require floating-point coordinates");
  if (count < 0)
    throw std::invalid_argument("point count must be nonnegative");

  std::vector<Real> values;
  values.reserve(static_cast<std::size_t>(count) * 3);
  for (int point = 0; point < count; ++point) {
    const auto index = static_cast<std::size_t>(point) + 1;
    values.push_back(detail::radical_inverse<Real>(index, 2));
    values.push_back(detail::radical_inverse<Real>(index, 3));
    values.push_back(detail::radical_inverse<Real>(index, 5));
  }
  return make_nd_array(values, {count, 3});
}

} // namespace tf::cpp::test
