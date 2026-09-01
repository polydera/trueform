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

#include "../core/none.hpp"
#include "../core/points.hpp"
#include "../core/views/mapped_range.hpp"
#include "../exact_coordinate_converter.hpp"
#include <utility>

namespace tf {

/// @ingroup exact_points
/// @brief View of `points` mapped through a converter to its integer space.
template <typename Policy, typename IntT, typename RealT, std::size_t Dims>
auto make_exact_points(
    const tf::points<Policy> &points,
    const tf::exact_coordinate_converter<IntT, RealT, Dims> &converter) {
  return tf::make_points(tf::make_mapped_range(
      points, [converter](const auto &p) { return converter(p); }));
}

/// @ingroup exact_points
/// @brief Build a converter from `points` and return (view, converter).
template <typename IntT = tf::none_t, typename Policy>
auto make_exact_points(const tf::points<Policy> &points) {
  auto converter = tf::make_exact_coordinate_converter<IntT>(points);
  auto view = make_exact_points(points, converter);
  return std::make_pair(std::move(view), std::move(converter));
}

} // namespace tf
