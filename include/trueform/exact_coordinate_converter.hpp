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

#include "./core/aabb_like.hpp"
#include "./core/none.hpp"
#include "./core/points.hpp"
#include "./core/polygons.hpp"
#include "./core/range.hpp"
#include "./core/segments.hpp"
#include "./exact/int32.hpp"
#include "./exact/pt_converter.hpp"

namespace tf {

/// @ingroup exact_coordinate_converter
/// @brief Conversion of point coordinates to an exact integer grid that
///        fills the full int range (~0.99 * max for the target int type).
///        Identity for integer input.
template <typename IntT, typename RealT, std::size_t Dims>
using exact_coordinate_converter = tf::exact::pt_converter<IntT, RealT, Dims>;

/// @ingroup exact_coordinate_converter
/// @brief Build a converter from an AABB.
template <typename IntT = tf::exact::int32, typename RealT = tf::none_t,
          std::size_t Dims, typename Policy>
auto make_exact_coordinate_converter(const tf::aabb_like<Dims, Policy> &aabb) {
  return tf::exact::make_pt_converter<IntT, RealT>(aabb);
}

/// @ingroup exact_coordinate_converter
/// @brief Build a converter from a points range.
template <typename IntT = tf::exact::int32, typename RealT = tf::none_t,
          typename Policy>
auto make_exact_coordinate_converter(const tf::points<Policy> &pts) {
  return tf::exact::make_pt_converter<IntT, RealT>(pts);
}

/// @ingroup exact_coordinate_converter
/// @brief Build a converter from a polygons range.
template <typename IntT = tf::exact::int32, typename RealT = tf::none_t,
          typename Policy>
auto make_exact_coordinate_converter(const tf::polygons<Policy> &form) {
  return tf::exact::make_pt_converter<IntT, RealT>(form);
}

/// @ingroup exact_coordinate_converter
/// @brief Build a converter from a segments range.
template <typename IntT = tf::exact::int32, typename RealT = tf::none_t,
          typename Policy>
auto make_exact_coordinate_converter(const tf::segments<Policy> &form) {
  return tf::exact::make_pt_converter<IntT, RealT>(form);
}

/// @ingroup exact_coordinate_converter
/// @brief Build a converter spanning two polygon forms.
template <typename IntT = tf::exact::int32, typename RealT = tf::none_t,
          typename Policy0, typename Policy1>
auto make_exact_coordinate_converter(const tf::polygons<Policy0> &form0,
                                     const tf::polygons<Policy1> &form1) {
  return tf::exact::make_pt_converter<IntT, RealT>(form0, form1);
}

/// @ingroup exact_coordinate_converter
/// @brief Build a converter spanning a range of forms.
template <typename IntT = tf::exact::int32, typename RealT = tf::none_t,
          typename Iterator, std::size_t N>
auto make_exact_coordinate_converter(tf::range<Iterator, N> forms) {
  return tf::exact::make_pt_converter<IntT, RealT>(forms);
}

} // namespace tf
