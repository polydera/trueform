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

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/core/buffer.hpp>
#include <trueform/core/distance.hpp>
#include <trueform/core/algorithm/parallel_transform.hpp>
#include <trueform/core/points.hpp>
#include <trueform/core/range.hpp>
#include <trueform/python/util/make_numpy_array.hpp>

namespace tf::py {

/// @brief Compute distance field from points to a primitive (vectorized, parallel)
/// @tparam Dims Dimensionality (2 or 3)
/// @tparam RealT Floating point type (float or double)
/// @tparam Primitive Type of geometric primitive
/// @param points_array Numpy array of points with shape (N, Dims)
/// @param primitive The target primitive
/// @return Numpy array of distances with shape (N,)
template <std::size_t Dims, typename RealT, typename Primitive>
auto distance_field_impl(
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1, Dims>>
        points_array,
    const Primitive &primitive) {

  // Create tf::points view from numpy array
  std::size_t num_points = points_array.shape(0);
  auto points = tf::make_points<Dims>(
      tf::make_range(points_array.data(), num_points * Dims));

  // Allocate result buffer
  tf::buffer<RealT> scalars;
  scalars.allocate(points.size());

  // Parallel distance computation using tf::distance_f
  tf::parallel_transform(points, scalars, tf::distance_f(primitive));

  // Release ownership from buffer and wrap as numpy array
  RealT *data_ptr = scalars.release();
  return make_numpy_array<nanobind::shape<-1>>(
      data_ptr, {static_cast<size_t>(points.size())});
}

auto register_core_distance_field(nanobind::module_ &m) -> void;

} // namespace tf::py
