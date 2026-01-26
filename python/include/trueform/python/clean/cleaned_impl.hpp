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
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <trueform/clean.hpp>
#include <trueform/core/points.hpp>
#include <trueform/core/range.hpp>
#include <trueform/python/core/offset_blocked_array.hpp>
#include <trueform/python/util/make_numpy_array.hpp>

namespace tf::py {

template <typename Index, typename RealT, std::size_t Dims>
auto cleaned_impl(
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1, Dims>>
        points,
    std::optional<RealT> tolerance, tf::return_index_map_t) {
  // Create points range from numpy array
  std::size_t num_points = points.shape(0);
  auto points_range =
      tf::make_points<Dims>(tf::make_range(points.data(), num_points * Dims));
  auto [res, maps] = [&] {
    if (tolerance)
      return tf::cleaned<Index>(points_range, *tolerance, tf::return_index_map);
    else
      return tf::cleaned<Index>(points_range, tf::return_index_map);
  }();
  return nanobind::make_tuple(make_numpy_array(std::move(res)),
                              make_numpy_array(std::move(maps)));
}

template <typename Index, typename RealT, std::size_t V, std::size_t Dims>
auto cleaned_impl(nanobind::ndarray<nanobind::numpy, const RealT,
                                    nanobind::shape<-1, V, Dims>>
                      points,
                  std::optional<RealT> tolerance) {
  // Create points range from numpy array
  std::size_t num_points = points.shape(0) * V;
  auto points_range = tf::make_blocked_range<V>(
      tf::make_points<Dims>(tf::make_range(points.data(), num_points * Dims)));
  auto primitive_range = [&] {
    if constexpr (V == 2)
      return tf::make_segments(points_range);
    else
      return tf::make_polygons(points_range);
  }();
  if (tolerance)
    return make_numpy_array(tf::cleaned<Index>(primitive_range, *tolerance));
  else
    return make_numpy_array(tf::cleaned<Index>(primitive_range));
}

template <typename Index, std::size_t V, typename RealT, std::size_t Dims>
auto cleaned_impl(
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1, V>>
        indices,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1, Dims>>
        points,
    std::optional<RealT> tolerance, tf::return_index_map_t) {
  // Create points range from numpy array
  std::size_t num_points = points.shape(0);
  auto points_range =
      tf::make_points<Dims>(tf::make_range(points.data(), num_points * Dims));
  std::size_t num_indices = indices.shape(0);
  auto indices_range = tf::make_blocked_range<V>(
      tf::make_range(indices.data(), num_indices * V));
  auto primitive_range = [&] {
    if constexpr (V == 2)
      return tf::make_segments(indices_range, points_range);
    else
      return tf::make_polygons(indices_range, points_range);
  }();
  auto [res, i_map, p_map] = [&] {
    if (tolerance)
      return tf::cleaned<Index>(primitive_range, *tolerance,
                                tf::return_index_map);
    else
      return tf::cleaned<Index>(primitive_range, tf::return_index_map);
  }();
  return nanobind::make_tuple(make_numpy_array(std::move(res)),
                              make_numpy_array(std::move(i_map)),
                              make_numpy_array(std::move(p_map)));
}

// Dynamic indexed geometry cleaning (variable-sized polygons)
template <typename Index, typename RealT, std::size_t Dims>
auto cleaned_impl_dynamic(
    const offset_blocked_array_wrapper<Index, Index> &indices,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1, Dims>>
        points,
    std::optional<RealT> tolerance, tf::return_index_map_t) {
  // Create points range from numpy array
  std::size_t num_points = points.shape(0);
  auto points_range =
      tf::make_points<Dims>(tf::make_range(points.data(), num_points * Dims));
  auto indices_range = indices.make_range();
  auto primitive_range = tf::make_polygons(indices_range, points_range);
  auto [res, i_map, p_map] = [&] {
    if (tolerance)
      return tf::cleaned<Index>(primitive_range, *tolerance,
                                tf::return_index_map);
    else
      return tf::cleaned<Index>(primitive_range, tf::return_index_map);
  }();
  return nanobind::make_tuple(make_numpy_array(std::move(res)),
                              make_numpy_array(std::move(i_map)),
                              make_numpy_array(std::move(p_map)));
}

} // namespace tf::py
