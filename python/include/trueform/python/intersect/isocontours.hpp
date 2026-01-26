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
#include <trueform/core/range.hpp>
#include <trueform/intersect/make_isocurves.hpp>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/util/make_numpy_array.hpp>

namespace tf::py {

/// @brief Compute isocontours for a single threshold value
/// @tparam Index Index type (int or int64_t)
/// @tparam RealT Floating point type (float or double)
/// @tparam Dims Dimensionality (2 or 3)
/// @param mesh Mesh wrapper containing the mesh data
/// @param scalars Scalar field values at mesh vertices
/// @param threshold Single threshold value
/// @return Tuple of (paths_offsets, paths_data, points)
template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto make_isocontours_single_impl(
    mesh_wrapper<Index, RealT, Ngon, Dims> &mesh,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1>>
        scalars,
    RealT threshold) -> nanobind::tuple {

  // Create view into mesh data
  auto polygons = mesh.make_primitive_range();

  // Create view into scalar field
  const RealT *scalars_data = static_cast<const RealT *>(scalars.data());
  auto scalars_range = tf::make_range(scalars_data, scalars.shape(0));

  // Call C++ make_isocontours with single threshold
  auto curves = tf::make_isocontours(polygons, scalars_range, threshold);

  // Extract and return as tuple
  auto [paths, c_points] = make_numpy_array(std::move(curves));
  return nanobind::make_tuple(nanobind::make_tuple(paths.first, paths.second),
                              std::move(c_points));
}

/// @brief Compute isocontours for multiple threshold values
/// @tparam Index Index type (int or int64_t)
/// @tparam RealT Floating point type (float or double)
/// @tparam Dims Dimensionality (2 or 3)
/// @param mesh Mesh wrapper containing the mesh data
/// @param scalars Scalar field values at mesh vertices
/// @param thresholds Array of threshold values
/// @return Tuple of (paths_offsets, paths_data, points)
template <typename Index, typename RealT, std::size_t Ngon, std::size_t Dims>
auto make_isocontours_multi_impl(
    mesh_wrapper<Index, RealT, Ngon, Dims> &mesh,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1>>
        scalars,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1>>
        thresholds) -> nanobind::tuple {

  // Create view into mesh data
  auto polygons = mesh.make_primitive_range();

  // Create view into scalar field
  const RealT *scalars_data = static_cast<const RealT *>(scalars.data());
  auto scalars_range = tf::make_range(scalars_data, scalars.shape(0));

  // Create view into thresholds array
  const RealT *thresholds_data = static_cast<const RealT *>(thresholds.data());
  auto thresholds_range = tf::make_range(thresholds_data, thresholds.shape(0));

  // Call C++ make_isocontours with multiple thresholds
  auto curves = tf::make_isocontours(polygons, scalars_range, thresholds_range);

  // Extract and return as tuple
  auto [paths, c_points] = make_numpy_array(std::move(curves));
  return nanobind::make_tuple(nanobind::make_tuple(paths.first, paths.second),
                              std::move(c_points));
}

auto register_intersect_isocontours(nanobind::module_ &m) -> void;

} // namespace tf::py
