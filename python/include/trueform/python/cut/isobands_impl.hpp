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
#include <trueform/cut/make_isobands.hpp>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/core/offset_blocked_array.hpp>
#include <trueform/python/intersect/isocontours.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <utility>

namespace tf::py {

/// @brief Implementation for make_isobands without return_curves
/// @tparam Index Index type
/// @tparam RealT Floating point type
/// @tparam NGon Polygon order (3 for triangles)
/// @tparam Dims Dimensionality (2 or 3)
template <typename Index, typename RealT, std::size_t NGon, std::size_t Dims>
auto make_isobands_impl(
    mesh_wrapper<Index, RealT, NGon, Dims> &mesh_wrap,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1>>
        scalars,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1>>
        cut_values,
    nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>>
        selected_bands) {

  auto polygons = mesh_wrap.make_primitive_range();

  // Create ranges
  auto scalars_range = tf::make_range(scalars.data(), scalars.shape(0));
  auto cut_values_range =
      tf::make_range(cut_values.data(), cut_values.shape(0));
  auto selected_bands_range =
      tf::make_range(selected_bands.data(), selected_bands.shape(0));

  // Call C++ make_isobands (returns mesh, labels)
  auto [result_mesh, labels] = tf::make_isobands<Index>(
      polygons, scalars_range, cut_values_range, selected_bands_range);

  // Extract mesh as (faces, points) - move ownership
  return nanobind::make_tuple(make_numpy_array(std::move(result_mesh)),
                              make_numpy_array(std::move(labels)));
}

/// @brief Implementation for make_isobands with return_curves
/// @tparam Index Index type
/// @tparam RealT Floating point type
/// @tparam NGon Polygon order (3 for triangles)
/// @tparam Dims Dimensionality (2 or 3)
template <typename Index, typename RealT, std::size_t NGon, std::size_t Dims>
auto make_isobands_with_curves_impl(
    mesh_wrapper<Index, RealT, NGon, Dims> &mesh_wrap,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1>>
        scalars,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1>>
        cut_values,
    nanobind::ndarray<nanobind::numpy, const int, nanobind::shape<-1>>
        selected_bands) {

  auto polygons = mesh_wrap.make_primitive_range();

  // Create ranges
  auto scalars_range = tf::make_range(scalars.data(), scalars.shape(0));
  auto cut_values_range =
      tf::make_range(cut_values.data(), cut_values.shape(0));
  auto selected_bands_range =
      tf::make_range(selected_bands.data(), selected_bands.shape(0));

  // Call C++ make_isobands with return_curves (returns mesh, labels, curves)
  auto [result_mesh, labels, curves] =
      tf::make_isobands<Index>(polygons, scalars_range, cut_values_range,
                               selected_bands_range, tf::return_curves);

  // Extract mesh as (faces, points) - move ownership
  auto mesh_pair = make_numpy_array(std::move(result_mesh));

  // Extract labels buffer - move ownership
  auto labels_array = make_numpy_array(std::move(labels));

  // Extract curves as ((paths_offsets, paths_data), curve_points) - move
  // ownership
  auto [paths, c_points] = make_numpy_array(std::move(curves));
  auto curve_pair = nanobind::make_tuple(
      nanobind::make_tuple(paths.first, paths.second), std::move(c_points));
  return nanobind::make_tuple(std::move(mesh_pair), std::move(labels_array),
                              std::move(curve_pair));
}

auto register_intersect_isobands(nanobind::module_ &m) -> void;

} // namespace tf::py
