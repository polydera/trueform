/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/core/range.hpp>
#include <trueform/intersect/make_isocurves.hpp>
#include <trueform/python/core/mesh.hpp>
#include <trueform/python/util/make_numpy_array.hpp>

namespace tf::py {

namespace {

/// @brief Helper to extract (paths_offsets, paths_data, points) from curves_buffer
/// @tparam Index Index type (int or int64_t)
/// @tparam RealT Floating point type (float or double)
/// @tparam Dims Dimensionality (2 or 3)
/// @param curves The curves_buffer to extract from
/// @return Tuple of (paths_offsets, paths_data, points) as numpy arrays
template <typename Index, typename RealT, std::size_t Dims>
auto extract_curves_as_tuple(tf::curves_buffer<Index, RealT, Dims> &curves) {
  // === Extract paths (OffsetBlockedArray pattern from face_membership) ===
  auto &paths_buf = curves.paths_buffer();

  // Get pointers and sizes BEFORE releasing
  Index *paths_offsets_ptr = paths_buf.offsets_buffer().begin();
  std::size_t paths_offsets_size = paths_buf.offsets_buffer().size();
  Index *paths_data_ptr = paths_buf.data_buffer().begin();
  std::size_t paths_data_size = paths_buf.data_buffer().size();

  // Release ownership
  paths_buf.offsets_buffer().release();
  paths_buf.data_buffer().release();

  // Create numpy arrays
  auto paths_offsets = make_numpy_array<nanobind::shape<-1>>(
      paths_offsets_ptr, {paths_offsets_size});
  auto paths_data =
      make_numpy_array<nanobind::shape<-1>>(paths_data_ptr, {paths_data_size});

  // === Extract points (read_stl pattern) ===
  auto &points_buf = curves.points_buffer();

  // Get size and pointer BEFORE releasing
  std::size_t num_points = points_buf.size();
  RealT *points_ptr = points_buf.data_buffer().release();

  // Create numpy array with shape (num_points, Dims)
  auto points = make_numpy_array<nanobind::shape<-1, Dims>>(
      points_ptr, {num_points, Dims});

  // Return tuple
  return nanobind::make_tuple(paths_offsets, paths_data, points);
}

} // anonymous namespace

/// @brief Compute isocontours for a single threshold value
/// @tparam Index Index type (int or int64_t)
/// @tparam RealT Floating point type (float or double)
/// @tparam Dims Dimensionality (2 or 3)
/// @param mesh Mesh wrapper containing the mesh data
/// @param scalars Scalar field values at mesh vertices
/// @param threshold Single threshold value
/// @return Tuple of (paths_offsets, paths_data, points)
template <typename Index, typename RealT, std::size_t Dims>
auto make_isocontours_single_impl(
    mesh_wrapper<Index, RealT, 3, Dims> &mesh,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1>>
        scalars,
    RealT threshold) {

  // Create view into mesh data
  auto polygons = mesh.make_primitive_range();

  // Create view into scalar field
  const RealT *scalars_data = static_cast<const RealT *>(scalars.data());
  auto scalars_range = tf::make_range(scalars_data, scalars.shape(0));

  // Call C++ make_isocontours with single threshold
  auto curves = tf::make_isocontours(polygons, scalars_range, threshold);

  // Extract and return as tuple
  return extract_curves_as_tuple<Index, RealT, Dims>(curves);
}

/// @brief Compute isocontours for multiple threshold values
/// @tparam Index Index type (int or int64_t)
/// @tparam RealT Floating point type (float or double)
/// @tparam Dims Dimensionality (2 or 3)
/// @param mesh Mesh wrapper containing the mesh data
/// @param scalars Scalar field values at mesh vertices
/// @param thresholds Array of threshold values
/// @return Tuple of (paths_offsets, paths_data, points)
template <typename Index, typename RealT, std::size_t Dims>
auto make_isocontours_multi_impl(
    mesh_wrapper<Index, RealT, 3, Dims> &mesh,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1>>
        scalars,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1>>
        thresholds) {

  // Create view into mesh data
  auto polygons = mesh.make_primitive_range();

  // Create view into scalar field
  const RealT *scalars_data = static_cast<const RealT *>(scalars.data());
  auto scalars_range = tf::make_range(scalars_data, scalars.shape(0));

  // Create view into thresholds array
  const RealT *thresholds_data = static_cast<const RealT *>(thresholds.data());
  auto thresholds_range =
      tf::make_range(thresholds_data, thresholds.shape(0));

  // Call C++ make_isocontours with multiple thresholds
  auto curves = tf::make_isocontours(polygons, scalars_range, thresholds_range);

  // Extract and return as tuple
  return extract_curves_as_tuple<Index, RealT, Dims>(curves);
}

auto register_intersect_isocontours(nanobind::module_ &m) -> void;

} // namespace tf::py
