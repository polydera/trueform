/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <string>
#include <trueform/io/read_stl.hpp>

namespace tf::py {

/// @brief Template implementation for read_stl
/// @tparam Index The index type (int or int64_t)
/// @param filename Path to STL file
/// @return Tuple of (faces, points) as numpy arrays
template <typename Index>
auto read_stl_impl(const std::string &filename) {
  // Read STL file using C++ function
  auto polys = tf::read_stl<Index>(filename);

  // Get sizes
  auto num_faces = polys.faces_buffer().size();
  auto num_points = polys.points_buffer().size();

  // Release ownership from C++ buffers
  Index *faces_ptr = polys.faces_buffer().data_buffer().release();
  float *points_ptr = polys.points_buffer().data_buffer().release();

  // Create capsules with deleters for memory management
  auto faces_capsule = nanobind::capsule(
      faces_ptr, [](void *p) noexcept { delete[] static_cast<Index *>(p); });

  auto points_capsule = nanobind::capsule(
      points_ptr, [](void *p) noexcept { delete[] static_cast<float *>(p); });

  // Create numpy arrays from raw pointers
  // Shape: (num_faces, 3) for faces, (num_points, 3) for points
  auto faces = nanobind::ndarray<nanobind::numpy, Index>(
      faces_ptr, {static_cast<size_t>(num_faces), 3}, faces_capsule);

  auto points = nanobind::ndarray<nanobind::numpy, float>(
      points_ptr, {num_points, 3}, points_capsule);

  // Return as tuple
  return nanobind::make_tuple(faces, points);
}

auto register_io_read_stl(nanobind::module_ &m) -> void;

} // namespace tf::py
