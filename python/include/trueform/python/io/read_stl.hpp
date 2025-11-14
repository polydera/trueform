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
#include <trueform/python/util/make_numpy_array.hpp>

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

  // Create numpy arrays with proper empty array handling
  // Shape: (num_faces, 3) for faces, (num_points, 3) for points
  auto faces = make_numpy_array<nanobind::shape<-1, 3>>(faces_ptr, {static_cast<size_t>(num_faces), 3});
  auto points = make_numpy_array<nanobind::shape<-1, 3>>(points_ptr, {num_points, 3});

  // Return as tuple
  return nanobind::make_tuple(faces, points);
}

auto register_io_read_stl(nanobind::module_ &m) -> void;

} // namespace tf::py
