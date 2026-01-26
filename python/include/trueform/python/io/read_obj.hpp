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
#include <string>
#include <trueform/io/read_obj.hpp>
#include <trueform/python/util/make_numpy_array.hpp>

namespace tf::py {

/// @brief Template implementation for read_obj
/// @tparam Index The index type (int or int64_t)
/// @tparam Ngon The number of vertices per face (3 or 4)
/// @param filename Path to OBJ file
/// @return Tuple of (faces, points) as numpy arrays
template <typename Index, std::size_t Ngon>
auto read_obj_impl(const std::string &filename) {
  // Read OBJ file using C++ function
  auto polys = tf::read_obj<Index, Ngon>(filename);

  // Use the new make_numpy_array overloads for clean extraction
  auto faces = make_numpy_array(std::move(polys.faces_buffer()));
  auto points = make_numpy_array(std::move(polys.points_buffer()));

  // Return as tuple
  return nanobind::make_tuple(faces, points);
}

auto register_io_read_obj(nanobind::module_ &m) -> void;

} // namespace tf::py
