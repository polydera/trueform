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
/// @tparam RealT The point coordinate type (float or double)
/// @tparam Ngon The number of vertices per face (3 or 4)
/// @param filename Path to OBJ file
/// @return Tuple of (faces, points) as numpy arrays
template <typename Index, typename RealT, std::size_t Ngon>
auto read_obj_impl(const std::string &filename) {
  auto polys = tf::read_obj<Index, Ngon, RealT>(filename);

  auto faces = make_numpy_array(std::move(polys.faces_buffer()));
  auto points = make_numpy_array(std::move(polys.points_buffer()));

  return nanobind::make_tuple(faces, points);
}

/// @brief Template implementation for read_obj with dynamic polygon sizes
/// @tparam Index The index type (int or int64_t)
/// @tparam RealT The point coordinate type (float or double)
/// @param filename Path to OBJ file
/// @return Tuple of (offsets, data, points) as numpy arrays
template <typename Index, typename RealT>
auto read_obj_dynamic_impl(const std::string &filename) {
  auto polys = tf::read_obj<Index, RealT>(filename);

  auto [offsets, data] = make_numpy_array(std::move(polys.faces_buffer()));
  auto points = make_numpy_array(std::move(polys.points_buffer()));

  return nanobind::make_tuple(offsets, data, points);
}

auto register_io_read_obj(nanobind::module_ &m) -> void;

} // namespace tf::py
