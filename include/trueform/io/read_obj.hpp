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
#include "../core/polygons_buffer.hpp"
#include "./obj_reader.hpp"

namespace tf {

/// @ingroup io
/// @brief Read OBJ file with dynamic polygon sizes.
///
/// Reads ASCII OBJ format with mixed polygon sizes.
/// Converts 1-based OBJ indices to 0-based.
/// Only reads vertex positions (ignores normals and texture coordinates).
///
/// @tparam Index The index type (defaults to int).
/// @param file_path Path to the OBJ file.
/// @return A @ref tf::polygons_buffer with dynamic face size, or empty on error.
template <typename Index = int>
auto read_obj(std::string_view file_path)
    -> tf::polygons_buffer<Index, float, 3, tf::dynamic_size> {
  tf::polygons_buffer<Index, float, 3, tf::dynamic_size> out;
  tf::io::obj_reader reader;
  if (!reader.read(file_path, out.points_buffer(), out.faces_buffer())) {
    return {}; // Return empty on error
  }
  return out;
}

/// @ingroup io
/// @brief Read OBJ file with fixed polygon size.
///
/// Reads ASCII OBJ format expecting uniform polygon size.
/// Converts 1-based OBJ indices to 0-based.
///
/// @tparam Index The index type (defaults to int).
/// @tparam Ngon The expected polygon size (e.g., 3 for triangles, 4 for quads).
/// @param file_path Path to the OBJ file.
/// @return A @ref tf::polygons_buffer with fixed face size, or empty on error.
template <typename Index = int, std::size_t Ngon>
auto read_obj(std::string_view file_path)
    -> tf::polygons_buffer<Index, float, 3, Ngon> {
  tf::polygons_buffer<Index, float, 3, Ngon> out;
  tf::io::obj_reader reader;
  if (!reader.read(file_path, out.points_buffer(), out.faces_buffer())) {
    return {}; // Return empty on error
  }
  return out;
}

/// @ingroup io
/// @brief Read OBJ file with fixed polygon size (int index).
/// @overload
template <std::size_t Ngon> auto read_obj(std::string_view file_path) {
  return read_obj<int, Ngon>(file_path);
}
} // namespace tf
