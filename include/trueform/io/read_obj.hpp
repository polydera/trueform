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
#include "../core/complete.hpp"
#include "../core/polygons_buffer.hpp"
#include "./obj_file.hpp"
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
/// @tparam Real The coordinate scalar type (defaults to float).
/// @param file_path Path to the OBJ file.
/// @return A @ref tf::polygons_buffer with dynamic face size, or empty on error.
template <typename Index = int, typename Real = float>
auto read_obj(std::string_view file_path)
    -> tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> {
  tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> out;
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
/// @tparam Real The coordinate scalar type (defaults to float).
/// @param file_path Path to the OBJ file.
/// @return A @ref tf::polygons_buffer with fixed face size, or empty on error.
template <typename Index, std::size_t Ngon, typename Real = float>
auto read_obj(std::string_view file_path)
    -> tf::polygons_buffer<Index, Real, 3, Ngon> {
  tf::polygons_buffer<Index, Real, 3, Ngon> out;
  tf::io::obj_reader reader;
  if (!reader.read(file_path, out.points_buffer(), out.faces_buffer())) {
    return {}; // Return empty on error
  }
  return out;
}

/// @ingroup io
/// @brief Read OBJ file with fixed polygon size (Ngon-first convenience).
/// @overload
template <std::size_t Ngon, typename Index = int, typename Real = float>
auto read_obj(std::string_view file_path)
    -> tf::polygons_buffer<Index, Real, 3, Ngon> {
  return read_obj<Index, Ngon, Real>(file_path);
}

/// @ingroup io
/// @brief Read OBJ from a memory buffer with dynamic polygon sizes.
template <typename Index = int, typename Real = float>
auto read_obj(tf::range<const char *, tf::dynamic_size> data)
    -> tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> {
  tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> out;
  tf::io::obj_reader reader;
  if (!reader.read(data, out.points_buffer(), out.faces_buffer())) {
    return {};
  }
  return out;
}

/// @ingroup io
/// @brief Read OBJ from a memory buffer with fixed polygon size.
template <typename Index, std::size_t Ngon, typename Real = float>
auto read_obj(tf::range<const char *, tf::dynamic_size> data)
    -> tf::polygons_buffer<Index, Real, 3, Ngon> {
  tf::polygons_buffer<Index, Real, 3, Ngon> out;
  tf::io::obj_reader reader;
  if (!reader.read(data, out.points_buffer(), out.faces_buffer())) {
    return {};
  }
  return out;
}

/// @overload
template <std::size_t Ngon, typename Index = int, typename Real = float>
auto read_obj(tf::range<const char *, tf::dynamic_size> data)
    -> tf::polygons_buffer<Index, Real, 3, Ngon> {
  return read_obj<Index, Ngon, Real>(data);
}

/// @brief Non-const char* overloads — delegate to const char*.
template <typename Index = int, typename Real = float>
auto read_obj(tf::range<char *, tf::dynamic_size> data)
    -> tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> {
  return read_obj<Index, Real>(
      tf::make_range(static_cast<const char *>(data.begin()), data.size()));
}

template <typename Index, std::size_t Ngon, typename Real = float>
auto read_obj(tf::range<char *, tf::dynamic_size> data)
    -> tf::polygons_buffer<Index, Real, 3, Ngon> {
  return read_obj<Index, Ngon, Real>(
      tf::make_range(static_cast<const char *>(data.begin()), data.size()));
}

template <std::size_t Ngon, typename Index = int, typename Real = float>
auto read_obj(tf::range<char *, tf::dynamic_size> data)
    -> tf::polygons_buffer<Index, Real, 3, Ngon> {
  return read_obj<Index, Ngon, Real>(data);
}

/// @ingroup io
/// @brief Read OBJ file with all attributes (positions, normals, textures,
/// groups, objects).
///
/// `points`, `normals` and `textures` are aligned `[0, n_pts)`. A position
/// with two distinct normals or texture coords in the source becomes two
/// distinct vertices.
///
/// All-or-nothing per attribute: the first `f` line locks the format mode
/// (`v`, `v/vt`, `v//vn`, `v/vt/vn`); inconsistent face refs return an
/// empty `obj_file`.
///
/// @tparam Index The index type (defaults to int).
/// @tparam Real The coordinate scalar type (defaults to float).
/// @param file_path Path to the OBJ file.
/// @return @ref tf::obj_file, or empty on error.
template <typename Index = int, typename Real = float>
auto read_obj(std::string_view file_path, tf::complete_t)
    -> tf::obj_file<Index, Real> {
  tf::obj_file<Index, Real> out;
  tf::io::obj_reader reader;
  if (!reader.read(file_path, out))
    return {};
  return out;
}

/// @ingroup io
/// @brief Read complete OBJ from a memory buffer.
/// @overload
template <typename Index = int, typename Real = float>
auto read_obj(tf::range<const char *, tf::dynamic_size> data, tf::complete_t)
    -> tf::obj_file<Index, Real> {
  tf::obj_file<Index, Real> out;
  tf::io::obj_reader reader;
  if (!reader.read(data, out))
    return {};
  return out;
}

template <typename Index = int, typename Real = float>
auto read_obj(tf::range<char *, tf::dynamic_size> data, tf::complete_t)
    -> tf::obj_file<Index, Real> {
  return read_obj<Index, Real>(
      tf::make_range(static_cast<const char *>(data.begin()), data.size()),
      tf::complete);
}
} // namespace tf
