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

#include "obj_reader.hpp"

#include "../core/complete.hpp"
#include "../core/polygons_buffer.hpp"
#include "../core/range.hpp"
#include "../core/static_size.hpp"
#include "obj_file.hpp"

#include <cstddef>
#include <string_view>

namespace tf {

/// @ingroup io
/// @brief Read OBJ positions and mixed-size polygons from a file.
///
/// Converts one-based OBJ indices to zero-based indices. Normals and texture
/// coordinates are ignored.
template <typename Index = int, typename Real = float>
auto read_obj(std::string_view path)
    -> tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> {
  tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> output;
  tf::io::obj_reader reader;
  if (!reader.read(path, output.points_buffer(), output.faces_buffer()))
    return {};
  return output;
}

/// @ingroup io
/// @brief Read OBJ positions and fixed-size polygons from a file.
///
/// Returns an empty buffer when a face does not have exactly `Ngon` vertices.
/// The file must not be modified for the duration of the read.
template <typename Index, std::size_t Ngon, typename Real = float>
auto read_obj(std::string_view path)
    -> tf::polygons_buffer<Index, Real, 3, Ngon> {
  tf::polygons_buffer<Index, Real, 3, Ngon> output;
  tf::io::obj_reader reader;
  if (!reader.read(path, output.points_buffer(), output.faces_buffer()))
    return {};
  return output;
}

/// @ingroup io
/// @brief Read fixed-size OBJ polygons with `Ngon` as the first template
/// argument.
/// @overload
template <std::size_t Ngon, typename Index = int, typename Real = float>
auto read_obj(std::string_view path)
    -> tf::polygons_buffer<Index, Real, 3, Ngon> {
  return read_obj<Index, Ngon, Real>(path);
}

/// @ingroup io
/// @brief Read OBJ positions and mixed-size polygons from memory.
template <typename Index = int, typename Real = float>
auto read_obj(tf::range<const char *, tf::dynamic_size> data)
    -> tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> {
  tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> output;
  tf::io::obj_reader reader;
  if (!reader.read(data, output.points_buffer(), output.faces_buffer()))
    return {};
  return output;
}

/// @ingroup io
/// @brief Read OBJ positions and fixed-size polygons from memory.
template <typename Index, std::size_t Ngon, typename Real = float>
auto read_obj(tf::range<const char *, tf::dynamic_size> data)
    -> tf::polygons_buffer<Index, Real, 3, Ngon> {
  tf::polygons_buffer<Index, Real, 3, Ngon> output;
  tf::io::obj_reader reader;
  if (!reader.read(data, output.points_buffer(), output.faces_buffer()))
    return {};
  return output;
}

/// @overload
template <std::size_t Ngon, typename Index = int, typename Real = float>
auto read_obj(tf::range<const char *, tf::dynamic_size> data)
    -> tf::polygons_buffer<Index, Real, 3, Ngon> {
  return read_obj<Index, Ngon, Real>(data);
}

/// @overload
template <typename Index = int, typename Real = float>
auto read_obj(tf::range<char *, tf::dynamic_size> data)
    -> tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> {
  return read_obj<Index, Real>(
      tf::make_range(static_cast<const char *>(data.begin()), data.size()));
}

/// @overload
template <typename Index, std::size_t Ngon, typename Real = float>
auto read_obj(tf::range<char *, tf::dynamic_size> data)
    -> tf::polygons_buffer<Index, Real, 3, Ngon> {
  return read_obj<Index, Ngon, Real>(
      tf::make_range(static_cast<const char *>(data.begin()), data.size()));
}

/// @overload
template <std::size_t Ngon, typename Index = int, typename Real = float>
auto read_obj(tf::range<char *, tf::dynamic_size> data)
    -> tf::polygons_buffer<Index, Real, 3, Ngon> {
  return read_obj<Index, Ngon, Real>(data);
}

/// @ingroup io
/// @brief Read OBJ positions, normals, textures, groups, and objects.
///
/// Positions, normals, and textures in the result are aligned. A position with
/// distinct normal or texture references becomes a distinct output vertex.
/// The first face fixes whether subsequent references use positions, textures,
/// normals, or both textures and normals.
template <typename Index = int, typename Real = float>
auto read_obj(std::string_view path, tf::complete_t)
    -> tf::obj_file<Index, Real> {
  tf::obj_file<Index, Real> output;
  tf::io::obj_reader reader;
  if (!reader.read(path, output))
    return {};
  return output;
}

/// @ingroup io
/// @brief Read complete OBJ data from memory.
/// @overload
template <typename Index = int, typename Real = float>
auto read_obj(tf::range<const char *, tf::dynamic_size> data, tf::complete_t)
    -> tf::obj_file<Index, Real> {
  tf::obj_file<Index, Real> output;
  tf::io::obj_reader reader;
  if (!reader.read(data, output))
    return {};
  return output;
}

/// @overload
template <typename Index = int, typename Real = float>
auto read_obj(tf::range<char *, tf::dynamic_size> data, tf::complete_t)
    -> tf::obj_file<Index, Real> {
  return read_obj<Index, Real>(
      tf::make_range(static_cast<const char *>(data.begin()), data.size()),
      tf::complete);
}

} // namespace tf
