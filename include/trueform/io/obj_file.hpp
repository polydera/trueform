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

#include "../core/buffer.hpp"
#include "../core/polygons_buffer.hpp"
#include "../core/static_size.hpp"
#include "../core/unit_vectors_buffer.hpp"
#include "../core/vectors_buffer.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace tf {

/// @ingroup io
/// @brief Full-attribute payload returned by `read_obj(path, tf::complete)`.
///
/// `points`, `normals` and `textures` are aligned `[0, n_pts)`. A position
/// with two distinct normals or texture coords in the source file becomes
/// two distinct vertices.
///
/// `face_groups[i]` / `face_objects[i]` index into `group_names` /
/// `object_names`. They are empty when the file has no `g` / `o`
/// directives.
///
/// `Ngon` is the arity the caller states: mixed-size faces by default, or a
/// fixed count every face of the file must have.
template <typename Index = int, typename Real = float,
          std::size_t Ngon = tf::dynamic_size>
struct obj_file {
  tf::polygons_buffer<Index, Real, 3, Ngon> polygons;
  tf::unit_vectors_buffer<Real, 3> normals;
  tf::vectors_buffer<Real, 2> textures;
  tf::buffer<Index> face_groups;
  std::vector<std::string> group_names;
  tf::buffer<Index> face_objects;
  std::vector<std::string> object_names;
};

} // namespace tf
