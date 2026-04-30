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
template <typename Index = int> struct obj_file {
  tf::polygons_buffer<Index, float, 3, tf::dynamic_size> polygons;
  tf::unit_vectors_buffer<float, 3> normals;
  tf::vectors_buffer<float, 2> textures;
  tf::buffer<Index> face_groups;
  std::vector<std::string> group_names;
  tf::buffer<Index> face_objects;
  std::vector<std::string> object_names;
};

} // namespace tf
