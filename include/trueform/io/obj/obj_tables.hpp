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

#include "../../core/buffer.hpp"
#include "../../core/none.hpp"
#include "../../core/point.hpp"
#include "../../core/static_size.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

namespace tf::io::obj {

/// @brief The tables an OBJ file states, in file order, before deduplication.
///
/// A face corner names a position and, when the file's faces carry them, a
/// texture and a normal; `corner_attributes` is empty when they do not.
/// `face_offsets` spans the corners of each face and exists only for
/// mixed-size faces; `face_groups` / `face_objects` hold `-1` where the face
/// inherits the directive of an earlier line partition.
template <typename Index, typename RealT, std::size_t Ngon>
struct obj_tables {
  tf::buffer<tf::point<RealT, 3>> positions;
  tf::buffer<tf::point<RealT, 2>> textures;
  tf::buffer<tf::point<RealT, 3>> normals;
  tf::buffer<int> corner_positions;
  tf::buffer<std::array<int, 2>> corner_attributes;
  std::conditional_t<Ngon == tf::dynamic_size, tf::buffer<Index>, tf::none_t>
      face_offsets;
  tf::buffer<Index> face_groups;
  tf::buffer<Index> face_objects;
  std::vector<std::string> group_names;
  std::vector<std::string> object_names;
};

} // namespace tf::io::obj
