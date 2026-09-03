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

namespace tf::io::obj {

/// @brief What a corner of one position adds to it: the attribute pair it
/// names and the corner that named it.
///
/// The order is the vertex order inside a position, so the records of a
/// position sort against each other by value.
template <typename Index> struct obj_corner_attributes {
  int texture;
  int normal;
  Index corner;

  auto operator<(const obj_corner_attributes &other) const -> bool {
    if (texture != other.texture)
      return texture < other.texture;
    if (normal != other.normal)
      return normal < other.normal;
    return corner < other.corner;
  }
};

/// @brief Whether two corners of one position resolve to the same vertex.
template <typename Index>
auto names_one_obj_vertex(const obj_corner_attributes<Index> &left,
                          const obj_corner_attributes<Index> &right) -> bool {
  return left.texture == right.texture && left.normal == right.normal;
}

} // namespace tf::io::obj
