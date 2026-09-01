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

#include "parse_obj_face_index.hpp"
#include "parse_obj_scalars.hpp"
#include "skip_obj_line.hpp"
#include "skip_obj_whitespace.hpp"

#include "../../core/blocked_buffer.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/static_size.hpp"

#include <array>
#include <cstddef>

namespace tf::io::obj {

template <typename Index, typename RealT, std::size_t Dims, std::size_t Ngon>
auto parse_fixed_obj_partition(const char *cursor, const char *end,
                               tf::points_buffer<RealT, Dims> &points,
                               std::size_t point_offset,
                               std::size_t expected_points,
                               tf::blocked_buffer<Index, Ngon> &faces,
                               std::size_t face_offset,
                               std::size_t expected_faces) -> bool {
  static_assert(Dims == 2 || Dims == 3);
  static_assert(Ngon != tf::dynamic_size);

  std::size_t point_count = 0;
  std::size_t face_count = 0;
  while (cursor < end) {
    cursor = skip_obj_whitespace(cursor, end);
    if (cursor >= end)
      break;

    if (cursor[0] == 'v' && cursor + 1 < end &&
        (cursor[1] == ' ' || cursor[1] == '\t')) {
      cursor += 2;
      RealT x{};
      RealT y{};
      RealT z{};
      if (!parse_obj_scalars(cursor, end, x, y, z))
        return false;
      auto point = points[point_offset + point_count++];
      point[0] = x;
      point[1] = y;
      if constexpr (Dims == 3)
        point[2] = z;
    } else if (cursor[0] == 'f' && cursor + 1 < end &&
               (cursor[1] == ' ' || cursor[1] == '\t')) {
      cursor += 2;
      std::array<Index, Ngon> face{};
      std::size_t count = 0;
      while (cursor < end && *cursor != '\n' && *cursor != '\r') {
        cursor = skip_obj_whitespace(cursor, end);
        if (cursor >= end || *cursor == '\n' || *cursor == '\r' ||
            *cursor == '#')
          break;
        int vertex_index = 0;
        if (!parse_obj_face_index(cursor, end, vertex_index) ||
            vertex_index <= 0)
          return false;
        if (count < Ngon)
          face[count] = static_cast<Index>(vertex_index - 1);
        ++count;
      }
      if (count != Ngon)
        return false;
      faces[face_offset + face_count++] = face;
    }
    cursor = skip_obj_line(cursor, end);
  }
  return point_count == expected_points && face_count == expected_faces;
}

} // namespace tf::io::obj
