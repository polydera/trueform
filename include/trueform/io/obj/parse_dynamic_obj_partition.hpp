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

#include "count_dynamic_obj_records.hpp"
#include "parse_obj_face_index.hpp"
#include "parse_obj_scalars.hpp"
#include "skip_obj_line.hpp"
#include "skip_obj_whitespace.hpp"

#include "../../core/offset_block_buffer.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/static_size.hpp"

#include <cstddef>

namespace tf::io::obj {

/// @brief Parses one line partition into the slices its bases reserved.
///
/// On success every table position between `base` and `limit` has been written
/// exactly once, so the partitions stay independent; a refusal leaves the
/// slice partly unwritten and the caller reads nothing. A face of fewer than
/// three corners is not emitted, so it claims neither an offset nor a corner.
///
/// @param point_base Where this read's points begin in `points`.
/// @param face_base Where this read's faces begin in the offsets.
/// @param corner_base Where this read's corners begin in the face data.
template <typename Index, typename RealT, std::size_t Dims>
auto parse_dynamic_obj_partition(const char *cursor, const char *end,
                                 const obj_dynamic_record_counts &base,
                                 const obj_dynamic_record_counts &limit,
                                 std::size_t point_base, std::size_t face_base,
                                 std::size_t corner_base,
                                 tf::points_buffer<RealT, Dims> &points,
                                 tf::offset_block_buffer<Index, Index> &faces)
    -> bool {
  static_assert(Dims == 2 || Dims == 3);

  auto &offsets = faces.offsets_buffer();
  auto &data = faces.data_buffer();
  auto written_points = base.points;
  auto written_faces = base.faces;
  auto written_corners = base.corners;

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
      auto point = points[point_base + written_points++];
      point[0] = x;
      point[1] = y;
      if constexpr (Dims == 3)
        point[2] = z;
    } else if (cursor[0] == 'f' && cursor + 1 < end &&
               (cursor[1] == ' ' || cursor[1] == '\t')) {
      cursor += 2;
      const auto first_corner = written_corners;
      std::size_t corner_count = 0;
      while (cursor < end && *cursor != '\n' && *cursor != '\r') {
        cursor = skip_obj_whitespace(cursor, end);
        if (cursor >= end || *cursor == '\n' || *cursor == '\r' ||
            *cursor == '#')
          break;
        int vertex_index = 0;
        if (!parse_obj_face_index(cursor, end, vertex_index) ||
            vertex_index <= 0)
          return false;
        // A face still short of three corners claims none of the slice, so
        // its corners are written only where the kept ones left room.
        if (written_corners < limit.corners)
          data[corner_base + written_corners++] =
              static_cast<Index>(vertex_index - 1);
        ++corner_count;
      }
      if (corner_count < 3) {
        written_corners = first_corner;
      } else {
        if (written_corners - first_corner != corner_count ||
            written_faces == limit.faces)
          return false;
        offsets[face_base + ++written_faces] =
            static_cast<Index>(corner_base + written_corners);
      }
    }
    cursor = skip_obj_line(cursor, end);
  }
  return written_points == limit.points && written_faces == limit.faces &&
         written_corners == limit.corners;
}

} // namespace tf::io::obj
