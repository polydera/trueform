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

#include "count_obj_face_corners.hpp"
#include "skip_obj_line.hpp"
#include "skip_obj_whitespace.hpp"

#include <cstddef>

namespace tf::io::obj {

/// @brief What one line partition contributes to a positions-only OBJ read.
///
/// A face of fewer than three corners is not a face, so neither it nor its
/// corners are counted.
struct obj_dynamic_record_counts {
  std::size_t points = 0;
  std::size_t faces = 0;
  std::size_t corners = 0;
};

inline auto count_dynamic_obj_records(const char *cursor, const char *end)
    -> obj_dynamic_record_counts {
  obj_dynamic_record_counts counts;
  while (cursor < end) {
    cursor = skip_obj_whitespace(cursor, end);
    if (cursor >= end)
      break;
    if (cursor[0] == 'v' && cursor + 1 < end &&
        (cursor[1] == ' ' || cursor[1] == '\t')) {
      ++counts.points;
    } else if (cursor[0] == 'f' && cursor + 1 < end &&
               (cursor[1] == ' ' || cursor[1] == '\t')) {
      cursor += 2;
      const auto corners = count_obj_face_corners(cursor, end);
      if (corners >= 3) {
        ++counts.faces;
        counts.corners += corners;
      }
    }
    cursor = skip_obj_line(cursor, end);
  }
  return counts;
}

} // namespace tf::io::obj
