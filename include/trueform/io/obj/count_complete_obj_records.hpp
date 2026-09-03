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
#include "obj_face_mode.hpp"
#include "obj_face_token_mode.hpp"
#include "skip_obj_line.hpp"
#include "skip_obj_whitespace.hpp"

#include "../../core/static_size.hpp"

#include <cstddef>

namespace tf::io::obj {

/// @brief What one line partition contributes to every table of an OBJ file.
///
/// Prefixed across the partitions it becomes each partition's base in those
/// tables, so `bases[p + 1] - bases[p]` is what partition `p` writes.
struct obj_complete_record_counts {
  std::size_t positions = 0;
  std::size_t textures = 0;
  std::size_t normals = 0;
  std::size_t faces = 0;
  std::size_t corners = 0;
  std::size_t groups = 0;
  std::size_t objects = 0;
};

/// @brief Counts what a line partition contributes and reads its first face.
///
/// `mode` is left `unknown` when the partition holds no face, so the file's
/// mode is the first partition that states one. A stated arity is the corner
/// count of every face, so only the first face's leading token is read,
/// for the mode.
template <std::size_t Ngon>
auto count_complete_obj_records(const char *cursor, const char *end,
                                obj_face_mode &mode)
    -> obj_complete_record_counts {
  obj_complete_record_counts counts;
  while (cursor < end) {
    cursor = skip_obj_whitespace(cursor, end);
    if (cursor >= end)
      break;
    const auto first = cursor[0];
    if (first == 'v' && cursor + 1 < end &&
        (cursor[1] == ' ' || cursor[1] == '\t')) {
      ++counts.positions;
    } else if (first == 'v' && cursor + 2 < end && cursor[1] == 't' &&
               (cursor[2] == ' ' || cursor[2] == '\t')) {
      ++counts.textures;
    } else if (first == 'v' && cursor + 2 < end && cursor[1] == 'n' &&
               (cursor[2] == ' ' || cursor[2] == '\t')) {
      ++counts.normals;
    } else if (first == 'g' && cursor + 1 < end &&
               (cursor[1] == ' ' || cursor[1] == '\t')) {
      ++counts.groups;
    } else if (first == 'o' && cursor + 1 < end &&
               (cursor[1] == ' ' || cursor[1] == '\t')) {
      ++counts.objects;
    } else if (first == 'f' && cursor + 1 < end &&
               (cursor[1] == ' ' || cursor[1] == '\t')) {
      cursor += 2;
      if (mode == obj_face_mode::unknown)
        mode = obj_face_token_mode(cursor, end);
      ++counts.faces;
      if constexpr (Ngon == tf::dynamic_size)
        counts.corners += count_obj_face_corners(cursor, end);
      else
        counts.corners += Ngon;
    }
    cursor = skip_obj_line(cursor, end);
  }
  return counts;
}

} // namespace tf::io::obj
