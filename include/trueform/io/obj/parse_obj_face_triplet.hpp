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

#include "obj_face_mode.hpp"

#include <array>
#include <charconv>
#include <system_error>

namespace tf::io::obj {

/// @brief Parses one face corner, refusing any that does not reference what
/// the file's first face stated.
inline auto parse_obj_face_triplet(const char *&cursor, const char *end,
                                   obj_face_mode mode,
                                   std::array<int, 3> &output) -> bool {
  int vertex = 0;
  int texture = 0;
  int normal = 0;
  bool has_texture = false;
  bool has_normal = false;

  auto result = std::from_chars(cursor, end, vertex);
  if (result.ec != std::errc{} || vertex == 0)
    return false;
  cursor = result.ptr;

  if (cursor < end && *cursor == '/') {
    ++cursor;
    if (cursor < end && *cursor == '/') {
      ++cursor;
      result = std::from_chars(cursor, end, normal);
      if (result.ec != std::errc{} || normal == 0)
        return false;
      cursor = result.ptr;
      has_normal = true;
    } else {
      result = std::from_chars(cursor, end, texture);
      if (result.ec != std::errc{} || texture == 0)
        return false;
      cursor = result.ptr;
      has_texture = true;
      if (cursor < end && *cursor == '/') {
        ++cursor;
        result = std::from_chars(cursor, end, normal);
        if (result.ec != std::errc{} || normal == 0)
          return false;
        cursor = result.ptr;
        has_normal = true;
      }
    }
  }

  const auto observed = !has_texture && !has_normal  ? obj_face_mode::v
                        : has_texture && !has_normal ? obj_face_mode::v_vt
                        : !has_texture && has_normal ? obj_face_mode::v_vn
                                                     : obj_face_mode::v_vt_vn;
  if (mode != observed)
    return false;

  output[0] = vertex - 1;
  output[1] = has_texture ? texture - 1 : -1;
  output[2] = has_normal ? normal - 1 : -1;
  return true;
}

} // namespace tf::io::obj
