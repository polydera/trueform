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

#include <charconv>
#include <system_error>

namespace tf::io::obj {

inline auto parse_obj_face_index(const char *&cursor, const char *end,
                                 int &vertex_index) -> bool {
  auto result = std::from_chars(cursor, end, vertex_index);
  if (result.ec != std::errc{})
    return false;
  cursor = result.ptr;
  if (cursor >= end || *cursor != '/')
    return true;

  ++cursor;
  if (cursor < end && *cursor == '/') {
    ++cursor;
  } else if (cursor < end && *cursor >= '0' && *cursor <= '9') {
    int ignored = 0;
    result = std::from_chars(cursor, end, ignored);
    if (result.ec == std::errc{})
      cursor = result.ptr;
    if (cursor < end && *cursor == '/')
      ++cursor;
  }
  if (cursor < end && *cursor >= '0' && *cursor <= '9') {
    int ignored = 0;
    result = std::from_chars(cursor, end, ignored);
    if (result.ec == std::errc{})
      cursor = result.ptr;
  }
  return true;
}

} // namespace tf::io::obj
