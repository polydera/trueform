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
#include "skip_obj_whitespace.hpp"

namespace tf::io::obj {

/// @brief Which attributes a face line's first corner names.
///
/// The first face of a file fixes the references of every later one, so this
/// is the file's own statement of what its corners carry. A token the parse
/// then refuses is classified here too, and the read fails on it either way.
inline auto obj_face_token_mode(const char *cursor, const char *end)
    -> obj_face_mode {
  const auto ends_index = [](char c) {
    return c == '/' || c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '#';
  };
  cursor = skip_obj_whitespace(cursor, end);
  while (cursor < end && !ends_index(*cursor))
    ++cursor;
  if (cursor >= end || *cursor != '/')
    return obj_face_mode::v;
  ++cursor;
  if (cursor < end && *cursor == '/')
    return obj_face_mode::v_vn;
  while (cursor < end && !ends_index(*cursor))
    ++cursor;
  return cursor < end && *cursor == '/' ? obj_face_mode::v_vt_vn
                                        : obj_face_mode::v_vt;
}

} // namespace tf::io::obj
