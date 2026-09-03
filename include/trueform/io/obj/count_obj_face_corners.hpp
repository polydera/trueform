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

#include "parse_obj_token.hpp"

#include <cstddef>

namespace tf::io::obj {

/// @brief Counts the corner tokens of a face line, leaving `cursor` at its end.
///
/// The count reserves the partition's slice of the corner table. A malformed
/// token can parse as more than one corner, so it is a lower bound on what a
/// line can demand, and the parse pass refuses a face that would leave the
/// slice rather than overrunning it.
inline auto count_obj_face_corners(const char *&cursor, const char *end)
    -> std::size_t {
  std::size_t count = 0;
  while (cursor < end && *cursor != '\n' && *cursor != '\r') {
    if (parse_obj_token(cursor, end).empty())
      break;
    ++count;
  }
  return count;
}

} // namespace tf::io::obj
