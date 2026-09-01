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

inline auto skip_obj_whitespace(const char *cursor, const char *end) -> const
    char * {
  while (cursor < end && (*cursor == ' ' || *cursor == '\t'))
    ++cursor;
  return cursor;
}

} // namespace tf::io::obj
