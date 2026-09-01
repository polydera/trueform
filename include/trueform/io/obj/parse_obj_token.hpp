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

#include "skip_obj_whitespace.hpp"

#include <cstddef>
#include <string_view>

namespace tf::io::obj {

inline auto parse_obj_token(const char *&cursor, const char *end)
    -> std::string_view {
  cursor = skip_obj_whitespace(cursor, end);
  const auto *start = cursor;
  while (cursor < end && *cursor != ' ' && *cursor != '\t' && *cursor != '\n' &&
         *cursor != '\r' && *cursor != '#')
    ++cursor;
  return {start, static_cast<std::size_t>(cursor - start)};
}

} // namespace tf::io::obj
