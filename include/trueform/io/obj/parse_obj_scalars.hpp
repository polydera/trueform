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

#include "../external/fast_float.hpp"

#include <system_error>

namespace tf::io::obj {

template <typename Scalar>
auto parse_obj_scalars(const char *&cursor, const char *end, Scalar &x,
                       Scalar &y) -> bool {
  cursor = skip_obj_whitespace(cursor, end);
  auto result = tf::external::fast_float::from_chars(cursor, end, x);
  if (result.ec != std::errc{})
    return false;
  cursor = skip_obj_whitespace(result.ptr, end);
  result = tf::external::fast_float::from_chars(cursor, end, y);
  if (result.ec != std::errc{})
    return false;
  cursor = result.ptr;
  return true;
}

template <typename Scalar>
auto parse_obj_scalars(const char *&cursor, const char *end, Scalar &x,
                       Scalar &y, Scalar &z) -> bool {
  if (!parse_obj_scalars(cursor, end, x, y))
    return false;
  cursor = skip_obj_whitespace(cursor, end);
  auto result = tf::external::fast_float::from_chars(cursor, end, z);
  if (result.ec != std::errc{})
    return false;
  cursor = result.ptr;
  return true;
}

} // namespace tf::io::obj
