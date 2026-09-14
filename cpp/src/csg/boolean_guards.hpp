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

#include "trueform/csg/boolean_op.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tf::cpp::boolean_detail {

inline auto require_boolean_operation(tf::boolean_op operation) -> void {
  switch (operation) {
  case tf::boolean_op::merge:
  case tf::boolean_op::intersection:
  case tf::boolean_op::left_difference:
  case tf::boolean_op::right_difference:
    return;
  }
  throw std::invalid_argument("make_boolean: invalid operation");
}

inline auto require_sheets(const std::vector<std::int32_t> &sheets) -> void {
  for (const auto sheet : sheets)
    if (sheet < 0 || sheet > 1)
      throw std::out_of_range("make_boolean: sheet index out of range");
}

} // namespace tf::cpp::boolean_detail
