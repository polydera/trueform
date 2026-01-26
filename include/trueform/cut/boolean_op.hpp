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
#include "./arrangement_class.hpp"
#include <array>

namespace tf {
enum class boolean_op {
  merge,
  intersection,
  left_difference,
  right_difference
};

namespace cut {
inline constexpr auto make_boolean_op_spec(boolean_op op)
    -> std::array<tf::arrangement_class, 2> {
  switch (op) {
  case boolean_op::merge:
    return {tf::arrangement_class::outside | tf::arrangement_class::aligned_boundary,
            tf::arrangement_class::outside};
  case boolean_op::intersection:
    return {tf::arrangement_class::inside | tf::arrangement_class::aligned_boundary,
            tf::arrangement_class::inside};
  case boolean_op::left_difference:
    return {tf::arrangement_class::outside | tf::arrangement_class::opposing_boundary,
            tf::arrangement_class::inside};
  case boolean_op::right_difference:
  default:
    return {tf::arrangement_class::inside,
            tf::arrangement_class::outside | tf::arrangement_class::opposing_boundary};
  }
}
} // namespace cut
} // namespace tf
