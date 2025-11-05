/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/containment.hpp"
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
    -> std::array<tf::strict_containment, 2> {
  switch (op) {
  case boolean_op::merge:
    return {tf::strict_containment::outside, tf::strict_containment::outside};
  case boolean_op::intersection:
    return {tf::strict_containment::inside, tf::strict_containment::inside};
  case boolean_op::left_difference:
    return {tf::strict_containment::outside, tf::strict_containment::inside};
  case boolean_op::right_difference:
  default:
    return {tf::strict_containment::inside, tf::strict_containment::outside};
  }
}
} // namespace cut
} // namespace tf
