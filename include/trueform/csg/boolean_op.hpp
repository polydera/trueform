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

namespace csg {
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
} // namespace csg

/// @ingroup csg
/// @brief The winding each operand's surviving faces carry under `op`.
///
/// A difference reverses the operand it carves with; union and intersection
/// reverse neither. Consumers that map an output face's edge slots back to its
/// input face — @ref tf::stitched_manifold_edge_link — need this.
inline constexpr auto make_directions(boolean_op op)
    -> std::array<direction, 2> {
  const auto spec = csg::make_boolean_op_spec(op);
  return make_directions(spec[0], spec[1]);
}
} // namespace tf
