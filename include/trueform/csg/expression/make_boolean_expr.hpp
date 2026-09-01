/*
 * Copyright (c) 2026 XLAB
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
#include "../boolean_op.hpp"
#include "./expr.hpp"
#include "./operators.hpp"

namespace tf::csg {

/// @brief The two-operand @ref tf::boolean_op as a CSG expression over
///        operand bits 0 (left) and 1 (right) — the boolean entry
///        points are expression queries on a two-form graph.
inline auto make_boolean_expr(tf::boolean_op op) -> tf::csg::expr {
  switch (op) {
  case tf::boolean_op::merge:
    return tf::csg::op(0) | tf::csg::op(1);
  case tf::boolean_op::intersection:
    return tf::csg::op(0) & tf::csg::op(1);
  case tf::boolean_op::left_difference:
    return tf::csg::op(0) - tf::csg::op(1);
  case tf::boolean_op::right_difference:
  default:
    return tf::csg::op(1) - tf::csg::op(0);
  }
}

} // namespace tf::csg
