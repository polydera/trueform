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
#include "./complement.hpp"
#include "./difference.hpp"
#include "./expr.hpp"
#include "./intersection.hpp"
#include "./merge.hpp"
#include <utility>

namespace tf::csg {

/// @ingroup csg_expression
/// @brief Leaf factory — same as `expr{i}` but reads as code.
///
/// @code
/// auto e = tf::csg::op(0) | tf::csg::op(1) - tf::csg::op(2);
/// @endcode
inline auto op(int i) -> expr { return expr{i}; }

/// @ingroup csg_expression
/// @brief Union (`a | b` == `merge(a, b)`).
inline auto operator|(expr a, expr b) -> expr {
  return merge(std::move(a), std::move(b));
}

/// @ingroup csg_expression
/// @brief Intersection (`a & b` == `intersection(a, b)`).
inline auto operator&(expr a, expr b) -> expr {
  return intersection(std::move(a), std::move(b));
}

/// @ingroup csg_expression
/// @brief Difference (`a - b` == `difference(a, b)`).
inline auto operator-(expr a, expr b) -> expr {
  return difference(std::move(a), std::move(b));
}

/// @ingroup csg_expression
/// @brief Complement (`~a` == `complement(a)`).
inline auto operator~(expr a) -> expr { return complement(std::move(a)); }

} // namespace tf::csg
