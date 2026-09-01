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
#include "./expr.hpp"
#include "./make_children.hpp"
#include <utility>

namespace tf::csg {

/// @ingroup csg_expression
/// @brief Intersection over the arguments.
///
/// Integral arguments are auto-promoted to leaves; @ref expr
/// arguments are forwarded as-is.
template <typename... Args>
auto intersection(Args &&...args) -> expr {
  return expr{expr::kind::intersection,
              make_children(std::forward<Args>(args)...)};
}

} // namespace tf::csg
