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

/// @ingroup csg
/// @brief Union over the arguments (`tf::csg::merge` matches the
///        naming of @ref tf::boolean_op::merge).
///
/// Integral arguments are auto-promoted to leaves; @ref expr
/// arguments are forwarded as-is.
template <typename... Args>
auto merge(Args &&...args) -> expr {
  return expr{expr::kind::merge,
              detail::make_children(std::forward<Args>(args)...)};
}

} // namespace tf::csg
