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
#include "./to_expr.hpp"
#include <utility>
#include <vector>

namespace tf::csg {

/// @ingroup csg_expression
/// @brief Variadic helper: convert each argument to an @ref expr
///        (integral → leaf, expr → expr) and collect them in a
///        size-reserved vector.
template <typename... Args>
auto make_children(Args &&...args) -> std::vector<expr> {
  std::vector<expr> children;
  children.reserve(sizeof...(Args));
  (children.emplace_back(to_expr(std::forward<Args>(args))), ...);
  return children;
}

} // namespace tf::csg
