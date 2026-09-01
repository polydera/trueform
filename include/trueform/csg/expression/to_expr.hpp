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
#include <type_traits>
#include <utility>

namespace tf::csg {

/// @ingroup csg_expression
/// @brief Promote any argument to an @ref expr: integral types are
///        wrapped as leaves; everything else is forwarded as-is
///        (so existing @ref expr values pass through unchanged).
template <typename T>
auto to_expr(T &&x) -> expr {
  if constexpr (std::is_integral_v<std::decay_t<T>>)
    return expr{static_cast<int>(x)};
  else
    return expr{std::forward<T>(x)};
}

} // namespace tf::csg
