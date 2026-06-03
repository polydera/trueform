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
/// @brief Logical NOT of the argument.
template <typename X>
auto complement(X &&x) -> expr {
  return expr{expr::kind::complement,
              detail::make_children(std::forward<X>(x))};
}

} // namespace tf::csg
