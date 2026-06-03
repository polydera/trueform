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
/// @brief `lhs \ rhs`. Equivalent to `intersection(lhs,
///        complement(rhs))` but evaluates without materialising the
///        complement.
template <typename Lhs, typename Rhs>
auto difference(Lhs &&lhs, Rhs &&rhs) -> expr {
  return expr{expr::kind::difference,
              detail::make_children(std::forward<Lhs>(lhs),
                                     std::forward<Rhs>(rhs))};
}

} // namespace tf::csg
