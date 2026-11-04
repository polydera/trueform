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
#include "./tet_edge_ref.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The edge's place among its cell's six, numbered by corner-slot pair
/// (0,1) (0,2) (0,3) (1,2) (1,3) (2,3).
template <typename Index>
auto tet_edge_slot(const tet_edge_ref<Index> &edge) -> std::size_t {
  const std::size_t lo = std::min(edge.a, edge.b);
  const std::size_t hi = std::max(edge.a, edge.b);
  return lo == 0 ? hi - 1 : lo + hi;
}

} // namespace tf::topology::cdt::dt3
