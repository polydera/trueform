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
#include "./tet_insphere_conflict_sites.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Test a positively oriented cell, symbolic ranks breaking cospherical ties.
template <typename Owner>
auto finite_tet_conflicts(const Owner &owner, typename Owner::index_type tet,
                          typename Owner::index_type site) -> bool {
  const auto corners = owner._corners[std::size_t(tet)];
  return tet_insphere_conflict_sites(owner, corners[0], corners[1], corners[2],
                                     corners[3], site);
}

} // namespace tf::topology::cdt::dt3
