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
#include "../../../exact/insphere.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Test a positively oriented cell, symbolic ranks breaking cospherical ties.
template <typename Owner>
auto finite_tet_conflicts(const Owner &owner, typename Owner::index_type tet,
                          typename Owner::index_type site) -> bool {
  const auto corners = owner._corners[std::size_t(tet)];
  return tf::exact::insphere_conflict_scaled<typename Owner::int_type>(
      owner._sites[std::size_t(corners[0])],
      owner._sites[std::size_t(corners[1])],
      owner._sites[std::size_t(corners[2])],
      owner._sites[std::size_t(corners[3])], owner._sites[std::size_t(site)],
      1);
}

} // namespace tf::topology::cdt::dt3
