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
#include "../tet_finite_cell.hpp"
#include "../tet_site_position.hpp"
#include "./walk_to_tet_point.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// A finite cell of `dt` holding the site `site`: its site cell where
/// `dt` keeps one that still holds it, otherwise the cell a walk from
/// `hint` reaches. `ticketed` says the site cell answered.
template <typename DT>
auto enter_tet_site_star(const DT &dt, typename DT::index_type site,
                         typename DT::index_type hint, bool &ticketed) ->
    typename DT::index_type {
  const auto cell = tet_site_cell(dt, site);
  ticketed = false;
  if (tet_finite_cell(dt, cell)) {
    for (auto corner : dt.tets()[std::size_t(cell)])
      ticketed = ticketed || corner == site;
    if (ticketed)
      return cell;
  }
  return walk_to_tet_point(dt, tet_site_position(dt.owner(), site), hint);
}

} // namespace tf::topology::cdt::dt3
