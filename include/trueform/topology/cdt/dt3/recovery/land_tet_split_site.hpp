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
#include "../tet_position.hpp"
#include "../tet_site_position.hpp"
#include "./walk_to_tet_point.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The standing site of the published complex of `dt` at the position `at`,
/// found by a walk from the cell `hint`, or `-1` when none stands there.
template <typename DT>
auto land_tet_split_site(const DT &dt,
                         const tet_position<typename DT::coord_type> &at,
                         typename DT::index_type hint) ->
    typename DT::index_type {
  using Index = typename DT::index_type;
  const auto cell = walk_to_tet_point(dt, at, hint);
  Index site = Index(-1);
  if (cell == DT::k_none)
    return site;
  for (auto corner : dt.tets()[std::size_t(cell)])
    if (tet_site_position(dt.owner(), corner) == at)
      site = corner;
  return site;
}

} // namespace tf::topology::cdt::dt3
