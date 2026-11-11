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
#include "../../../../core/buffer.hpp"
#include "../find_tet_site_within.hpp"
#include "../tet_position.hpp"
#include "../tet_site_position.hpp"
#include "./walk_to_tet_point.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The site of the published complex of `dt` the position `at` lands on,
/// found by a walk from the cell `hint`: the site standing at it, else the
/// smallest site `admits` names in its cell of the grid of side `reach`,
/// which the position is
/// (@ref find_tet_site_within), `-1` for none. `take` and `flood` are the
/// caller's scratch for the reach's flood.
template <typename DT, typename Admits, typename Take>
auto land_tet_split_site(const DT &dt,
                         const tet_position<typename DT::coord_type> &at,
                         typename DT::index_type hint,
                         typename DT::coord_type reach, const Admits &admits,
                         const Take &take,
                         tf::buffer<typename DT::index_type> &flood) ->
    typename DT::index_type {
  using Index = typename DT::index_type;
  const auto cell = walk_to_tet_point(dt, at, hint);
  if (cell == DT::k_none)
    return Index(-1);
  for (auto corner : dt.tets()[std::size_t(cell)])
    if (tet_site_position(dt.owner(), corner) == at)
      return corner;
  const Index within =
      find_tet_site_within(dt, at, cell, reach, admits, take, flood);
  return within == DT::k_none ? Index(-1) : within;
}

} // namespace tf::topology::cdt::dt3
