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
#include "../../../exact/orient3d.hpp"
#include "./finite_tet_conflicts.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Whether `site` lies in the conflict cavity of the live cell `tet`, `1` or
/// `0`, by the owner's admission law. A hull cell whose plane holds the site
/// reads its finite mate only once `hold(mate)` admits it; a refused hold
/// answers `-1`.
template <typename Owner, typename Hold>
auto tet_conflicts(const Owner &owner, typename Owner::index_type tet,
                   typename Owner::index_type site, const Hold &hold) -> int {
  const auto corners = owner._corners[std::size_t(tet)];
  if (corners[3] != Owner::infinite)
    return finite_tet_conflicts(owner, tet, site);
  const auto visibility =
      tf::exact::orient3d_value_scaled<typename Owner::int_type>(
          owner._sites[std::size_t(corners[0])].pt,
          owner._sites[std::size_t(corners[1])].pt,
          owner._sites[std::size_t(corners[2])].pt,
          owner._sites[std::size_t(site)].pt);
  if (visibility != 0)
    return visibility > 0;
  const auto mate = owner._neighbors[std::size_t(tet)][3];
  if (!hold(mate))
    return -1;
  return finite_tet_conflicts(owner, mate, site);
}

} // namespace tf::topology::cdt::dt3
