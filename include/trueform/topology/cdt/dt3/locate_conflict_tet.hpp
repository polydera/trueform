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
#include "./step_tet_walk.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Walk from `hint` to a finite cell holding the site or a hull cell seeing
/// it.
template <typename Owner>
auto locate_conflict_tet(const Owner &owner, typename Owner::index_type site,
                         typename Owner::index_type hint) ->
    typename Owner::index_type {
  const auto &p = owner._sites[std::size_t(site)].pt;
  auto from = Owner::none;
  for (auto tet = hint;;) {
    const auto next = step_tet_walk(owner, tet, from, p);
    if (next == Owner::none)
      return tet;
    from = tet;
    tet = next;
  }
}

template <typename Owner>
auto locate_conflict_tet(const Owner &owner, typename Owner::index_type site) ->
    typename Owner::index_type {
  return locate_conflict_tet(owner, site, owner._hint);
}

} // namespace tf::topology::cdt::dt3
