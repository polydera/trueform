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
#include "../../../exact/meta.hpp"
#include "./tet_deep_lattice.hpp"
#include "./tet_site_position.hpp"
#include "./tetrahedralization_owner.hpp"

namespace tf::topology::cdt::dt3 {

/// The change of the lifted volume a removal's flips make
/// (@ref tet_lifted_volume), summed at one depth, since no flip moves a
/// site: `shallow` on the owner's `Int` lattice while every site has stood
/// on the frame, `deep` on its deep lattice once one has stood below it
/// (@ref tet_deep_lattice).
template <typename Int, typename Coord> struct tet_lifted_change {
  typename tf::exact::meta<Int>::T3 shallow{};
  typename tf::exact::meta<typename tet_deep_lattice<Int, Coord>::int_type>::T3
      deep{};
};

/// Whether the flips `change` sums lower the lifted volume.
template <typename Index, typename Coord, typename Int>
auto tet_lifted_change_lowers(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    const tet_lifted_change<Int, Coord> &change) -> bool {
  using deep_t = typename tf::exact::meta<
      typename tet_deep_lattice<Int, Coord>::int_type>::T3;
  using shallow_t = typename tf::exact::meta<Int>::T3;
  return tet_deepest_depth(owner) ? change.deep < deep_t(0)
                                  : change.shallow < shallow_t(0);
}

} // namespace tf::topology::cdt::dt3
