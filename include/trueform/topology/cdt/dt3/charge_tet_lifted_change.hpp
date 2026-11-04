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
#include "./tet_lifted_change.hpp"
#include "./tet_lifted_volume.hpp"
#include "./tet_site_position.hpp"
#include "./tet_with_sites.hpp"
#include "./tetrahedralization_owner.hpp"
#include <cstddef>
#include <type_traits>

namespace tf::topology::cdt::dt3 {

/// Charge `change` with the lifted volume of the finite cells among `cells`,
/// added where `sign` is positive and taken away otherwise, every cell read
/// at the owner's deepest depth (@ref tet_lifted_change).
template <typename Index, typename Coord, typename Int, typename Cells>
auto charge_tet_lifted_change(
    const tetrahedralization_owner<Index, Coord, Int> &owner,
    tet_lifted_change<Int, Coord> &change, const Cells &cells, int sign)
    -> void {
  using owner_t = tetrahedralization_owner<Index, Coord, Int>;
  tet_with_sites(owner, tet_deepest_depth(owner), [&](const auto &sites) {
    using sites_t = std::decay_t<decltype(sites)>;
    using T3 = typename tf::exact::meta<typename sites_t::int_type>::T3;
    T3 sum(0);
    for (auto cell : cells) {
      const auto c = owner._corners[std::size_t(cell)];
      if (c[3] != owner_t::infinite)
        sum += tet_lifted_volume(sites, c[0], c[1], c[2], c[3]);
    }
    auto &lane = [&change]() -> T3 & {
      if constexpr (sites_t::deep)
        return change.deep;
      else
        return change.shallow;
    }();
    lane = sign > 0 ? lane + sum : lane - sum;
  });
}

} // namespace tf::topology::cdt::dt3
