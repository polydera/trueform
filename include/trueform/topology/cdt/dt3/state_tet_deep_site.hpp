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
#include "../../../core/config/verify.hpp"
#include "./tet_deep_lattice.hpp"
#include "./tet_position.hpp"
#include "./tetrahedralization_owner.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// State that the site named `name`, entering the lane, stands at
/// `position`: a row where it stands below the frame and its ticket by name,
/// nothing where it stands on it. The one writer of the rows, their tickets
/// and the depth no site has stood below; a name states one row, and no
/// position stands deeper than the owner's deep lattice admits
/// (@ref tet_deep_lattice).
template <typename Index, typename Coord, typename Int>
auto state_tet_deep_site(tetrahedralization_owner<Index, Coord, Int> &owner,
                         Index name, const tet_position<Coord> &position)
    -> void {
  using Owner = tetrahedralization_owner<Index, Coord, Int>;
  if (!position.depth)
    return;
  auto &at = owner._deep_at;
  if (std::size_t(name) >= at.size()) {
    const std::size_t known = at.size();
    at.reallocate(std::size_t(name) + 1);
    std::fill(at.begin() + std::ptrdiff_t(known), at.end(), Owner::none);
  }
  if constexpr (tf::core::verification) {
    tf::core::verify(position.depth <= tet_deep_lattice<Int, Coord>::cap,
                     "a site stands no deeper than its deep lattice admits");
    tf::core::verify(at[std::size_t(name)] == Owner::none,
                     "a name states one deep row");
  }
  at[std::size_t(name)] = Index(owner._deep.size());
  owner._deep.push_back({name, position.depth, position.residual});
  owner._deepest = std::max(owner._deepest, position.depth);
}

} // namespace tf::topology::cdt::dt3
