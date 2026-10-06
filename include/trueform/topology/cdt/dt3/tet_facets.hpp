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
#include <array>
#include <cstddef>
#include <type_traits>

namespace tf::topology::cdt::dt3 {

/// The corners of the facet opposite each corner of a cell, wound so that
/// the cell's own interior lies on the facet's negative side. Two cells
/// sharing a facet therefore wind it oppositely, and that single fact is the
/// whole adjacency law: the hull cell's slot-3 facet is its finite face read
/// outward, and a star's new cell inherits its base facet already reversed.
inline constexpr int tet_facet_corners[4][3] = {
    {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};

/// The facet of `corners` opposite `slot`.
template <typename Corners>
auto tet_facet(const Corners &corners, std::size_t slot)
    -> std::array<std::decay_t<decltype(corners[0])>, 3> {
  const auto *of = tet_facet_corners[slot];
  return {corners[std::size_t(of[0])], corners[std::size_t(of[1])],
          corners[std::size_t(of[2])]};
}

} // namespace tf::topology::cdt::dt3
