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
#include "./tet_facets.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// A facet's key, its corners ascending: two cells name a shared facet by
/// the same key, and the protected lane holds keys.
template <typename Index>
auto tet_facet_key(std::array<Index, 3> facet) -> std::array<Index, 3> {
  std::sort(facet.begin(), facet.end());
  return facet;
}

/// The key of the facet of `corners` opposite `slot`.
template <typename Corners>
auto tet_facet_key(const Corners &corners, std::size_t slot) {
  return tet_facet_key(tet_facet(corners, slot));
}

} // namespace tf::topology::cdt::dt3
