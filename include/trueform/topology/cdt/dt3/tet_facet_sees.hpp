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
#include "./tet_orient3d_sites.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Whether `site` lies strictly on the positive side of the finite facet of
/// `cell` opposite `slot`, the side away from the cell's interior.
template <typename Owner>
auto tet_facet_sees(const Owner &owner, typename Owner::index_type cell,
                    std::size_t slot, typename Owner::index_type site) -> bool {
  const auto facet = tet_facet(owner._corners[std::size_t(cell)], slot);
  return tet_orient3d_sites(owner, facet[0], facet[1], facet[2], site) > 0;
}

} // namespace tf::topology::cdt::dt3
