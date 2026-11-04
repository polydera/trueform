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
#include "../../core/buffer.hpp"
#include "../../topology/cdt/dt3/tet_facet_is_protected.hpp"
#include "../../topology/label_connected_components.hpp"
#include <cstddef>

namespace tf::tetmesh {

/// Label every cell of the published complex `owner`, hull cells included,
/// by the region it lies in: cells meeting on a facet the protected lane
/// does not hold share one. Returns the number of regions.
template <typename Owner>
auto label_tet_regions(const Owner &owner,
                       tf::buffer<typename Owner::index_type> &labels)
    -> std::size_t {
  using Index = typename Owner::index_type;
  labels.allocate(std::size_t(owner._corners.size()));
  return std::size_t(tf::label_connected_components<Index>(
      labels, [&owner](Index cell, const auto &apply) {
        for (std::size_t slot = 0; slot < 4; ++slot)
          if (!tf::topology::cdt::dt3::tet_facet_is_protected(owner, cell,
                                                              slot))
            apply(owner._neighbors[std::size_t(cell)][slot]);
      }));
}

} // namespace tf::tetmesh
