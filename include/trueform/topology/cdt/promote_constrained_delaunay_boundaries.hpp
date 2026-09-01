/*
 * Copyright (c) 2025 XLAB
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

#include "./constrained_delaunay_edge_constraint_provenance.hpp"

#include <cstddef>

namespace tf::topology::cdt {

/// Promote selected retained physical constraints to region boundaries. The
/// mask is indexed by the latest original input owner; false entries preserve
/// generic CDT occurrence parity.
template <typename Owner, typename Promotions>
auto promote_constrained_delaunay_boundaries(Owner &owner,
                                             const Promotions &promotions)
    -> void {
  using Index = typename Owner::index_type;
  for (std::size_t edge = 0; edge + 1 < owner._edges.size(); edge += 2) {
    if (owner._edges[edge].constrained == Owner::unconstrained)
      continue;
    const auto provenance = constrained_delaunay_edge_constraint_provenance(
        owner, static_cast<Index>(edge));
    if (provenance.input_id == Owner::none ||
        std::size_t(provenance.input_id) >= promotions.size() ||
        promotions[std::size_t(provenance.input_id)] == 0)
      continue;
    owner._edges[edge].constrained = Owner::boundary_constrained;
    owner._edges[edge + 1].constrained = Owner::boundary_constrained;
  }
}

} // namespace tf::topology::cdt
