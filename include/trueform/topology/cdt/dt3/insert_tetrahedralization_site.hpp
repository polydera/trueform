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
#include "./compact_tetrahedralization.hpp"
#include "./flood_tet_cavity.hpp"
#include "./locate_conflict_tet.hpp"
#include "./star_tet_cavity.hpp"
#include "./tet_cavity_edge_table.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Insert one site and compact when dead cells reach the owner's threshold.
/// Returns false if the replacement cells exceed index capacity.
template <typename Owner>
auto insert_tetrahedralization_site(
    Owner &owner, typename Owner::index_type site,
    tet_cavity_edge_table<typename Owner::index_type> &edges) -> bool {
  flood_tet_cavity(owner, locate_conflict_tet(owner, site), site);
  if (std::size_t(owner._corners.size()) + owner._boundary.size() >
      std::size_t(Owner::max_tets))
    return false;

  owner._hint = star_tet_cavity(owner, site, edges);
  ++owner._stats.insertions;
  const std::size_t slots = std::size_t(owner._corners.size());
  if (owner._n_dead * Owner::compaction_dead_divisor >= slots)
    compact_tetrahedralization(owner);
  return true;
}

} // namespace tf::topology::cdt::dt3
