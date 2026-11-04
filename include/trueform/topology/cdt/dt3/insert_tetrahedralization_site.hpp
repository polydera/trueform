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
#include "./keep_taken_tet_edges.hpp"
#include "./locate_conflict_tet.hpp"
#include "./split_tetrahedralization_site.hpp"
#include "./star_tet_cavity.hpp"
#include "./tet_insertion.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Insert one site and compact when dead cells reach the owner's threshold:
/// by its conflict region and star while the complex is Delaunay, by the
/// split of the simplex it lies in once a flip has made it otherwise
/// (@ref split_tetrahedralization_site) — a site beyond the hull, which only
/// a hull cell sees, takes the flood in either case, the hull being its
/// cavity. The edges it records as taken are the ones its star does not
/// hold again.
template <typename Owner>
auto insert_tetrahedralization_site(Owner &owner,
                                    typename Owner::index_type site)
    -> tet_insertion {
  const auto at = locate_conflict_tet(owner, site);
  if (!owner._delaunay && owner._corners[std::size_t(at)][3] != Owner::infinite)
    return split_tetrahedralization_site(owner, site, at);
  const std::size_t recorded = owner._killed.size();
  if (!flood_tet_cavity(owner, at, site))
    return tet_insertion::wrapped;
  if (std::size_t(owner._corners.size()) + owner._boundary.size() >
      std::size_t(Owner::max_tets))
    return tet_insertion::capacity;

  owner._hint = star_tet_cavity(owner, site);
  if (owner._record_killed)
    keep_taken_tet_edges(owner._killed, recorded, owner._links);
  ++owner._stats.insertions;
  const std::size_t slots = std::size_t(owner._corners.size());
  if (owner._n_dead * Owner::compaction_dead_divisor >= slots)
    compact_tetrahedralization(owner);
  return tet_insertion::placed;
}

} // namespace tf::topology::cdt::dt3
