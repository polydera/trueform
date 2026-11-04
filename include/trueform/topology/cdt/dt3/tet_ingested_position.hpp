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
#include "../../../exact/vertex.hpp"
#include "./tet_minted_site.hpp"
#include "./tet_position.hpp"

namespace tf::topology::cdt::dt3 {

/// The position a site record handed to a build or an append stands at: a
/// minted site's own, as given.
template <typename Index, typename Coord>
auto tet_ingested_position(const tet_minted_site<Index, Coord> &site)
    -> tet_position<Coord> {
  return site.position;
}

/// The position a caller's site record stands at, minted as it enters.
template <typename Index, typename Coord>
auto tet_ingested_position(const tf::exact::vertex<Index, Coord> &site)
    -> tet_position<Coord> {
  return tet_minted_position<Coord>(site.pt);
}

} // namespace tf::topology::cdt::dt3
