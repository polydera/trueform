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
#include "../../../../../core/buffer.hpp"
#include "../../../../../core/config/verify.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// THE ADMISSION LAW: a task of patch `patch` reads or writes row `row` only
/// where the row's own region is its own, and A ROW'S REGION IS A FUNCTION,
/// NOT A LANE — it is the patch of its FIRST CORNER
/// (@ref make_tet_first_facet_patches, @ref tet_patch_tables): two loads, the
/// corners line the caller reads next anyway, and no table over the rows to
/// state, carry through a compaction or mark at a birth. A site past the
/// patches holds none, so a row standing on one is nobody's and every task
/// that reaches it stops.
template <typename Owner, typename Range>
auto tet_row_admits(const Range &patch_of_site, const Owner &owner,
                    typename Owner::index_type row,
                    typename Owner::index_type patch) -> bool {
  const auto site = owner._corners[std::size_t(row)][0];
  return site >= 0 && std::size_t(site) < patch_of_site.size() &&
         patch_of_site[std::size_t(site)] == patch;
}

} // namespace tf::topology::cdt::dt3
