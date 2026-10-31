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
#include "../../core/config/verify.hpp"
#include "./plane_tables.hpp"

#include <algorithm>

namespace tf::intersect::graph {

/// A carve's statement accounts every created identity it mints: each of
/// its `n_minted` mints past `n_standing` is named by a row of the tables it
/// stated (`after`).
template <typename Index, typename Int>
auto assert_carve_identities_are_accounted(
    const plane_tables<Index, Int> &after, Index n_standing, Index n_minted)
    -> void {
  tf::buffer<Index> stated;
  for (const auto &def : after.edge_defs()) {
    if (def.point_tag_0 < 0)
      stated.push_back(def.point_0);
    if (def.point_tag_1 < 0)
      stated.push_back(def.point_1);
  }
  std::sort(stated.begin(), stated.end());
  stated.erase_till_end(std::unique(stated.begin(), stated.end()));
  for (Index k = 0; k < n_minted; ++k)
    tf::core::verify(
        std::binary_search(stated.begin(), stated.end(), n_standing + k),
        "a carve's mint is named by a row it states");
}

} // namespace tf::intersect::graph
