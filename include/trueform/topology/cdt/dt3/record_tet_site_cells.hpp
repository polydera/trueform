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
#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "./tet_site_position.hpp"
#include "./tet_state.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Ask every later write to keep each site's cell, stating the table once
/// from the live finite cells when it is not yet kept: the lowest cell
/// holding each site, the same at any worker count.
template <typename Owner> auto record_tet_site_cells(Owner &owner) -> void {
  using Index = typename Owner::index_type;
  if (owner._record_site_cells)
    return;
  owner._record_site_cells = true;
  owner._site_cells.resize(tet_site_count(owner), Owner::none);
  tf::parallel_for_each(
      tf::make_sequence_range(owner._corners.size()),
      [&owner](std::size_t cell) {
        if (owner._states[cell] != tet_state::live)
          return;
        for (auto corner : owner._corners[cell])
          owner._site_cells.lower(std::size_t(corner), Index(cell),
                                  Owner::none);
      },
      tf::checked);
}

} // namespace tf::topology::cdt::dt3
