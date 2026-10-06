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
#include "../../../core/buffer.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Per site of a published complex, the last finite cell in published order
/// that holds it, or `k_none` for a site no cell holds.
template <typename DT, typename Index>
auto state_tet_site_cells(const DT &dt, tf::buffer<Index> &cells) -> void {
  cells.allocate_and_initialize(dt.n_sites(), DT::k_none);
  const auto tets = dt.tets();
  for (std::size_t cell = 0; cell < tets.size(); ++cell)
    for (auto site : tets[cell])
      cells[std::size_t(site)] = Index(cell);
}

} // namespace tf::topology::cdt::dt3
