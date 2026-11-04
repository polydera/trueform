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
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Whether `cell` names a live finite cell of the complex `dt` reads. A
/// published complex numbers its finite cells first and holds no dead one.
template <typename DT>
auto tet_finite_cell(const DT &dt, typename DT::index_type cell) -> bool {
  return cell >= 0 && std::size_t(cell) < dt.n_tets();
}

/// A live finite cell of the complex `dt` reads, for a walk with no other
/// place to start.
template <typename DT>
auto tet_entry_cell(const DT &) -> typename DT::index_type {
  return typename DT::index_type(0);
}

/// A cell of the complex `dt` reads that holds `site`, or `k_none` where
/// `dt` keeps none, for a star gathered without a walk.
template <typename DT>
auto tet_site_cell(const DT &, typename DT::index_type) ->
    typename DT::index_type {
  return DT::k_none;
}

} // namespace tf::topology::cdt::dt3
