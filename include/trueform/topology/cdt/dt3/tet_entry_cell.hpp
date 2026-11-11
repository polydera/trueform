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
#include "./tet_owner_complex.hpp"

namespace tf::topology::cdt::dt3 {

/// The finite cell the owner's last write left beside its hint.
template <bool Parallel, typename Owner, typename Work>
auto tet_entry_cell(const tet_owner_complex<Parallel, Owner, Work> &dt) ->
    typename Owner::index_type {
  return dt.entry();
}

} // namespace tf::topology::cdt::dt3
