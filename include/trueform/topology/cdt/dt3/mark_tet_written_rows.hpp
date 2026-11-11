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
#include "./tetrahedralization_owner.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Mark the rows `owner` recorded writing: every row written after the mark
/// is recorded past it, so a reader of the records from the mark reads every
/// row written since. Returns the mark.
template <typename Owner> auto mark_tet_written_rows(Owner &owner) -> std::size_t {
  owner._written_floor = owner._written.size();
  return owner._written_floor;
}

} // namespace tf::topology::cdt::dt3
