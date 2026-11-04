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
#include <array>

namespace tf::topology::cdt::dt3 {

/// A facet the Lawson pass could not flip: a cell holding it and its
/// corners, the reflex edge a b first, as the reference records them.
template <typename Index> struct tet_stuck_facet {
  Index cell;
  std::array<Index, 3> corners;
};

} // namespace tf::topology::cdt::dt3
