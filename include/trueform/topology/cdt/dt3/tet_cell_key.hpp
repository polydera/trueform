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
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// A cell's corners in the order it is ranked by, paired with its row, the
/// slot order it is published in and whether that order exchanges two of
/// the ranked corners: a finite cell's ascending, the last two exchanged
/// (@ref canonical_tet_key); a hull cell's finite corners ascending and the
/// infinite one last, the second and third exchanged
/// (@ref canonical_tet_hull_key).
template <typename Index> struct tet_cell_key {
  std::array<Index, 4> corners;
  Index row;
  std::uint8_t slots;
  bool exchanged;
};

} // namespace tf::topology::cdt::dt3
