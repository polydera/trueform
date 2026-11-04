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

/// A finite cell's corners in ascending order, paired with its row and the
/// slot order it is published in (@ref canonical_tet_slots).
template <typename Index> struct tet_cell_key {
  std::array<Index, 4> corners;
  Index row;
  std::uint8_t slots;
};

} // namespace tf::topology::cdt::dt3
