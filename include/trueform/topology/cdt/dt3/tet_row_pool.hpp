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

/// The rows a region's own work takes when a round runs the regions beside
/// each other: a block the round reserved for it, `[next, end)`, taken by a
/// cursor. A flip appends to the owner where no pool is open, which is every
/// serial round; an open pool holds a row for every row the round's flips can
/// make, which is the reserving round's contract.
struct tet_row_pool {
  std::size_t next = 0;
  std::size_t end = 0;

  auto open() const -> bool { return end != 0; }
  auto spent() const -> bool { return next == end; }
};

} // namespace tf::topology::cdt::dt3
