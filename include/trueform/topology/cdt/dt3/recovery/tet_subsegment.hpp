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
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// One piece of a protected segment between two site slots, `a` toward the
/// segment's first end, and `next` the row continuing it from `b`, `-1` at
/// the last. `n0` and `n1` place the ends on the segment's own dyadic frame,
/// `-1` for a shared site off it; `halvings` counts the splits that made the
/// piece. `hint` is a cell row near `a` from the last query.
template <typename Index> struct tet_subsegment {
  Index segment;
  Index a;
  Index b;
  Index next;
  std::int64_t n0;
  std::int64_t n1;
  Index hint;
  int halvings;
};

} // namespace tf::topology::cdt::dt3
