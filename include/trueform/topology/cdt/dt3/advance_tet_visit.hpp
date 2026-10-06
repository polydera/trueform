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
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// Open a visit over `n` cells and return its stamp: a cell is visited when
/// its mark equals the stamp, so a visit never clears what an earlier one
/// marked, and the marks are cleared only when the stamp wraps.
inline auto advance_tet_visit(tf::buffer<std::uint32_t> &marks,
                              std::uint32_t &stamp, std::size_t n)
    -> std::uint32_t {
  if (marks.size() != n) {
    marks.allocate_and_initialize(n, std::uint32_t(0));
    stamp = 0;
  }
  if (++stamp == 0) {
    std::fill(marks.begin(), marks.end(), std::uint32_t(0));
    stamp = 1;
  }
  return stamp;
}

} // namespace tf::topology::cdt::dt3
