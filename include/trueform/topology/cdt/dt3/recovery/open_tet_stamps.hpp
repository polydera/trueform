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
#include "../../../../core/algorithm/parallel_fill.hpp"
#include "../../../../core/views/drop.hpp"
#include "./tet_stamps.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// Open a generation of `stamps` over `keys` keys: marks grow to cover them,
/// fresh ones clear, and a generation that wraps clears all.
inline auto open_tet_stamps(tet_stamps &stamps, std::size_t keys) -> void {
  auto &marks = stamps.marks;
  const std::size_t known = marks.size();
  if (keys > known) {
    marks.reallocate(keys);
    tf::parallel_fill(tf::drop(marks, known), std::uint32_t(0));
  }
  if (++stamps.generation == 0) {
    tf::parallel_fill(marks, std::uint32_t(0));
    stamps.generation = 1;
  }
}

/// Take `key` in the open generation of `stamps`: true the first time.
inline auto take_tet_stamp(tet_stamps &stamps, std::size_t key) -> bool {
  if (stamps.marks[key] == stamps.generation)
    return false;
  stamps.marks[key] = stamps.generation;
  return true;
}

} // namespace tf::topology::cdt::dt3
