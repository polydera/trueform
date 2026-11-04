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
#include <utility>

namespace tf::topology::cdt::dt3 {

/// The four corners of a cell in ascending order, by a five-comparator
/// network: the key a cell is ranked and compared by.
template <typename Index>
auto sort_tet_corner_key(std::array<Index, 4> &key) -> void {
  const auto order = [&key](std::size_t a, std::size_t b) {
    if (key[b] < key[a])
      std::swap(key[a], key[b]);
  };
  order(0, 1);
  order(2, 3);
  order(0, 2);
  order(1, 3);
  order(1, 2);
}

} // namespace tf::topology::cdt::dt3
