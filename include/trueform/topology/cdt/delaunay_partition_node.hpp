/*
 * Copyright (c) 2025 XLAB
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
#include "./delaunay_frontier.hpp"
#include <array>
#include <cstddef>
#include <limits>

namespace tf::topology::cdt {

template <typename Index> struct delaunay_partition_node {
  static constexpr std::size_t no_child =
      std::numeric_limits<std::size_t>::max();

  auto is_terminal() const -> bool { return children[0] == no_child; }

  std::size_t first = 0;
  std::size_t last = 0;
  std::array<std::size_t, 2> children{no_child, no_child};
  delaunay_axis axis = delaunay_axis::x;
  delaunay_frontier<Index> frontier{};
};

} // namespace tf::topology::cdt
