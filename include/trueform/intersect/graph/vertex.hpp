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

#include <cstdint>

#include "../../topology/topo_id.hpp"

namespace tf::intersect::graph {

enum class vertex_source : uint8_t { original = 0, created = 1 };

template <typename Index> struct vertex {
  vertex_source source;
  Index id;
  tf::topo_id<short> sub_id; // where on the original polygon (vertex/edge/face)

  friend auto operator==(const vertex &a, const vertex &b) {
    return a.source == b.source && a.id == b.id;
  }
  friend auto operator!=(const vertex &a, const vertex &b) {
    return !(a == b);
  }
};

} // namespace tf::intersect::graph
