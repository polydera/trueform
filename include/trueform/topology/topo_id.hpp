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

#include "./topo_type.hpp"
#include <utility>

namespace tf {

/// Identifies a topological element: a (type, index) pair.
/// E.g. (vertex, 3) or (edge, 5) or (face, 0).
template <typename Index> struct topo_id {
  Index id;
  tf::topo_type label;

  friend auto operator<(const topo_id &a, const topo_id &b) -> bool {
    return std::make_pair(a.label, a.id) < std::make_pair(b.label, b.id);
  }

  friend auto operator==(const topo_id &a, const topo_id &b) -> bool {
    return std::make_pair(a.label, a.id) == std::make_pair(b.label, b.id);
  }
};

} // namespace tf
