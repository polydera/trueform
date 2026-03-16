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

namespace tf::intersect::graph {

enum class vertex_source : uint8_t { original = 0, created = 1 };

template <typename Index> struct vertex {
  vertex_source source;
  Index id;
  Index sub_id; // original: face-local edge index
                // created: flat intersection index

  auto operator==(const vertex &o) const {
    return source == o.source && id == o.id;
  }
};

} // namespace tf::intersect::graph
