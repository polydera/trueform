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

template <typename Index> struct edge {
  short tag;
  short tag_other;
  Index object;
  Index object_other;
  Index point_0;
  Index point_1;
  Index id; // canonical group ID
  std::int16_t ordinal;     // base-loop position of start vertex; -1 if interior
  std::int16_t sub_ordinal; // parametric order along parent base-loop segment; -1 if interior
  // Endpoint identity is the pair (point_tag, point): -1 = intersection
  // point, otherwise an original vertex of that form. Record-derived
  // edges are all-created; only tolerated base-loop chords carry
  // originals.
  std::int16_t point_tag_0 = -1;
  std::int16_t point_tag_1 = -1;
};

} // namespace tf::intersect::graph
