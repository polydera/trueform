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
#include "./tet_face_crossing.hpp"

namespace tf::topology::cdt::dt3 {

/// A crossing a recovery's walk met, named where it refuses: the `segment`
/// phase S could not recover or the `face` phase F could not cover, the
/// other `-1`, and the constraint it `crosses`.
template <typename Index> struct tet_crossing_pair {
  Index segment = -1;
  Index face = -1;
  tet_face_crossing<Index> crosses;
};

} // namespace tf::topology::cdt::dt3
