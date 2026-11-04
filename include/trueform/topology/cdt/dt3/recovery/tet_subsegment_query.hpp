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

namespace tf::topology::cdt::dt3 {

/// What one round read of a subsegment: whether the complex holds it,
/// whether a redirect reads it encroached, the encroacher a split is
/// placed by, `-1` for none, with the cosine of the angle it sees the piece
/// under, the site strictly on it a missing
/// one meets, `-1` for none, and the cell the walk to its first end reached.
template <typename Index> struct tet_subsegment_query {
  Index on;
  Index hint;
  Index encroacher;
  double cosine;
  bool present;
  bool encroached;
};

} // namespace tf::topology::cdt::dt3
