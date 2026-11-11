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

namespace tf::topology::cdt::dt3 {

/// A facet, corner slots ascending, beside a face that owns it, and the
/// `layers` of that face the facet covers: the face's own triangles lying on
/// one another over it count once each.
template <typename Index> struct tet_facet_owner {
  std::array<Index, 3> key;
  Index face;
  Index layers = 1;
};

} // namespace tf::topology::cdt::dt3
