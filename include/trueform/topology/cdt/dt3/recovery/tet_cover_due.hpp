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

/// A facet a spread of a face's cover has taken and whose own edges it owes
/// the spread: the edge `u` `v` it was reached across and the apex `x` it
/// closed on (@ref spread_tet_face_cover).
template <typename Index> struct tet_cover_due {
  Index u, v, x;
};

} // namespace tf::topology::cdt::dt3
