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
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// What face recovery made of one face.
enum class tet_face_status : std::uint8_t {
  pending,   ///< Still due; never published.
  recovered, ///< The lane facets it owns tile it.
  refused    ///< A rim segment refused, a segment pierces it, a point had no place, or an edge piercing it could be neither flipped nor cut away.
};

} // namespace tf::topology::cdt::dt3
