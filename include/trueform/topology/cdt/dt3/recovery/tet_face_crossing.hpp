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

/// The constraint a refused face was found crossing: `segment`, the segment
/// whose piece pierces it, or `face`, the face whose protected facet holds
/// the edge piercing it, `-1` where it names none.
template <typename Index> struct tet_face_crossing {
  Index segment = -1;
  Index face = -1;
};

} // namespace tf::topology::cdt::dt3
