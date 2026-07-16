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

namespace tf {

enum class intersect_mode : int {
  sos = 1,        // SoS fan triangulation — all records are (edge, face)
  primitives = 2, // Conforming 5-type classification (EF, EE, VE, VF, VV)
  resolve_crossing_contours = 4,       // crossings between different contours (tag_i,tag_j) vs (tag_i,tag_k)
  resolve_self_crossing_contours = 8,  // self-crossings within a single contour (tag_i,tag_j)
  resolve_contours = resolve_crossing_contours | resolve_self_crossing_contours,
  // Atomic bit: also generate each form's self-intersection records
  self_intersections = 16,
  // What callers write: a self contour class only has self-crossings,
  // so generating self records implies resolving them
  within = self_intersections | resolve_self_crossing_contours
};

constexpr auto operator|(intersect_mode a, intersect_mode b) -> intersect_mode {
  return static_cast<intersect_mode>(static_cast<int>(a) | static_cast<int>(b));
}

constexpr auto operator&(intersect_mode a, intersect_mode b) -> bool {
  return (static_cast<int>(a) & static_cast<int>(b)) != 0;
}

} // namespace tf
