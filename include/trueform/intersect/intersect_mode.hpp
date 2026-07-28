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
  // Crossings between contours of DIFFERENT classes, (tag_i,tag_j) vs
  // (tag_i,tag_k). Such a pair needs a third tag to exist, so the graph
  // derives this from arity and never reads the flag; it is declarative.
  resolve_crossing_contours = 4,
  // Crossings within one contour class (tag_i,tag_j): a contour with
  // itself, or with another contour of the same pair — e.g. two disjoint
  // components of one mesh cutting the same face.
  resolve_self_crossing_contours = 8,
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
