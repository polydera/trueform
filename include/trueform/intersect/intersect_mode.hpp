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

/// The classifier and optional within request for an intersection run.
/// `primitives | within` is the canonical spelling for self-intersections.
enum class intersect_mode : int {
  // Every contact is perturbed into a generic crossing, so all records
  // are (edge, face) and none is ever coplanar.
  sos = 1,
  // Each contact is stated as what it is: EF, EE, VE, VF, VV.
  primitives = 2,
  within = 4,
};

constexpr auto operator|(intersect_mode a, intersect_mode b) -> intersect_mode {
  return static_cast<intersect_mode>(static_cast<int>(a) | static_cast<int>(b));
}

constexpr auto operator&(intersect_mode a, intersect_mode b) -> bool {
  return (static_cast<int>(a) & static_cast<int>(b)) != 0;
}

} // namespace tf
