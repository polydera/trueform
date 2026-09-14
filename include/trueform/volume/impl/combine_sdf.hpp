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
#include "../boolean_op.hpp"
#include <algorithm>

namespace tf {
namespace volume_detail {

// Combine two SDF samples under `op`, using the negative-inside convention
// (`< 0` is inside, matching `tf::make_isosurface`). These are the exact,
// bound-preserving CSG combinators:
//   union        = min(a, b)      (inside either)
//   intersection = max(a, b)      (inside both)
//   difference   = max(a, -b)     (inside A, outside B)
// They are not everywhere-exact distances (min/max can under/over-estimate
// distance away from the surface), but the zero level set — and therefore the
// extracted isosurface — is exactly the boolean of the two input solids.
template <typename Real>
inline auto combine_sdf(tf::volume_boolean_op op, Real a, Real b) -> Real {
  switch (op) {
  case tf::volume_boolean_op::union_:
    return std::min(a, b);
  case tf::volume_boolean_op::intersection:
    return std::max(a, b);
  case tf::volume_boolean_op::difference:
    return std::max(a, -b);
  }
  return std::min(a, b);
}

} // namespace volume_detail
} // namespace tf
