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

#include "./int128.hpp"
#include "./vertex.hpp"

namespace tf::exact {

/// Exact 3x3 determinant det(a, b, c) of three int32 3D points.
/// Returns the scalar triple product a . (b x c).
inline auto determinant_value(const pt3 &a, const pt3 &b, const pt3 &c)
    -> int128 {
  using I64 = int64_t;
  using I = int128;
  I64 b1c2 = I64(b[1]) * c[2], b2c1 = I64(b[2]) * c[1];
  I64 b2c0 = I64(b[2]) * c[0], b0c2 = I64(b[0]) * c[2];
  I64 b0c1 = I64(b[0]) * c[1], b1c0 = I64(b[1]) * c[0];
  return I(a[0]) * (I(b1c2) - b2c1) + I(a[1]) * (I(b2c0) - b0c2) +
         I(a[2]) * (I(b0c1) - b1c0);
}

/// Exact sign of det(a, b, c). Returns -1, 0, or +1.
inline auto determinant_sign(const pt3 &a, const pt3 &b, const pt3 &c)
    -> int {
  auto val = determinant_value(a, b, c);
  return (val > 0) ? 1 : (val < 0) ? -1 : 0;
}

} // namespace tf::exact
