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
#include <array>

namespace tf::spatial {

/// The build-time moment row, in double throughout: the raw mixed
/// second moment (row-major 3x3) and third moment (i-major, symmetric
/// jk packed xx, xy, xz, yy, yz, zz) about the row's own centroid —
/// the shapes the shift theorem composes. The stored
/// @ref winding_moment folds these once, at finalize, into the
/// evaluation's combinations; the raw row never leaves the build.
struct raw_winding_moment {
  std::array<double, 3> position;
  double area;
  std::array<double, 3> directed_area;
  std::array<double, 9> second_moment;
  std::array<double, 18> third_moment;
};

} // namespace tf::spatial
