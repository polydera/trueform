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
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// A site standing below the frame, by its name: the depth and residual of
/// its position (@ref tet_position), whose floor its slot holds.
template <typename Index> struct tet_deep_site {
  Index id;
  int depth;
  std::array<std::int64_t, 3> residual;
};

} // namespace tf::topology::cdt::dt3
