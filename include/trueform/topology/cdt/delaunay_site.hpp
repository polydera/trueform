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
#include <cstddef>

namespace tf::topology::cdt {

/// Physically Morton-ordered exact site used by every Delaunay retention tier.
/// `output` keeps the caller-visible vertex identity independent of topology
/// order. The explicit fourth word keeps the Int32/Index32 lane at a 16-byte
/// stride and fills existing tail padding in the Int64/Index32 lane.
template <typename Int, typename Index> struct delaunay_site {
  Int x;
  Int y;
  Index output;
  Index storage_pad;

  auto operator[](std::size_t axis) const -> Int { return axis == 0 ? x : y; }
};

} // namespace tf::topology::cdt
