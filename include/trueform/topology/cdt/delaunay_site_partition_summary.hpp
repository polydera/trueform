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
#include <cstdint>

namespace tf::topology::cdt {

template <typename Int> struct delaunay_site_partition_summary {
  Int minimum_x;
  Int minimum_y;
  Int maximum_x;
  Int maximum_y;
  std::uint32_t largest_morton_key;
};

} // namespace tf::topology::cdt
