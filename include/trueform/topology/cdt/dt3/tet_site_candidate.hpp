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
#include "../../../core/point.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

template <typename Coord> struct tet_site_candidate {
  tf::point<Coord, 3> pt;
  std::size_t ticket;
};

} // namespace tf::topology::cdt::dt3
