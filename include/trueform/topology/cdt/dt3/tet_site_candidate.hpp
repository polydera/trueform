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
#include "./tet_position.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

template <typename Coord> struct tet_site_candidate {
  tet_position<Coord> position;
  std::size_t ticket;
};

} // namespace tf::topology::cdt::dt3
