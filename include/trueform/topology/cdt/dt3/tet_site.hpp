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
#include "../../../exact/vertex.hpp"

namespace tf::topology::cdt::dt3 {

/// A weld record: exact position, canonical name and original input slot.
template <typename Index, typename Coord> struct tet_site {
  tf::exact::pt3<Coord> pt;
  Index id;
  Index input;
};

} // namespace tf::topology::cdt::dt3
