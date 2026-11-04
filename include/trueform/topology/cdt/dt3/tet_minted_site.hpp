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

namespace tf::topology::cdt::dt3 {

/// A site a tier minted, handed to a build or an append with its name.
template <typename Index, typename Coord> struct tet_minted_site {
  Index id;
  tet_position<Coord> position;
};

} // namespace tf::topology::cdt::dt3
