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
#include "../../../../core/buffer.hpp"

namespace tf::topology::cdt::dt3 {

/// The constraints a cut made due: the subsegment `rows` phase S recovers
/// and the `faces` phase F covers, each joining the queue the phase drains.
template <typename Index> struct tet_crossing_due {
  tf::buffer<Index> rows;
  tf::buffer<Index> faces;
};

} // namespace tf::topology::cdt::dt3
