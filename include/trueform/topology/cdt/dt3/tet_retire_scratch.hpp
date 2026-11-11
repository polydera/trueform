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
#include "../../../core/buffer.hpp"
#include "./recovery/tet_stamps.hpp"
#include "./recovery/tet_star_stamps.hpp"

namespace tf::topology::cdt::dt3 {

/// The lists a retirement of faces walks on (@ref retire_tet_faces): `sites`
/// the sites it reached, one mark per site, `stars` the star marks of the
/// cells it gathered, `queue` the sites in the order reached and `star` the
/// cells about the one being read.
template <typename Index> struct tet_retire_scratch {
  tet_stamps sites;
  tet_star_stamps stars;
  tf::buffer<Index> queue;
  tf::buffer<Index> star;
};

} // namespace tf::topology::cdt::dt3
