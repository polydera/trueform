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
#include "./tet_face_census.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// What constraint recovery did: the rounds phase S ran, the splits its
/// corner shells proposed, the segments it refused and the facets its
/// restoration left not locally Delaunay, every face route of phase F, and
/// the segment rows flips recovered without and with the full search, the
/// rows that split, and the rings a hull cell kept from flipping.
struct tet_constraint_census {
  std::size_t segment_rounds = 0;
  std::size_t shell_splits = 0;
  std::size_t segments_refused = 0;
  std::size_t unrestored_facets = 0;
  tet_face_census faces;
  std::size_t flip_recovered = 0;
  std::size_t search_recovered = 0;
  std::size_t split_rows = 0;
  std::size_t hull_barred = 0;
};

} // namespace tf::topology::cdt::dt3
