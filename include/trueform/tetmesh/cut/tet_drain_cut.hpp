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
#include <cstddef>

namespace tf::tetmesh {

/// What one drain's cut did (@ref cut_tet_pairs): the triangles it traced
/// and those beyond them it split, the triangles traced, the points it
/// created, the facets it retired — those holding a traced triangle and
/// those only a split beyond reached — and the triangles the planar lane
/// refused.
struct tet_drain_cut {
  std::size_t delta = 0;
  std::size_t cut = 0;
  std::size_t created = 0;
  std::size_t retired_crossing = 0;
  std::size_t retired_neighbour = 0;
  std::size_t failed = 0;
};

} // namespace tf::tetmesh
