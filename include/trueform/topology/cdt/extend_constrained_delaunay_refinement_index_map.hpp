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

/// State the refinement's own sites in the seed's index map. A Steiner point
/// is a vertex the build created, so it takes the sentinel `f().size()` the
/// seed already states for recovery's crossings, and `kept_ids()` speaks for
/// every output point.
template <typename Owner>
auto extend_constrained_delaunay_refinement_index_map(Owner &owner) -> void {
  using Index = typename Owner::index_type;

  auto &map = owner._cdt.index_map();
  const auto n_seed = map.kept_ids().size();
  const auto created = static_cast<Index>(map.f().size());
  map.kept_ids().reallocate(owner._ip.size());
  for (auto output = n_seed; output < owner._ip.size(); ++output)
    map.kept_ids()[output] = created;
}

} // namespace tf::topology::cdt
