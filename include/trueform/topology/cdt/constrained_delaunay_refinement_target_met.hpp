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
#include "./queue_constrained_delaunay_refinement_encroachments.hpp"
#include <cstddef>

namespace tf::topology::cdt {

/// Whether the refinement's target holds on the triangulation that stands.
///
/// The encroachment half is asked of the one producer of that fact, so an
/// exempt constraint-connected encroacher and a constraint at the split cap
/// are read here exactly as the refinement itself reads them; the
/// obligations it proposes are scratch and are dropped again.
template <typename Owner>
auto constrained_delaunay_refinement_target_met(Owner &owner) -> bool {
  using Index = typename Owner::index_type;
  bool met = true;
  for (Index face = 0; met && face < Index(owner._t.size()); ++face) {
    if (owner._label[std::size_t(face)] % 2 != 1)
      continue;
    if (owner.quality(face) < owner._min_quality)
      met = false;
    else if (queue_constrained_delaunay_refinement_encroachments(owner, face))
      met = false;
  }
  owner._pending.clear();
  return met;
}

} // namespace tf::topology::cdt
