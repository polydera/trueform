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

namespace tf::topology::cdt {

template <typename Owner>
auto clear_constrained_delaunay_refinement(Owner &owner) -> void {
  owner._cdt.clear();
  owner._ip.clear();
  owner._dp.clear();
  owner._t.clear();
  owner._label.clear();
  owner._segments.clear();
  owner._con_nbr.clear();
  owner._queue.clear();
  owner._cavity.clear();
  owner._lawson.clear();
  owner._pending.clear();
  owner._generation.clear();
  owner._splits.clear();
  owner._split_offsets.clear();
  owner._representative_split_offsets.clear();
  owner._expanded_splits.clear();
  owner._constraint_split_parent.clear();
  owner._constraint_split_reversed.clear();
  owner._constraint_aliases.clear();
  owner._constraint_alias_blocks.clear();
  owner._min_quality = 0.0;
  owner._scale = 1;
  owner._offx = 0;
  owner._offy = 0;
  owner._n_input_points = 0;
  owner._n_input_edges = 0;
  owner._ok = false;
}

} // namespace tf::topology::cdt
