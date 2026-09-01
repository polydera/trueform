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

/// Constraint intervals are oriented per half-edge: `first` belongs to its
/// origin and `second` to its target. The opposite half-edge therefore reads
/// the same provenance with the interval reversed.
template <typename Owner>
auto note_constrained_delaunay_constraint_provenance(
    Owner &owner, typename Owner::index_type edge,
    typename Owner::index_type input_id, typename Owner::param_type first,
    typename Owner::param_type second) -> void {
  if (input_id == Owner::none)
    return;
  owner._constraint_provenance.prepare(owner._edges.size(), Owner::none);
  owner._constraint_provenance.set(edge, input_id, first, second);
}

} // namespace tf::topology::cdt
