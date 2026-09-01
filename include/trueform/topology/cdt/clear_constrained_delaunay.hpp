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
#include "./clear_constrained_delaunay_products.hpp"

namespace tf::topology::cdt {

template <typename Owner>
auto clear_constrained_delaunay(Owner &owner) -> void {
  owner.clear_delaunay_topology();
  clear_constrained_delaunay_products(owner);
  owner._scratch_face_offsets.clear();
  owner._scratch_stack.clear();
  owner._points.clear();
  owner._v_first_edge.clear();
  owner._flip_stack.clear();
  owner._deleted_edges.clear();
  owner._vertices_cw.clear();
  owner._vertices_ccw.clear();
  owner._retriangulation_stack.clear();
  owner._constraint_input_ids.clear();
  owner._constraint_provenance.clear();
  owner._allow_crossing_splits = false;
  owner._track_constraint_provenance = false;
  owner._incremental_crossings.clear();
  owner._constraint_collisions.clear();
  owner._final_constraint_edges.clear();
  owner._final_is_boundary.clear();
  owner._has_prepared_input_index_map = false;
  owner._last_edge = Owner::none;
  owner._locate_hint = Owner::none;
}

} // namespace tf::topology::cdt
