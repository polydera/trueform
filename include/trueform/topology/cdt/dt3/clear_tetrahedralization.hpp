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
#include "../../tetrahedralization_refusal.hpp"
#include "../../tetrahedralization_stats.hpp"
#include "./tetrahedralization_owner.hpp"

namespace tf::topology::cdt::dt3 {

/// Drop every product and build state, keeping the capacity a later build
/// would ask for again.
template <typename Owner> auto clear_tetrahedralization(Owner &owner) -> void {
  using Index = typename Owner::index_type;
  owner._sites.clear();
  owner._site_scratch.clear();
  owner._index_map.f().clear();
  owner._index_map.kept_ids().clear();
  owner._order.clear();
  owner._priorities.clear();
  owner._keys.clear();
  owner._key_records.clear();
  owner._key_scratch.clear();
  owner._counts.clear();
  owner._block_offsets.clear();
  owner._corners.clear();
  owner._neighbors.clear();
  owner._states.clear();
  owner._corner_scratch.clear();
  owner._neighbor_scratch.clear();
  owner._state_scratch.clear();
  owner._compaction.clear();
  owner._cell_keys.clear();
  owner._cavity.clear();
  owner._boundary.clear();
  owner._sides.clear();
  owner._hint = Index(0);
  owner._next_name = 0;
  owner._n_dead = 0;
  owner._n_finite = 0;
  owner._refusal = tf::tetrahedralization_refusal::none;
  owner._stats = tf::tetrahedralization_stats{};
}

} // namespace tf::topology::cdt::dt3
