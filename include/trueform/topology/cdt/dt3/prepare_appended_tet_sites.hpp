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
#include "./advance_tet_name.hpp"
#include "./clear_tetrahedralization_cells.hpp"
#include "./state_tet_deep_site.hpp"
#include "./tet_ingested_position.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// Enter `sites` into the lane and the input map, each standing where
/// @ref tet_ingested_position places it: its floor in the lane, its row
/// where it stands below the frame (@ref state_tet_deep_site).
template <bool Parallel, typename Owner, typename Sites>
auto prepare_appended_tet_sites(Owner &owner, const Sites &sites) -> bool {
  using Index = typename Owner::index_type;
  const auto count = sites.size(), first = owner._floors.size();
  const auto input = owner._index_map.f().size();
  if (count > std::size_t(Owner::max_sites) - first ||
      count > std::size_t(Owner::max_sites) - input) {
    clear_tetrahedralization_cells(owner);
    owner._refusal = tf::tetrahedralization_refusal::index_capacity;
    return false;
  }
  owner._order.allocate(count);
  owner._index_map.f().reallocate(input + count);
  owner._index_map.kept_ids().reallocate(first + count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto &site = sites[i];
    const auto position = tet_ingested_position(site);
    owner._floors.push_back({site.id, position.floor});
    state_tet_deep_site(owner, Index(site.id), position);
    owner._next_name = advance_tet_name(owner._next_name, site.id);
    owner._order[i] = Index(first + i);
    owner._index_map.f()[input + i] = Index(first + i);
    owner._index_map.kept_ids()[first + i] = Index(input + i);
  }
  if (owner._record_site_cells)
    owner._site_cells.resize(owner._floors.size(), Owner::none);
  return true;
}

} // namespace tf::topology::cdt::dt3
