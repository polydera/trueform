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
#include "./encode_tet_site_band.hpp"
#include "./measure_tet_site_domain.hpp"
#include "./tet_site_priority.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

template <bool Parallel, typename Owner, typename Sites>
auto prepare_appended_tet_sites(Owner &owner, const Sites &sites) -> bool {
  using Index = typename Owner::index_type;
  const auto count = sites.size(), first = owner._sites.size();
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
    owner._sites.push_back(sites[i]);
    owner._next_name = advance_tet_name(owner._next_name, sites[i].id);
    owner._order[i] = Index(first + i);
    owner._index_map.f()[input + i] = Index(first + i);
    owner._index_map.kept_ids()[first + i] = Index(input + i);
  }
  owner._priorities.reallocate(owner._sites.size());
  for (std::size_t i = first; i < owner._sites.size(); ++i)
    owner._priorities[i] = tet_site_priority(std::uint64_t(owner._sites[i].id));
  owner._domain =
      owner._domain.merged(measure_tet_site_domain<Parallel>(owner, first));
  encode_tet_site_band<Parallel>(owner, first, owner._sites.size(),
                                 owner._domain);
  return true;
}

} // namespace tf::topology::cdt::dt3
