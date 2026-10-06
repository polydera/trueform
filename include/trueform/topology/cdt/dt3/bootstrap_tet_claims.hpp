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
#include "./compact_tetrahedralization.hpp"
#include "./insert_tetrahedralization_site.hpp"
#include "./order_tet_site_band.hpp"
#include "./tet_cavity_edge_table.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

template <typename Owner>
auto bootstrap_tet_claims(Owner &owner, std::size_t prefix) -> bool {
  using Index = typename Owner::index_type;
  const std::array<Index, 4> seeded{owner._corners[0][0], owner._corners[0][1],
                                    owner._corners[0][2], owner._corners[0][3]};
  const auto is_seed = [&](Index site) {
    return std::find(seeded.begin(), seeded.end(), site) != seeded.end();
  };
  for (auto end = prefix; end != 0; end >>= 1)
    order_tet_site_band(owner, end >> 1, end);
  tet_cavity_edge_table<Index> edges{};
  for (std::size_t i = 0; i < prefix; ++i)
    if (!is_seed(owner._order[i]) &&
        !insert_tetrahedralization_site(owner, owner._order[i], edges))
      return false;
  compact_tetrahedralization<true>(owner);
  auto last = std::remove_if(owner._order.begin() + prefix, owner._order.end(),
                             is_seed);
  owner._order.reallocate(std::size_t(last - owner._order.begin()));
  return true;
}

} // namespace tf::topology::cdt::dt3
