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
#include "./clear_tetrahedralization.hpp"
#include "./clear_tetrahedralization_cells.hpp"
#include "./compact_tetrahedralization.hpp"
#include "./insert_tetrahedralization_site.hpp"
#include "./prepare_tetrahedralization.hpp"
#include "./tet_execution_tuning.hpp"
#include "./tet_insertion.hpp"
#include "./tet_site_position.hpp"
#include "./tetrahedralization_owner.hpp"
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Build all sites serially and publish the canonical complex; `Parallel`
/// selects the executor of that publication alone.
template <bool Parallel = false, typename Owner, typename Sites>
auto build_tetrahedralization(Owner &owner, const Sites &sites) -> bool {
  using Index = typename Owner::index_type;
  if (!prepare_tetrahedralization(owner, sites))
    return false;

  const std::array<Index, 4> seeded{owner._corners[0][0], owner._corners[0][1],
                                    owner._corners[0][2], owner._corners[0][3]};
  if (tet_site_count(owner) > seeded.size()) {
    const std::size_t rows =
        tet_execution_tuning::reserve_per_site * tet_site_count(owner);
    owner._corners.reserve(rows);
    owner._neighbors.reserve(rows);
    owner._states.reserve(rows);
    for (std::size_t i = 0; i < owner._order.size(); ++i) {
      const Index site = owner._order[i];
      if (site == seeded[0] || site == seeded[1] || site == seeded[2] ||
          site == seeded[3])
        continue;
      if (insert_tetrahedralization_site(owner, site) !=
          tet_insertion::placed) {
        clear_tetrahedralization_cells(owner);
        owner._refusal = tf::tetrahedralization_refusal::index_capacity;
        return false;
      }
    }
  }

  compact_tetrahedralization<Parallel, true>(owner);
  return true;
}

} // namespace tf::topology::cdt::dt3
